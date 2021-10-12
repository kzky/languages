import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Any
import open3d as o3d
import torch

from render_utils import Camera, Screen, FitResolutionGate, MaterialType, TriangleMesh, ShadowMap, to_matrix4x4, DistantLight, PointLight

inch_to_mm = 25.4

class FitResolutionGate(Enum):
    FILL = 0
    OVER_SCREEN = 1


@dataclass
class Camera:
    # Intrinsic Parameter
    focal_length: float = 20              # in mm 
    film_aperture_width: float = 0.980    # 35mm Full Aperture in inches
    film_aperture_height: float = 0.735   # 35mm Full Aperture in inches
    z_near: float = 1
    z_far: float = 1000
    image_width: int = 640
    image_height: int = 480
    k_fit_resolution_gate: FitResolutionGate = FitResolutionGate.OVER_SCREEN
    # Extrinsic Parameter
    camera_to_world: Any = None

    def field_of_view(self, film_aperture):
        return 2 * 180 / np.pi * np.arctan((film_aperture * inch_to_mm / 2) / self.focal_length)

    def compute_screen(self):
        # fovs, top, right
        fov_h = self.field_of_view(self.film_aperture_width)
        fov_v = self.field_of_view(self.film_aperture_height)
        top = ((self.film_aperture_height * inch_to_mm / 2) / self.focal_length) * self.z_near
        right = ((self.film_aperture_width * inch_to_mm / 2) / self.focal_length) * self.z_near

        # Scale if necessary
        x_scale = 1.0
        y_scale = 1.0
        film_aspect_ratio = self.compute_film_aspect_ratio()
        image_aspect_ratio = self.compute_image_aspect_ratio()
        if self.k_fit_resolution_gate == FitResolutionGate.FILL:
            if film_aspect_ratio > image_aspect_ratio:
                x_scale *= image_aspect_ratio / film_aspect_ratio
            else:
                y_scale *= film_aspect_ratio / image_aspect_ratio
        elif self.k_fit_resolution_gate == FitResolutionGate.OVER_SCREEN:
            if film_aspect_ratio > image_aspect_ratio:
                y_scale *= film_aspect_ratio / image_aspect_ratio
            else:
                x_scale *= image_aspect_ratio / film_aspect_ratio
        else:
            raise ValueError(f"k_fit_resolution_gate(={k_fit_resolution_gate}) must be in {list(FitResolutionGate)}.")
        top *= y_scale
        right *= x_scale
        bottom = -top
        left = -right

        # Screen (Image Plane)
        #screen = Screen(left=left, right=right, top=top, bottom=bottom)
        screen = Screen(left=left, right=right, top=top, bottom=bottom)
        return screen

    def compute_film_aspect_ratio(self):
        return 1.0 * self.film_aperture_width / self.film_aperture_height

    def compute_image_aspect_ratio(self):
        return 1.0 * self.image_width / self.image_height

    def compute_projection_matrix(self, screen):
        """Project v to the homogeneous clipping space
        """
        l = screen.left
        r = screen.right
        t = screen.top
        b = screen.bottom
        n = self.z_near
        f = self.z_far
        P = np.asarray([[2 * n / (r - l), 0, 0, 0],
                        [0, 2 * n / (t - b), 0, 0],
                        [(r + l) / (r - l), (t + b) / (t - b), - (f + n) / (f - n), -1.0],
                        [0, 0, - (2 * f * n) / (f - n), 0]])
        return P

    
@dataclass
class Screen:
    left: float = 0
    right: float = 0
    top: float = 0
    bottom: float = 0


"""
Conversion functions: Note that the coordinates are the homogeneous coordinates.
"""
def world_to_camera(vertices, world_to_camera):
    return np.hstack([vertices, np.ones(len(vertices))[:, np.newaxis]]).dot(world_to_camera)[:, :3]


def convert_world_to_camera(vertex, world_to_camera):
    return vertex.dot(world_to_camera)


def convert_camera_to_screen(vertex0, camera, screen):
    vertex1 = np.ones(4)
    vertex1[0] = camera.z_near * vertex0[0] / -vertex0[2]
    vertex1[1] = camera.z_near * vertex0[1] / -vertex0[2]
    vertex1[2] = -vertex0[2]  # since this is used in z-buffer
    return vertex1


def convert_screen_to_raster(vertex0, camera, screen):
    # to NDC [-1, 1]
    vertex1 = np.ones(4)
    r = screen.right
    l = screen.left
    t = screen.top
    b = screen .bottom
    vertex1[0] = 2 * vertex0[0] / (r - l) - (r + l) / (r - l)
    vertex1[1] = 2 * vertex0[1] / (t - b) - (t + b) / (t - b)
    vertex1[2] = vertex0[2]
    # to Raster
    vertex1[0] = (vertex1[0] + 1) / 2 * camera.image_width
    vertex1[1] = (1 - vertex1[1]) / 2 * camera.image_height
    return vertex1


def convert_cammera_to_raster(vertex1, camera, screen):
    vertex2 = convert_camera_to_screen(vertex1, camera, screen)
    vertex3 = convert_screen_to_raster(vertex2, camera, screen)
    return vertex3


def convert_world_to_raster(vertex0, world_to_camera, camera, screen):
    vertex1 = convert_world_to_camera(vertex0, world_to_camera)
    vertex2 = convert_camera_to_screen(vertex1, camera, screen)
    vertex3 = convert_screen_to_raster(vertex2, camera, screen)
    return vertex3


def convert_to_homogeneous_clipping(v, P):
    return v.dot(P)


def clipping_condition(v):
    x, y, z, w = v[0], v[1], v[2], v[3]
    cond = True
    cond *= w > 0
    cond *= (x >= -w) and (x <= w)
    cond *= (y >= -w) and (y <= w)
    cond *= (z >= -w) and (z <= w)
    return cond


def convert_to_unit_cube(v):
    x, y, z, w = v[0], v[1], v[2], v[3]
    return np.array([x / w, y / w, z / w])


def convert_to_raster(v, camera):
    x, y, z = v[0], v[1], v[2]
    return np.array([(x + 1) * 0.5 * camera.image_width,
                     (1 - (y + 1) * 0.5) * camera.image_height,
                     z])


def normalize(vertex):
    assert len(vertex) == 3
    vertex = vertex / np.sqrt(np.sum(vertex ** 2))
    return np.array([vertex[0], vertex[1], vertex[2]])


def compute_from_to_matrix(F, T, tmp=np.array([0.0, 1.0, 0.0])):
    if len(F) == 4 or len(T) == 4:
        F = F[:3]
        T = T[:3]
    forward = normalize(F - T)
    right = np.cross(normalize(tmp), forward)
    up = np.cross(forward, right)
    M = np.stack([right, up, forward, F])
    H = np.array([0.0, 0.0, 0.0, 1.0])[:, np.newaxis]
    M = np.concatenate([M, H], axis=1)
    return M


def look_at(F, T, tmp=np.array([0.0, 1.0, 0.0])):
    """Return a colunm-major transformation matrix
    """
    if len(F) == 4 or len(T) == 4:
        F = F[:3]
        T = T[:3]
    forward = normalize(F - T)
    right = np.cross(normalize(tmp), forward)
    up = np.cross(forward, right)
    M = np.stack([right, up, forward, F])
    H = np.array([0.0, 0.0, 0.0, 1.0])[:, np.newaxis]
    M = np.concatenate([M, H], axis=1)
    return M.T


def homogeneous(vertex):
    assert len(vertex) == 3
    return np.array([vertex[0], vertex[1], vertex[2], 1.0])


def edge_function(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])


def compute_normal_2d(u, v):
    R = rotation_matrix_2d(np.pi / 2)
    n = R.dot((v - u)[:2].T)
    n = n / np.sqrt(np.sum(n ** 2))
    return n


def compute_normal_and_bias_2d(u, v):
    n = compute_normal_2d(u, v)
    d = -n.dot(u[:2])
    return n, d


def inv_2d_mat(M):
    det = 1.0 / ((M[0, 0] * M[1, 1]) - (M[0, 1] * M[1, 0]))
    return np.asarray([[M[1, 1], -M[0, 1]],
                       [-M[1, 0], M[0, 0]]]) * det


def compute_conservative_xy(v0, v1, v2, shift=0):
    n0, d0 = compute_normal_and_bias_2d(v1, v2)
    n1, d1 = compute_normal_and_bias_2d(v2, v0)
    n2, d2 = compute_normal_and_bias_2d(v0, v1)
    c = shift * np.ones(2)
    d0 -= c.dot(np.abs(n0))
    d1 -= c.dot(np.abs(n1))
    d2 -= c.dot(np.abs(n2))
    plane0 = np.asarray([n0[0], n0[1], d0])
    plane1 = np.asarray([n1[0], n1[1], d1])
    plane2 = np.asarray([n2[0], n2[1], d2])
    v0_xy = np.cross(plane1, plane2)
    v1_xy = np.cross(plane2, plane0)
    v2_xy = np.cross(plane0, plane1)
    v0_xy = v0_xy[:2] / v0_xy[2]
    v1_xy = v1_xy[:2] / v1_xy[2]
    v2_xy = v2_xy[:2] / v2_xy[2]
    return v0_xy, v1_xy, v2_xy


def rotation_matrix_2d(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R
    

def rotation_matrix_3d(alpha, beta, gamma):
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha), np.cos(alpha),  0],
                   [0,             0,              1]])
    Ry = np.array([[np.cos(beta),  0, np.sin(beta)],
                   [0,             1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rx = np.array([[1, 0,             0],
                   [0, np.cos(gamma), -np.sin(gamma)],
                   [0, np.sin(gamma), np.cos(gamma)]])
    R = Rz.dot(Ry.dot(Rx))
    R = np.concatenate([R, np.array([0, 0, 0])[np.newaxis, :]], axis=0)
    R = np.concatenate([R, np.array([0, 0, 0, 1])[:, np.newaxis]], axis=1)
    return R


def to_dtypes(np_vertices, np_triangles, np_vertex_colors):
    return np_vertices.astype(np.float32), np_triangles.astype(np.int32), np_vertex_colors.astype(np.float32)

def create_box(W=1.0, H=1.0, D=1.0, scale=1.0, bias=0.0, vertex_colors=None):
    box = o3d.geometry.TriangleMesh.create_box(W, H, D)
    np_vertices = scale * (np.asarray(box.vertices) + [-0.5, -0.5, -0.5]) + bias
    np_triangles = np.asarray(box.triangles)
    np_vertex_colors = np.linspace(0, 1, len(np_vertices) * 3).reshape((len(np_vertices), 3))
    return to_dtypes(np_vertices, np_triangles, np_vertex_colors)


def create_sphere(radius=1.0, resolution=20, scale=1.0, bias=0.0, vertex_colors=None):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
    np_vertices = scale * np.asarray(sphere.vertices) + bias
    np_triangles = np.asarray(sphere.triangles)
    np_vertex_colors = np.linspace(1, 0, len(np_vertices) * 3).reshape((len(np_vertices), 3))
    return to_dtypes(np_vertices, np_triangles, np_vertex_colors)


def create_cone(radius=1.0, height=2.0, resolution=20, split=1, scale=1.0, bias=0.0, vertex_colors=None):
    cone = o3d.geometry.TriangleMesh.create_cone(radius, height, resolution, split)
    np_vertices = scale * np.asarray(cone.vertices) + bias
    np_triangles = np.asarray(cone.triangles)
    np_vertex_colors = np.linspace(0, 1, len(np_vertices) * 3).reshape((len(np_vertices), 3))
    return to_dtypes(np_vertices, np_triangles, np_vertex_colors)


def create_coord_frame(scale=1.0, bias=0.0, vertex_colors=None):
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    np_vertices = scale * np.asarray(coord_frame.vertices) + bias
    np_triangles = np.asarray(coord_frame.triangles)
    np_vertex_colors = np.asarray(coord_frame.vertex_colors) if vertex_colors is None else vertex_colors
    return to_dtypes(np_vertices, np_triangles, np_vertex_colors)


def create_plane(scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices = np.asarray([[1, -0.2, 1], 
                              [1, -0.2, -1],
                              [-1, -0.2, 1],
                              [-1, -0.2, -1]]) * scale + bias
    np_triangles = np.asarray([[1, 2, 0],
                               [2, 1, 3]])
    np_vertex_colors = np.linspace(0.5, 0.5, len(np_vertices) * 3).reshape((len(np_vertices), 3)) \
      if vertex_colors is None else vertex_colors
    return to_dtypes(np_vertices, np_triangles, np_vertex_colors)


create_xy_plane = create_plane


def create_zy_plane(scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices = np.asarray([[-0.2, 1, 1], 
                              [-0.2, 1, -1],
                              [-0.2, -1, 1],
                              [-0.2, -1, -1]]) * scale + bias
    np_triangles = np.asarray([[0, 2, 1],
                               [2, 3, 1]])
    np_vertex_colors = np.linspace(0.5, 0.5, len(np_vertices) * 3).reshape((len(np_vertices), 3)) \
      if vertex_colors is None else vertex_colors
    return to_dtypes(np_vertices, np_triangles, np_vertex_colors)


def create_yx_plane(scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices = np.asarray([[1, 1, -0.2], 
                              [1, -1, -0.2],
                              [-1, 1, -0.2],
                              [-1, -1, -0.2]]) * scale + bias
    np_triangles = np.asarray([[0, 2, 1],
                               [2, 3, 1]])
    np_vertex_colors = np.linspace(0.5, 0.5, len(np_vertices) * 3).reshape((len(np_vertices), 3)) \
      if vertex_colors is None else vertex_colors
    return to_dtypes(np_vertices, np_triangles, np_vertex_colors)


class TriangleMeshPy:
    
    def __init__(self, vertices, triangles, vertex_colors):
        self.vertices = torch.from_numpy(vertices)
        self.triangles = torch.from_numpy(triangles)
        self.vertex_colors = torch.from_numpy(vertex_colors)


    def __add__(self, rhs):
        vertices = np.concatenate([self.vertices.data, rhs.vertices.data])
        triangles = np.concatenate([self.triangles.data, rhs.triangles.data + self.vertices.shape[0]])
        vertex_colors = np.concatenate([self.vertex_colors.data, rhs.vertex_colors.data])
        return TriangleMeshPy(vertices, triangles, vertex_colors)


    def create_triangle_mesh(self, device):
        #TODO: hold the references of tensor in cuda
        tmesh = TriangleMesh(self.triangles.shape[0], self.vertices.shape[0],
                             self.triangles.to(device).data_ptr(), 
                             self.vertices.to(device).data_ptr(), 
                             self.vertex_colors.to(device).data_ptr(), 
                             MaterialType.DIFFUSE)
        return tmesh


def create_box_meshpy(W=1.0, H=1.0, D=1.0, scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices, np_triangles, np_vertex_colors = create_box(W, H, D, scale, bias, vertex_colors)
    tmeshpy = TriangleMeshPy(np_vertices, np_triangles, np_vertex_colors)
    return tmeshpy


def create_sphere_meshpy(radius=1.0, resolution=20, scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices, np_triangles, np_vertex_colors = create_sphere(radius, resolution, scale, bias, vertex_colors)
    tmeshpy = TriangleMeshPy(np_vertices, np_triangles, np_vertex_colors)
    return tmeshpy


def create_coord_frame_meshpy(scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices, np_triangles, np_vertex_colors = create_coord_frame(scale, bias, vertex_colors)
    tmeshpy = TriangleMeshPy(np_vertices, np_triangles, np_vertex_colors)
    return tmeshpy


def create_plane_meshpy(scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices, np_triangles, np_vertex_colors = create_plane(scale, bias, vertex_colors)
    tmeshpy = TriangleMeshPy(np_vertices, np_triangles, np_vertex_colors)
    return tmeshpy


create_xz_plane_meshpy = create_plane_meshpy


def create_plane_meshpy(scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices, np_triangles, np_vertex_colors = create_plane(scale, bias, vertex_colors)
    tmeshpy = TriangleMeshPy(np_vertices, np_triangles, np_vertex_colors)
    return tmeshpy


def create_zy_plane_meshpy(scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices, np_triangles, np_vertex_colors = create_zy_plane(scale, bias, vertex_colors)
    tmeshpy = TriangleMeshPy(np_vertices, np_triangles, np_vertex_colors)
    return tmeshpy


def create_yx_plane_meshpy(scale=1.0, bias=0.0, vertex_colors=None):
    np_vertices, np_triangles, np_vertex_colors = create_yx_plane(scale, bias, vertex_colors)
    tmeshpy = TriangleMeshPy(np_vertices, np_triangles, np_vertex_colors)
    return tmeshpy



class ShadowMapPy:


    def __init__(self, H, W, dlight, M, P):
        self.H = H
        self.W = W
        self.dlight = dlight
        self.M = M
        self.P = P
        self.z_buffer = torch.zeros((H, W), dtype=torch.float32)
        self.z_buffer.fill_(1e24)


    def create_shadow_map(self, device):
        shadow_map = ShadowMap(self.z_buffer.to(device).data_ptr(), 
                               self.H, self.W,
                               self.dlight, self.M, self.P)
        return shadow_map
    
        

def main():
    camera = Camera()
    screen = camera.compute_screen()
    print("Screen")
    print(screen)
    view = np.array([0.0, 0.0, 0.0])
    objc = np.array([5, 3, 4])
    M = compute_from_to_matrix(view, objc)
    print("World-to-Camera Matrix")
    print(M)
    world_to_camera = np.linalg.inv(M)
    print("Camera-to-World Matrix")
    print(world_to_camera)

    print("In World Coordinate System")
    print(objc)
    print("In Camera Coordinate System")
    print(convert_world_to_camera(homogeneous(objc), world_to_camera))
    print("In Raster Coordinate System")
    print(convert_world_to_raster(homogeneous(objc), world_to_camera, camera, screen))
    


if __name__ == '__main__':
    main()
