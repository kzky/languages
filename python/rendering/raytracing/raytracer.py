import numpy as np
import argparse
import os
import open3d as o3d
from itertools import product
from tqdm import tqdm
from enum import Enum
from collections import namedtuple
from dataclasses import dataclass
import cv2
import itertools

import sys
#sys.path.append('../')
from world import *


class MaterialType(Enum):
    DIFFUSE = 0
    REFLECTION_AND_REFRACTION = 1
    REFLECTION = 2


@dataclass
class IntersectInfo:
    idx_obj: int
    idx_near: int
    t: float
    bc_coords: np.ndarray
    normal: np.ndarray = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    material_type: MaterialType = MaterialType.DIFFUSE


def render(meshes, camera, screen, args):
    # Settings
    fov = camera.field_of_view(camera.film_aperture_height)
    scale = np.tan(fov * 0.5 * np.pi / 180)
    image_aspect_ratio = camera.compute_image_aspect_ratio()
    H = camera.image_height
    W = camera.image_width
    M_c2w = camera.camera_to_world
    # Image buffer
    image = np.ones([3, args.image_height, args.image_width], dtype=np.uint8) * 255
    # Pixel loop
    o = np.asarray([0, 0, 0, 1])
    o = o.dot(M_c2w)[:3]
    for j, i in tqdm(itertools.product(range(camera.image_height), range(camera.image_width))):
        print(j, i, H * W)
        # generate primary ray
        x = (2 * (i + 0.5) / W - 1) * image_aspect_ratio * scale
        y = (1 - 2 * (j + 0.5) / H) * scale
        p = np.asarray([x, y, -1, 1])
        p = p.dot(M_c2w)[:3]
        d = p - o
        d = normalize(d)
        color = cast_ray(meshes, o, d, 0, args)
        image[:, j, i] = color
    return image


def cast_ray(meshes, o, d, depth, args):
    """
    Args:
      o (numpy.ndarray): Origin of the ray
      d (numpy.ndarray): Direction of the ray
      meshes (open3d.utility.Vector3dVector): mesh in camera space
      depth (int): number of recursions
    """
    if (depth > args.depth):
        return args.background_color

    color = args.background_color

    intersect_info = trace(meshes, o, d, args)
    if intersect_info.idx_obj == -1:
        return color

    # TODO: temporary
    mesh = meshes[intersect_info.idx_obj]
    triangle = mesh.triangles[intersect_info.idx_near]
    u, v, w = intersect_info.bc_coords
    c0 = mesh.vertex_colors[triangle[0]]
    c1 = mesh.vertex_colors[triangle[2]]
    c2 = mesh.vertex_colors[triangle[1]]
    color = u * c0 + v * c1 + w * c2

    ## TODO: finally this is needed
    p = o + d * intersect_info.t

    ## if intersect_info.material == MaterialType.REFLECTION_AND_REFRACTION:
    ##     # reflection
    ##     o = ...
    ##     d = ...
    ##     color = cast_ray(meshes, o, d, depth + 1)
    ##     # reflection
    ##     o = ...
    ##     d = ...
    ##     color = cast_ray(meshes, o, d, depth + 1)
    ## elif intersect_info.material == MaterialType.REFLECTION:
    ##     # reflection
    ##     o = ...
    ##     d = ...
    ##     color = cast_ray(meshes, o, d, depth + 1)
    ## else intersect_info.material == MaterialType.DIFFUSE:
    ##     # shadow ray
    ##     o = ...
    ##     d = ...
    ##     color = ...

    return color
    
    
def trace(meshes, o, d, args):
    """
    Args:
      o (numpy.ndarray): Origin of the ray
      d (numpy.ndarray): Direction of the ray
      meshes (open3d.utility.Vector3dVector): mesh in camera space
    """
    idx_obj = -1
    idx_near = -1
    tnear = np.finfo(np.float32).max    
    for i in range(len(meshes)):
        triangles = mesh.triangles
        for j in range(len(triangles)):
            triangle = triangles[j]
            # counter-clockwise
            v0 = mesh.vertices[triangle[0]]
            v1 = mesh.vertices[triangle[2]]
            v2 = mesh.vertices[triangle[1]]
            t, u, v = intersect(v0, v1, v2, o, d, args)
            if t < tnear:
                idx_obj = i
                idx_near = j
                tnear = t
    intersect_info = IntersectInfo(idx_obj=idx_obj, idx_near=idx_near, t=tnear,
                                   bc_coords=np.asarray([u, v, 1 - u - v]))
    return intersect_info
    

def intersect(v0, v1, v2, o, d, args):
    # MT method
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(d, v0v2)
    det = v0v1.dot(pvec)
    ## if det < args.epsilon:  # back-face culling
    ##     return -1, -1, 1
    if abs(det) < args.epsilon:  # ray is parallel planar
        return -1, -1, -1
    inv_det = 1.0 / det

    tvec = o - v0
    u = tvec.dot(pvec) * inv_det
    if u < 0 or u > 1:
        return -1, -1, -1

    qvec = np.cross(tvec, v0v1)
    v = d.dot(qvec) * inv_det
    if v < 0 or (u + v) > 1:
        return -1, -1, -1
    
    t = v0v2.dot(qvec) * inv_det
    return t, u, v
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", "-f", type=str, required=True)
    parser.add_argument("--focal-length", "-fl", type=float, default=20)
    parser.add_argument("--film-aperture-width", "-faw", type=float, default=0.980)
    parser.add_argument("--film-aperture-height", "-fah", type=float, default=0.735)
    parser.add_argument("--z-near", "-zn", type=float, default=1)
    parser.add_argument("--z-far", "-zf", type=float, default=1000)
    parser.add_argument("--image-width", "-iw", type=int, default=640)
    parser.add_argument("--image-height", "-ih", type=int, default=480)
    parser.add_argument('--camera-position', "-cp", nargs='+', type=float,
                        default=np.array([0.0, 0.0, 0.0]))
    parser.add_argument("--background_color", "-bg", type=int, default=255)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--num-random-views", "-nrv", type=int, default=1)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotation-grain", "-rg", type=int, default=3)
    args = parser.parse_args()
    
    # Camera and Screen
    camera = Camera(args.focal_length, args.film_aperture_width, args.film_aperture_height,
                    args.z_near, args.z_far, args.image_width, args.image_height)
    screen = camera.compute_screen()
    # Mesh data (world)
    mesh = o3d.io.read_triangle_mesh(args.file_path)
    center = np.mean(np.asarray(mesh.vertices), axis=0)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    rng = np.random.RandomState(412)
    i = 0
    # View setting
    viewp = np.asarray(vertices)[i] * 2.0
    M_c2w = compute_from_to_matrix(viewp, center)
    M_w2c = np.linalg.inv(M_c2w)
    camera.camera_to_world = M_c2w
    P = camera.compute_projection_matrix(screen)
    # Mesh (camera)
    mesh.vertices = o3d.open3d_pybind.utility.Vector3dVector(world_to_camera(vertices, M_w2c))
    mesh.triangles = o3d.open3d_pybind.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.open3d_pybind.utility.Vector3dVector(colors)
    meshes = [mesh]
    # Render
    image = render(meshes, camera, screen, args)
    # Save
    _, filename = os.path.split(args.file_path)
    name, _ = os.path.splitext(filename)
    cv2.imwrite(f"{name}_{args.image_width}x{args.image_height}_{i:02d}.png", 
                image.transpose([1, 2, 0])[:, :, ::-1])

