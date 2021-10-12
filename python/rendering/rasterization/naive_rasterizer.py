import numpy as np
import argparse
import os
import open3d as o3d
from itertools import product
from tqdm import tqdm
import cv2

import sys
#sys.path.append('../')
from world import *

def rasterize(mesh, camera, screen, args):
    M_w2c = np.linalg.inv(camera.camera_to_world)
    # Mesh data
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    # Image and Z Buffer
    image = np.ones([3, args.image_height, args.image_width], dtype=np.uint8) * 255
    z_buffer = np.ones([args.image_height, args.image_width], dtype=np.float32) * args.z_far
    # Vertex loop
    for i in tqdm(range(len(triangles)), "Triangles Loop"):
        triangle = triangles[i]
        v0 = vertices[triangle[0]]
        v1 = vertices[triangle[1]]
        v2 = vertices[triangle[2]]
        # Convert to the raster space
        v0_r = convert_world_to_raster(homogeneous(v0), M_w2c, camera, screen)
        v1_r = convert_world_to_raster(homogeneous(v1), M_w2c, camera, screen)
        v2_r = convert_world_to_raster(homogeneous(v2), M_w2c, camera, screen)
        # Precompute the reciprocal of z
        v0_r[2] = 1.0 / v0_r[2]
        v1_r[2] = 1.0 / v1_r[2]
        v2_r[2] = 1.0 / v2_r[2]
        # Prepare the vertex attributes (colors) and precompute for the correction interpolation
        c0 = colors[triangle[0]]
        c1 = colors[triangle[1]]
        c2 = colors[triangle[2]]
        c0 = c0 * v0_r[2]
        c1 = c1 * v1_r[2]
        c2 = c2 * v2_r[2]
        # Compute the bounding box of the triangle (integer)
        xmin = np.min([v0_r[0], v1_r[0], v2_r[0]])
        xmax = np.max([v0_r[0], v1_r[0], v2_r[0]])
        ymin = np.min([v0_r[1], v1_r[1], v2_r[1]])
        ymax = np.max([v0_r[1], v1_r[1], v2_r[1]])
        if xmin > camera.image_width - 1 or xmax < 0 or ymin > camera.image_height or ymax < 0:
            #TODO: this makes the boundary of an image not rendered.
            continue
        x0, x1 = max(int(np.floor(xmin)), 0), min(int(np.floor(xmax)), camera.image_width - 1)
        y0, y1 = max(int(np.floor(ymin)), 0), min(int(np.floor(ymax)), camera.image_height - 1)
        area = edge_function(v0_r, v1_r, v2_r)
        e0 = v2_r - v1_r
        e1 = v0_r - v2_r
        e2 = v1_r - v0_r
        # Overlopping pixel loop
        for y in range(y0, y1):
            py = y + 0.5
            for x in range(x0, x1):
                # (unnormalized) barycentric coordinates
                px = x + 0.5
                pl = np.array([px, py, 0.0])
                w0 = edge_function(v1_r, v2_r, pl)
                w1 = edge_function(v2_r, v0_r, pl)
                w2 = edge_function(v0_r, v1_r, pl)
                # in/out-side test (+ top-left rule)
                overlaps = True
                overlaps *= ((e0[1] == 0 and e0[0] > 0) or e0[1] > 0) if w0 == 0 else w0 > args.slope_ratio
                overlaps *= ((e1[1] == 0 and e1[0] > 0) or e1[1] > 0) if w1 == 0 else w1 > args.slope_ratio
                overlaps *= ((e2[1] == 0 and e2[0] > 0) or e2[1] > 0) if w2 == 0 else w2 > args.slope_ratio
                #if not (w0 >= 0 and w1 >= 0 and w2 >= 0):
                if not overlaps:
                    continue
                w0 /= area
                w1 /= area
                w2 /= area
                z = 1.0 / (v0_r[2] * w0 + v1_r[2] * w1 + v2_r[2] * w2)
                # visibility check
                if z > z_buffer[y, x]:
                    continue
                z_buffer[y, x] = z
                # perspective correct interpolation for the pixel color over (v0, v1, v2)
                pc = z * (c0 * w0 + c1 * w1 + c2 * w2)
                image[:, y, x] = pc * 255
    return image


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
    parser.add_argument("--slope-ratio", "-sr", type=float, default=0)
    parser.add_argument("--naive-color", action="store_true")
    parser.add_argument("--num-random-views", "-nrv", type=int, default=1)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotation-grain", "-rg", type=int, default=3)
    args = parser.parse_args()
    
    # Camera and Screen
    camera = Camera(args.focal_length, args.film_aperture_width, args.film_aperture_height,
                    args.z_near, args.z_far, args.image_width, args.image_height)
    screen = camera.compute_screen()
    # Mesh data
    mesh = o3d.io.read_triangle_mesh(args.file_path)
    center = np.mean(np.asarray(mesh.vertices), axis=0)
    rng = np.random.RandomState(412)
    for i in rng.randint(0, len(np.asarray(mesh.vertices)), args.num_random_views):
        # View setting
        viewp = np.asarray(mesh.vertices)[i] * 2.0
        M_c2w = compute_from_to_matrix(viewp, center)
        camera.camera_to_world = M_c2w
        # Rasterize
        image = rasterize(mesh, camera, screen, args)
        # Save
        _, filename = os.path.split(args.file_path)
        name, _ = os.path.splitext(filename)
        name = name + "_naive_color" if args.naive_color else name
        cv2.imwrite(f"{name}_{args.image_width}x{args.image_height}_{i:02d}.png", 
                    image.transpose([1, 2, 0])[:, :, ::-1])
    # Rotate
    if args.rotate:
        # View setting
        center = np.mean(np.asarray(mesh.vertices), axis=0)
        viewp0 = np.asarray(mesh.vertices)[0] * 2.0
        for alpha0, beta0, gamma0 in product(np.linspace(-180, 180, args.rotation_grain),
                                             np.linspace(-90, 90, args.rotation_grain),
                                             np.linspace(-180, 180, args.rotation_grain)):
            alpha = np.pi / 180 * int(alpha0)
            beta = np.pi / 180 * int(beta0)
            gamma = np.pi / 180 * int(gamma0)
            R = rotation_matrix_3d(0, 0, gamma)
            viewp = (homogeneous(viewp0 - center)).dot(R.T)[:3] + center
            ## # Another way
            ## M_o2w = compute_from_to_matrix(center, viewp0)
            ## M_w2o = np.linalg.inv(M_o2w)
            ## viewp = R.dot(homogeneous(viewp0).dot(M_w2o)).dot(M_o2w)[:3]
            M_c2w = compute_from_to_matrix(viewp, center)
            camera.camera_to_world = M_c2w
            # Rasterize
            image = rasterize(mesh, camera, screen, args)
            # Save
            _, filename = os.path.split(args.file_path)
            name, _ = os.path.splitext(filename)
            name = name + "_naive_color" if args.naive_color else name
            cv2.imwrite(f"{name}_{args.image_width}x{args.image_height}"\
                        f"_{int(alpha0):03d}x{int(beta0):03d}x{int(gamma0):03d}.png",
                        image.transpose([1, 2, 0])[:, :, ::-1])
    
