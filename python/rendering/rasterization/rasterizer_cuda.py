import numpy as np
import argparse
import os
import open3d as o3d
from itertools import product
from tqdm import tqdm
import cv2

import sys
#sys.path.append('../')
from render_utils import Camera, Screen, FitResolutionGate, MaterialType, TriangleMesh, ShadowMap, to_matrix4x4, DistantLight, PointLight
from world import compute_from_to_matrix, world_to_camera, look_at, create_box, create_sphere
import rasterizer_cuda_kernel

import torch

import time

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
    parser.add_argument("--shift", type=float, default=0)
    parser.add_argument("--background_color", "-bg", type=float, default=0)    
    parser.add_argument("--num-random-views", "-nrv", type=int, default=1)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotation-grain", "-rg", type=int, default=3)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda:0")

    # Camera and Screen
    camera = Camera(args.focal_length, args.film_aperture_width, args.film_aperture_height,
                    args.z_near, args.z_far, args.image_width, args.image_height,
                    FitResolutionGate.OVER_SCREEN)
    screen = camera.compute_screen()
    # Mesh (world)
    mesh = o3d.io.read_triangle_mesh(args.file_path)
    np_center = np.mean(np.asarray(mesh.vertices, dtype=np.float32), axis=0)
    np_triangles = np.asarray(mesh.triangles, dtype=np.int32)
    np_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    np_vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    rng = np.random.RandomState(412)
    # Lights
    plights = []
    dlights = []
    H, W = camera.image_height, camera.image_width
    for i in rng.randint(0, len(np_vertices), args.num_random_views):
        # Mesh
        vertices = torch.from_numpy(np_vertices).to(device)
        triangles = torch.from_numpy(np_triangles).to(device)
        vertex_colors = torch.from_numpy(np_vertex_colors).to(device)
        vertices_ptr = vertices.data_ptr()
        triangles_ptr = triangles.data_ptr()
        vertex_colors_ptr = vertex_colors.data_ptr()
        tmesh = TriangleMesh(triangles.shape[0], vertices.shape[0],
                             triangles_ptr, vertices_ptr, vertex_colors_ptr,
                             MaterialType.DIFFUSE)
        # View setting
        viewp = np_center + 5 * (np_vertices[i] - np_center)
        M_c2w = look_at(viewp, np_center)
        M_w2c = to_matrix4x4(np.linalg.inv(M_c2w))
        ## z_buffer and image
        z_buffer = torch.zeros((camera.image_height, camera.image_width), dtype=torch.float32).to(device)
        image = torch.zeros((3, camera.image_height, camera.image_width), dtype=torch.float32).to(device)
        z_buffer.fill_(1e24)
        image.fill_(args.background_color)
        z_buffer_ptr = z_buffer.data_ptr()
        image_ptr = image.data_ptr()
        P = camera.perspective_projection_matrix(screen);
        # Render
        st = time.perf_counter()
        rasterizer_cuda_kernel.render(camera, screen, tmesh,
                                      z_buffer_ptr, image_ptr,
                                      M_w2c,
                                      P, 
                                      args.shift)
        
        # Save
        _, filename = os.path.split(args.file_path)
        name, _ = os.path.splitext(filename)
        cv2.imwrite(f"{name}_{args.image_width}x{args.image_height}_{i:02d}.png", 
                    image.cpu().numpy().transpose([1, 2, 0])[:, :, ::-1])
        print(f"{time.perf_counter() - st}[s]")
