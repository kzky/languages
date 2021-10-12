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
from render_utils import (Camera, Screen, FitResolutionGate, MaterialType,
                          TriangleMesh, DistantLight, PointLight)
from world import compute_from_to_matrix, world_to_camera

import raytracer_cuda_kernel

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
    parser.add_argument("--background_color", "-bg", type=float, default=0)
    parser.add_argument("--shadow-bias", "-sb", type=float, default=0.0001)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--Ks", type=float, default=0.08)
    parser.add_argument("--n-specular", type=float, default=10)
    parser.add_argument("--num-random-views", "-nrv", type=int, default=1)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotation-grain", "-rg", type=int, default=3)
    args = parser.parse_args()

    # Context
    device = torch.device("cuda:0")
    
    # Camera and Screen
    camera = Camera(args.focal_length, args.film_aperture_width, args.film_aperture_height,
                    args.z_near, args.z_far, args.image_width, args.image_height, FitResolutionGate.OVER_SCREEN)
    screen = camera.compute_screen()
    # Mesh data (world)
    mesh = o3d.io.read_triangle_mesh(args.file_path)
    np_center = np.mean(np.asarray(mesh.vertices), axis=0)
    np_triangles = np.asarray(mesh.triangles)
    np_vertices = np.asarray(mesh.vertices)
    np_vertex_colors = np.asarray(mesh.vertex_colors)
    rng = np.random.RandomState(412)
    tmeshes = []
    for i in rng.randint(0, len(np.asarray(mesh.vertices)), args.num_random_views):
        # View setting
        viewp = np_center + 5 * (np_vertices[i] - np_center)
        M_c2w = compute_from_to_matrix(viewp, np_center)
        M_w2c = np.linalg.inv(M_c2w)
        camera.camera_to_world = M_c2w.T
        # Light setting
        dlights = [DistantLight([0.21176471, 0.28235294, 0.92156863], 10.8, [1, 1, 1]),
                   ## DistantLight([1., 1., 1.], 0.4, [1, 1, 0.1]),
                   ## DistantLight([1., 1., 1.], 0.8, [1, -0.1, 1]),
                   ## DistantLight([1., 1., 1.], 0.0, [-1, -1, -1])
                   ]
        ##dlights = []
        plights = []
        ## plights = [PointLight([0.92156863, 0.29803922, 0.20392157], 1., [0, 0, 0]),  # red
        ##            PointLight([0.90980392, 0.92156863, 0.20392157], 1., [0, 0, 0]),  # yellow
        ##            PointLight([0.21176471, 0.28235294, 0.92156863], 1., [0, 0, 0]),  # blue
        ##            PointLight([0.09803922, 0.87843137, 0.25490196], 1., [0, 0, 0])   # green
        ##            ]
        ## n = len(plights)
        ## for j, k in enumerate(rng.randint(0, len(np.asarray(mesh.vertices)), n)):
        ##     lpos = np_center + (k % n + 1) * (np_vertices[k] - np_center)
        ##     plights[j].intensity = (k % n + 1) * 10000
        ##     plights[j].position = lpos.tolist()

        # NdArray
        ## mesh
        vertices = torch.from_numpy(np_vertices).to(device)
        triangles = torch.from_numpy(np_triangles).to(device)
        vertex_colors = torch.from_numpy(np_vertex_colors).to(device)
        vertices_ptr = vertices.data_ptr()
        triangles_ptr = triangles.data_ptr()
        vertex_colors_ptr = vertex_colors.data_ptr()
        tmesh = TriangleMesh(triangles.shape[0], vertices.shape[0],
                             triangles_ptr, vertices_ptr, vertex_colors_ptr,
                             MaterialType.DIFFUSE)
        tmesh.n_specular = args.n_specular
        tmesh.Ks = args.Ks
        tmeshes.append(tmesh)
        ## image
        image = torch.zeros((3, camera.image_height, camera.image_width), dtype=torch.float32).to(device)
        image.fill_(0.0)
        image_ptr = image.data_ptr()
        # Render
        st = time.perf_counter()
        raytracer_cuda_kernel.render(camera, screen, tmeshes, dlights, plights, 
                                     image_ptr,
                                     args.background_color, args.shadow_bias)
        # Save
        _, filename = os.path.split(args.file_path)
        name, _ = os.path.splitext(filename)
        data = image.cpu().numpy()
        data = np.clip(data.transpose([1, 2, 0])[:, :, ::-1], 0, 255)
        cv2.imwrite(f"{name}_{args.image_width}x{args.image_height}_{i:02d}.png", data)
        print(f"{time.perf_counter() - st}[s]")

