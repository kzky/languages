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
from world import (compute_from_to_matrix, world_to_camera, create_box, create_sphere,
                   create_cone, create_coord_frame)

import raytracer_cuda_kernel

import torch

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    device = torch.device("cuda")
    
    # Camera and Screen
    camera = Camera(args.focal_length, args.film_aperture_width, args.film_aperture_height,
                    args.z_near, args.z_far, args.image_width, args.image_height, FitResolutionGate.OVER_SCREEN)
    screen = camera.compute_screen()
    # Light setting
    dlights = [#DistantLight([1, 1, 1], 2, [1, -1, 1]),
               #DistantLight([1, 1, 1], 5, [-1, -1, 1]),
               #DistantLight([1, 1, 1], 1, [-1, -1, -1]),
               DistantLight([1, 1, 1], 8, [-0.5, -0.01, -1]),
               DistantLight([1, 1, 1], 8, [0.5, 3, 100]),
               ]
    ## dlights = []
    ## plights = [PointLight([1, 1, 1], 10, [0, 0, 0]),
    ##            PointLight([1, 1, 1], 10, [1, 1, 1]),
    ##            PointLight([1, 1, 1], 10, [-1, -4, -2]),
    ##            PointLight([1, 1, 1], 10, [3, -2, 3]),
    ##            ]
    plights = []
    rng = np.random.RandomState(412)
    # NdArray
    tmeshes = []
    ## coordinate frame
    scale = 10
    bias = 0
    np_vertices, np_triangles, np_vertex_colors = create_coord_frame(scale=scale, bias=bias)
    vertices0 = torch.from_numpy(np_vertices).to(device)
    triangles0 = torch.from_numpy(np_triangles).to(device)
    vertex_colors0 = torch.from_numpy(np_vertex_colors).to(device)
    vertices_ptr = vertices0.data_ptr()
    triangles_ptr = triangles0.data_ptr()
    vertex_colors_ptr = vertex_colors0.data_ptr()
    tmesh = TriangleMesh(triangles0.shape[0], vertices0.shape[0],
                         triangles_ptr, vertices_ptr, vertex_colors_ptr,
                         MaterialType.DIFFUSE)
    tmesh.n_specular = args.n_specular
    tmesh.Ks = 0.0
    tmeshes.append(tmesh)
    ## box
    scale = 10
    bias = np.asarray([-10, -5, -5])
    np_vertex_colors = np.asarray([0.90980392, 0.92156863, 0.20392157])
    np_vertices, np_triangles, np_vertex_colors = create_box(scale=scale, bias=bias, vertex_colors=np_vertex_colors)
    vertices1 = torch.from_numpy(np_vertices).to(device)
    triangles1 = torch.from_numpy(np_triangles).to(device)
    vertex_colors1 = torch.from_numpy(np_vertex_colors).to(device)
    vertices_ptr = vertices1.data_ptr()
    triangles_ptr = triangles1.data_ptr()
    vertex_colors_ptr = vertex_colors1.data_ptr()
    tmesh = TriangleMesh(triangles1.shape[0], vertices1.shape[0],
                         triangles_ptr, vertices_ptr, vertex_colors_ptr,
                         MaterialType.DIFFUSE)
    tmesh.n_specular = args.n_specular
    tmesh.Ks = args.Ks
    tmeshes.append(tmesh)
    ## sphere
    scale = 1.
    bias = np.asarray([6, 5, 5])
    np_vertices, np_triangles, np_vertex_colors = create_sphere(resolution=80, scale=scale, bias=bias)
    vertices3 = torch.from_numpy(np_vertices).to(device)
    triangles3 = torch.from_numpy(np_triangles).to(device)
    vertex_colors3 = torch.from_numpy(np_vertex_colors).to(device)
    vertices_ptr = vertices3.data_ptr()
    triangles_ptr = triangles3.data_ptr()
    vertex_colors_ptr = vertex_colors3.data_ptr()
    tmesh = TriangleMesh(triangles3.shape[0], vertices3.shape[0],
                         triangles_ptr, vertices_ptr, vertex_colors_ptr,
                         MaterialType.DIFFUSE)
    tmesh.n_specular = args.n_specular
    tmesh.Ks = args.Ks
    tmesh.material_type, tmesh.ior = MaterialType.REFLECTION_AND_REFRACTION, 1.6
    tmeshes.append(tmesh)
    # View setting
    viewp = np.asarray([1, 1, 1]) * 10
    world_center = np.asarray([0, 0, 0])
    M_c2w = compute_from_to_matrix(viewp, world_center)
    M_w2c = np.linalg.inv(M_c2w)
    camera.camera_to_world = M_c2w.T
    ## image
    image = torch.zeros((3, camera.image_height, camera.image_width), dtype=torch.float32).to(device)
    image.fill_(0.0)
    image_ptr = image.data_ptr()
    print(tmeshes)
    # Render
    st = time.perf_counter()
    raytracer_cuda_kernel.render(camera, screen, tmeshes, dlights, plights, 
                                 image_ptr,
                                 args.background_color, args.shadow_bias)
    # Save
    name = "naive_scene"
    data = image.cpu().numpy()
    data = np.clip(data.transpose([1, 2, 0])[:, :, ::-1], 0, 255)
    cv2.imwrite(f"{name}_{args.image_width}x{args.image_height}.png", data)
    print(f"{time.perf_counter() - st}[s]")

