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
from world import (look_at, compute_from_to_matrix, world_to_camera, create_box, create_sphere, 
                   create_cone, create_coord_frame,
                   create_box_meshpy, create_sphere_meshpy,
                   create_coord_frame_meshpy, create_plane_meshpy,
                   create_yx_plane_meshpy,create_zy_plane_meshpy,
                   ShadowMapPy)
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
    parser.add_argument("--background-color", "-bg", type=float, default=0)
    parser.add_argument("--shadow-bias", "-sb", type=float, default=0.0001)
    parser.add_argument("--max-depth", "-mD", type=int, default=5)
    parser.add_argument("--n-indirect-rays", "-Nir", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--Ks", type=float, default=0.0)
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
    # Light setting
    dlights = [DistantLight([1, 1, 1], 25, [-0.75, -1.5, -0.3]),
               #DistantLight([1, 1, 1], 25, [-0.75, -1.5, -0.75]),
               #DistantLight([1, 1, 1], 10, [-0.3, -1.5, -1.2]),
               ]
    # dlights = []
    ## plights = [PointLight([1, 1, 1], 1000, [10, 10., 1.]),
    ##            PointLight([1, 1, 1], 1000, [5, 20, 10.]),
    ##            ]
    plights = []
    rng = np.random.RandomState(412)

    # Mesh
    tmeshes = []
    ## planes
    scale = 8
    bias = np.asarray([0.0, +0.2, 0.0]) * scale
    vertex_colors = np.asarray([[0.5, 0.2, 0.2], [0.5, 0.2, 0.2], [0.5, 0.2, 0.2], [0.5, 0.2, 0.2]])
    tmeshpy0 = create_plane_meshpy(scale=scale, bias=bias, vertex_colors=vertex_colors)
    tmeshes.append(tmeshpy0.create_triangle_mesh(device))
    scale = 8
    bias = np.asarray([-scale, 0.0, -0.2 * scale])
    vertex_colors = np.asarray([[0.2, 0.5, 0.2], [0.2, 0.5, 0.2], [0.2, 0.5, 0.2], [0.2, 0.5, 0.2]])
    tmeshpy1 = create_zy_plane_meshpy(scale=scale, bias=bias, vertex_colors=vertex_colors)
    tmeshes.append(tmeshpy1.create_triangle_mesh(device))
    scale = 8
    bias = np.asarray([-0.2 * scale, 0.0, -scale])
    vertex_colors = np.asarray([[0.2, 0.2, 0.5], [0.2, 0.2, 0.5], [0.2, 0.2, 0.5], [0.2, 0.2, 0.5]])
    tmeshpy2 = create_yx_plane_meshpy(scale=scale, bias=bias, vertex_colors=vertex_colors)
    tmeshes.append(tmeshpy2.create_triangle_mesh(device))
    ## sphere
    scale = 2
    bias = [-0.5, 5, 1.]
    tmeshpys = create_sphere_meshpy(scale=scale, bias=bias, resolution=20)
    tmeshes.append(tmeshpys.create_triangle_mesh(device))
    print(tmeshpys.triangles.shape)
    exit()

    # View setting
    viewp = np.asarray([0.2, 1.4, 1.4]) * 12
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
                                 args.background_color, args.shadow_bias,
                                 args.max_depth, args.n_indirect_rays)
    # Save
    name = "naive_scene_global_illumination"
    data = image.cpu().numpy()
    data = np.clip(data.transpose([1, 2, 0])[:, :, ::-1], 0, 255)
    cv2.imwrite(f"{name}_{args.image_width}x{args.image_height}_{args.n_indirect_rays:03d}.png", data)
    print(f"{time.perf_counter() - st}[s]")





