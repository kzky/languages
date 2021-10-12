import numpy as np
import argparse
import os
import open3d as o3d
from itertools import product
from tqdm import tqdm
import cv2

import sys
#sys.path.append('../')
from render_utils import (Camera, Screen, FitResolutionGate, MaterialType, TriangleMesh, ShadowMap,
                          to_matrix4x4, from_matrix4x4, 
                          DistantLight, PointLight)
from world import (look_at, compute_from_to_matrix, world_to_camera, create_box, create_sphere, 
                   create_cone, create_coord_frame,
                   create_box_meshpy, create_sphere_meshpy,
                   create_coord_frame_meshpy, create_plane_meshpy,
                   ShadowMapPy)
import rasterizer_cuda_kernel

import torch

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", "-f", type=str, required=False)
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
    center = np.array([0., 0., 0.])

    # Scene
    tmeshes = []
    ## plane
    scale = 8
    bias = 0
    tmeshpy0 = create_plane_meshpy(scale=scale, bias=bias)
    ## sphere
    scale = 1
    bias = [2.5, 1, 0]
    tmeshpy1 = create_sphere_meshpy(scale=scale, bias=bias, resolution=160)
    ## box
    scale = 1
    bias = [0, 0.5, 2.0]
    tmeshpy2 = create_box_meshpy(scale=scale, bias=bias)
    ## ## box 
    ## scale = 1
    ## bias = [-0.5, 0.5, -1.]
    ## tmeshpy3 = create_box_meshpy(scale=scale, bias=bias)
    # Mesh (constructed by scene)
    tmeshpy = tmeshpy0 + tmeshpy1 + tmeshpy2# + tmeshpy3
    tmesh = tmeshpy.create_triangle_mesh(device)
    tmeshes.append(tmesh)
    # Lights
    plights = []
    dlights = [DistantLight([1, 1, 1], 1, [-1, 1, -1]), 
               ## DistantLight([1, 1, 1], 0.5, [1, 1, 1]),
               ## DistantLight([1, 1, 1], 0.2, [-0.5, 5, -0.5]),
               ## DistantLight([1, 1, 1], 0.6, [0.25, 1, -0.25])
               ]
    # ShadowMaps
    camera_l = Camera(args.focal_length, args.film_aperture_width, args.film_aperture_height,
                      args.z_near, args.z_far, args.image_width, args.image_height,
                      FitResolutionGate.OVER_SCREEN)
    camera_l.focal_length = 10
    camera_l.z_near = 1
    camera_l.z_far = 1000
    screen_l = camera_l.compute_screen()
    shadow_mappys = []
    for l in range(len(dlights)):
        viewp = np.array(dlights[l].direction) * 5
        M_l2w = look_at(viewp, center)
        M_w2l = to_matrix4x4(np.linalg.inv(M_l2w))
        #P = camera_l.orthographic_projection_matrix(screen_l)
        P = camera_l.perspective_projection_matrix(screen)
        shadow_mappy = ShadowMapPy(camera_l.image_width, camera_l.image_height,
                                   dlights[l], M_w2l, P)
        shadow_mappys.append(shadow_mappy)
    # View setting
    viewp = np.array([1, 1, 1]) * 5.
    M_c2w = look_at(viewp, center)
    M_w2c = to_matrix4x4(np.linalg.inv(M_c2w))
    M_c2w = to_matrix4x4(M_c2w)
    # z_buffer and image
    z_buffer = torch.zeros((camera.image_height, camera.image_width), dtype=torch.float32).to(device)
    image = torch.zeros((3, camera.image_height, camera.image_width), dtype=torch.float32).to(device)
    z_buffer.fill_(1e24)
    image.fill_(args.background_color)
    z_buffer_ptr = z_buffer.data_ptr()
    image_ptr = image.data_ptr()
    P = camera.perspective_projection_matrix(screen);
    P_inv = to_matrix4x4(np.linalg.inv(from_matrix4x4(P)))
    # Render
    st = time.perf_counter()
    shadow_maps = [shadow_mappy.create_shadow_map(device) for shadow_mappy in shadow_mappys]
    rasterizer_cuda_kernel.render(camera, screen, tmesh,
                                  #[], 
                                  shadow_maps, 
                                  z_buffer_ptr, image_ptr,
                                  M_w2c, M_c2w,
                                  P, P_inv, 
                                  args.shift)
    
    # Save
    ## _, filename = os.path.split(args.file_path)
    ## name, _ = os.path.splitext(filename)
    data = np.clip(image.cpu().numpy(), 0.0, 255.0)
    cv2.imwrite(f"naive_scene_{args.image_width}x{args.image_height}_shadowmap.png", 
                data.transpose([1, 2, 0])[:, :, ::-1])
    print(f"{time.perf_counter() - st}[s]")
    ## data = np.clip(shadow_mappys[0].z_buffer.data, 0.0, 255.0)
    ## cv2.imwrite(f"naive_scene_{args.image_width}x{args.image_height}.png", 
    ##             data)

