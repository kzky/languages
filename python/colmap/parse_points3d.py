import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def main(args):
    # Log format
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)  
    point3d_ids = []
    point3ds = []
    colors = []
    errors = []
    image_ids = []
    point2d_ids = []

    with open(args.fpath) as fp:
        for l in fp:
            if l.startswith("#"):
                continue
            l = l.rstrip()
            elms = l.split(" ")
            point3d_id = int(elms[0])
            point3d = np.asarray([float(elms[1]), float(elms[2]), float(elms[3])])
            color = np.asarray([int(elms[4]), int(elms[5]), int(elms[6])])
            error = float(elms[7])
            for i in range(len(elms[8:]) // 2):
                image_id = elms[8 + i * 2]
                point2d_id = elms[8 + i * 2 + 1]

            point3d_ids.append(point3d_id)
            point3ds.append(point3d)
            colors.append(color)
            errors.append(error)
            image_ids.append(image_id)
            point2d_ids.append(point2d_id)

    point3d_ids = np.asarray(point3d_ids)
    point3ds = np.asarray(point3ds)
    colors = np.asarray(colors) / 255.0
    errors = np.asarray(errors)
    image_ids = np.asarray(image_ids)
    point2d_ids = np.asarray(point2d_ids)

    # Stats
    mean = point3ds.mean(axis=0)
    min_ = point3ds.min(axis=0)
    max_ = point3ds.max(axis=0)
    dist = np.sum((point3ds - mean) ** 2, axis=1) ** 0.5
    print("min_, mean, max_")
    print(min_, mean, max_)
    print("dist.min(), dist.max()")
    print(dist.min(), dist.max())
    print("errors.min(), errors.mean(), errors.max()")
    print(errors.min(), errors.mean(), errors.max())

    # Plots
    plt.plot(np.sort(errors))
    plt.ylabel("Errors in pixels")
    plt.title("Errors (sorted) in pixels")
    plt.savefig("Errors_points3d.png")
    plt.clf()

    plt.plot(np.sort(dist))
    plt.ylabel("Distance")
    plt.title("Distance (sorted)")
    plt.savefig("Distance_points3d.png")
    plt.clf()

    # Point cloud
    idx = (dist < args.distance_threshold) * (errors < args.pixel_threshold)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point3ds[idx])
    pcd.colors = o3d.utility.Vector3dVector(colors[idx])
    o3d.io.write_point_cloud("PCD.ply", pcd)
    print(f"Points in PCD = {np.sum(idx)} / {len(dist)}")

if __name__ == '__main__':
    import argparse
    import shutil
    import os
    import glob

    parser = argparse.ArgumentParser(description="COLMAP points3D.txt parser")
    parser.add_argument('--fpath', '-f', type=str, required=True, help="File path to points3D.txt")
    parser.add_argument('--distance-threshold', '-dt', type=float, default=2.0,
                        help="Distance threshold to remove outlier")
    parser.add_argument('--pixel-threshold', '-pt', type=float, default=2.0,
                        help="Pixel threshold to remove outlier")

    args = parser.parse_args()
    main(args)

