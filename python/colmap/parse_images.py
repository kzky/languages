import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

def main(args):
    # Log format
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)

    image_data = dict()
    with open(args.fpath) as fp:
        for i, l in enumerate(fp):
            if l.startswith("#"):
                continue
            l = l.rstrip()
            elms = l.split(" ")
            if i % 2 == 0:
                image_id = int(elms[0])
                quaternion = np.asarray([float(elms[1]), float(elms[2]), float(elms[3]), float(elms[4])])
                translation = np.asarray([float(elms[5]), float(elms[6]), float(elms[7])])
                camera_id = int(elms[8])
                name = elms[9]
            else:
                pixels = []
                point3d_ids = []
                for j in range(len(elms) // 3):
                    x = float(elms[j * 3])
                    y = float(elms[j * 3 + 1])
                    point3d_id = int(elms[j * 3 + 2])
                    
                    pixels.append(np.asarray([x, y]))
                    point3d_ids.append(point3d_id)
                pixels = np.asarray(pixels)
                point3d_ids = np.asarray(point3d_ids)

                data = dict(image_id=image_id,
                            quaternion=quaternion,
                            translation=translation,
                            camera_id=camera_id,
                            pixels=pixels,
                            point3d_ids=point3d_ids)
                image_data[name] = data
            
    print(image_data[args.name])

    ## camlocs = np.asarray([v["translation"] for k, v in image_data.items()])
    ## camloc_mean = np.mean(camlocs, axis=0)
    ## camloc_centered = camlocs - camloc_mean
    ## dists = np.sum(camloc_centered ** 2, axis=-1) ** 0.5
    ## print(camloc_mean)
    ## print(dists)
    ## print(dists.min(), dists.max())
    ## print(dists.max() / dists.min())

    # Save image with discriptors
    image = cv2.imread(os.path.join(args.dpath, args.name))
    H, W, _ = image.shape
    data = image_data[args.name]
    cnt1 = 0
    cnt0 = 0
    for pixel, point3d_id in zip(data["pixels"], data["point3d_ids"]):
        cnt0 += 1
        if point3d_id == -1:
            continue
        
        y, x = int(pixel[1]), int(pixel[0])
        y0, x0 = y, x
        y1 = max(y0 + 1, H - 1)
        x1 = max(x0 + 1, W - 1)
        image[y0, x0, :] = [0, 0, 255]
        image[y0, x1, :] = [0, 0, 255]
        image[y1, x0, :] = [0, 0, 255]
        image[y1, x1, :] = [0, 0, 255]
        cnt1 += 1
    cv2.imwrite(args.name, image)
    print(f"Num. of points = {cnt1} / {cnt0}")

if __name__ == '__main__':
    import argparse
    import shutil
    import os
    import glob

    parser = argparse.ArgumentParser(description="COLMAP images.txt parser")
    parser.add_argument('--fpath', '-f', type=str, required=True, help="File path to images.txt")
    parser.add_argument('--dpath', '-d', type=str, required=True, help="Directory path to images")
    parser.add_argument('--name', '-n', type=str, required=True, help="File name of image")
                                        
    args = parser.parse_args()
    main(args)

