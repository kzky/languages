#!/usr/bin/env bash

files=("4arms_monstre.ply" "Cable_car.ply" "Duck.ply" "Green_monstre.ply" "Jaguar.ply" "Man.ply" "Mario_car.ply" "Rabbit.ply" "Statue.ply" "Asterix.ply" "Dragon.ply" "Green_dinosaur.ply" "Horse.ply" "Long_dinosaur.ply" "Mario.ply" "Pokemon_ball.ply" "Red_horse.ply") 

for f in "${files[@]}"; do
  echo ${f}
  python rasterization/rasterizer.py -f ~/data/Reference_3D_colored_meshes/${f} -iw 640 -ih 480
  python rasterization/rasterizer.py -f ~/data/Reference_3D_colored_meshes/${f} -iw 1280 -ih 960
  python rasterization/rasterizer.py -f ~/data/Reference_3D_colored_meshes/${f} -iw 2560 -ih 1920
done
