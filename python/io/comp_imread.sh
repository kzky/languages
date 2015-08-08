#!/bin/sh

# scipy (457.126190)
echo 1 > /proc/sys/vm/drop_caches
kernprof -l comp_scipy_imread.py

# pil (457.224693)
echo 1 > /proc/sys/vm/drop_caches
kernprof -l comp_pil_imread.py

# opencv (441.201759)
echo 1 > /proc/sys/vm/drop_caches
kernprof -l comp_cv2_imread.py

