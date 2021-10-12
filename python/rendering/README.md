# Rendering Examples

Computer graphics rendering is a method to produce a photorealistic images from a scene in 3D. 
Mainly, rendering methods address two main problem: visibility problem and light transport (shading or lighting). The latter is futher decomposed into direct lighting and indirect lighting. Visibility is a problem where two points in a scene can be seen each other. The light transport is the problem of how light behaves and an amount of lights changes when it is traveling in a scene.

Roughtly, there are only two ways to tackle the visibility problem, these are called rasterization and ray-casting. On the other hand, there are many ways for solving light transport problem, sometime, it is specific to visibility problem or somtims not.

This example shows some methods for solving the two problems.

Note that there are multiple representation for a scene or simply object: voxel, mesh, signed distance function, density. However, in this example, we limit and use 3D mesh to represent an object. In term of speed, any code level optimization is not peformed, the look-and-feel was preferred to understand what is going on.


## Prerequisite

Install the followings

In system level, 
- cuda
- (g++)

In python level, 
- torch
- pybind11
- open3d


since we have to compile the c++/cuda file.

## (TODO) Dockder

Use the dokcerfile in the same directory.

## Directory Structure

Here is the high-level description for the files and directory structure.

- cuda\_common.cuh: cuda basic utility
- helper\_math.h: cuda basic math (almost same as Mark Harris)
- render\_utils.hpp: rendering utility common in c++ and cuda 
- render\_utils.cpp: bind file for render\_utils.hpp
- render\_utils.cuh; rendering utility in cuda
- rasterization
  - naive_rasterizer.py: naive python sample
  - rasterizer.cuda: rasterization algorithm and its bind method
  - rasterizer_conservative.cuda: conservative rasterization algorithm and bind method
  - rasterizer_cuda.py: execution file
  - rasterizer\_cuda\_shadowmaps.py: execution file with shadowmaps
- raytracing
  - raytracer.py: 
  - raytracer.cu: naive ray tracing algorithm and its bind method
  - raytracer_cuda.py: execution file
  - raytracer\_naive\_reflection\_scene.py: execution file for the reflection simulation
  - raytracer\_naive\_refraction\_scene.py: execution file for the refraction and (reflection) simulation
  - raytracer\_naive\_global_illumination.py: execution file for the global illumination simulation
  

## Dataset

Now, the scripts relies on the open3d data loader, so ply file is preferable. Any file of that format can be used if the object representation is 3D mesh. I used the following data because of simplicity and a bit fun.

```bash
wget https://downloads.greyc.fr/Greyc3DColoredMeshDatabase/Reference_3D_colored_meshes.zip
```


## Rasterization

Rasterization is called the object-centric method since we start from the object (or world) space to transform and arrive at the image (raster) space at the end. First, we transform an object (all triangle mesh) in the world space to the camera space using a world-to-camera matrix (aka. the extrinsic camera parameter), then transform it to the unit cuda (aka. canonical viewing frustum) by the perspective projection matrix (or orthonormal projection matrix) which are deterined by the intrinsic camera parameters. Roughly, we finally rasterize a triangle into an image by looking up z-buffer (aka depth-buffer).


### Compile

In this directory, 

```bash
c++ -O1 -Wall -shared -std=c++11 -fPIC -I./ -I/home/EU/deyoshik/git/dlpack/include `python3 -m pybind11 --includes` render_utils.cpp -o render_utils`python3-config --extension-suffix`

nvcc -O1 -shared --std=c++11 --compiler-options -fPIC -I./  `python3 -m pybind11 --includes` rasterization/rasterizer.cu -o rasterizer_cuda_kernel`python3-config --extension-suffix`
```

### Execute

This example does not include any lighting, just use the object specific colors to render a scene.


```bash
python rasterization/rasterizer_cuda.py -f /pathto/Reference_3D_colored_meshes/Mario.ply
```


### Shadowmap

TODO: the code doe contain some bug. In some condition (perhaps in triangle winding order, culling, and/or sign in coordinates), a shadow becomes the other way around.

```bash
python rasterization/rasterizer_cuda_shadowmaps.py
```


## Ray Tracing

This ray tracing is a naive ray tracer to trace the ray in backward. First, it casts a ray from an eye (camera) position to each pixel and transform it by the camera intrinsic parameter, then transform to the world space using the camera-to-world matrix (aka. the extrinsic camera parameter). If a ray hits the surface of an object, cast a ray called a shadow ray again to a light position to check whether this point is visible from the light. If not, it is shaded, and if so, it is lit.


### Compile

In this directory, 


```bash
c++ -O1 -Wall -shared -std=c++11 -fPIC -I./ `python3 -m pybind11 --includes` render_utils.cpp -o render_utils`python3-config --extension-suffix`

nvcc -shared --std=c++11 --compiler-options -fPIC -I./ `python3 -m pybind11 --includes` -lcurand raytracing/raytracer.cu -o raytracer_cuda_kernel`python3-config --extension-suffix`
```

Note there are some warnings when compiling and some errors when executing, this relates to the number of recursions and the number of threads launched. To remove the error in runtmie, reduce the number of threads in that case.


### Execute

This example include the first-order lighting. Take care that the execution takes long time since any acceleration structure (e.g., bounding volume hierarchy) is not used. In other words, the ray tracing is really slow without an acceleration structure.

To change lighting effect, change the light setting in the script.


```bash
python raytracing/raytracer_cuda.py -f ~/data/Reference_3D_colored_meshes/Mario.ply -iw 320 -ih 240
```


### Reflection and Refraction

This is a simulation of the reflection and refraction with a very simple scene.

```bash
python raytracing/raytracer_naive_reflection_scene.py --Ks 0.5

python raytracing/raytracer_naive_refraction_scene.py --Ks 0.5

```


### Global Illumination

This is a simulation of the global illumination by diffuse interreflection with a very simple scene.

```bash
python raytracing/raytracer_naive_global_illumination.py --Nir 16
```

For ablation of the number of indirect rays, do 

```bash
for i in 0 32 64 128; do
    python raytracing/raytracer_naive_global_illumination.py -iw 320 -ih 240 -Nir ${i}
done
```


