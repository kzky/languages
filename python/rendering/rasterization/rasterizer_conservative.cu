#include <pybind11/pybind11.h>
#include <iostream>

#include <helper_math.h>
#include <cuda_common.cuh>
#include <render_utils.cuh>
#include <render_utils.hpp>

#include <pybind11/stl.h>

namespace py = pybind11;
using namespace render_utils;
using namespace render_utils_cuda;


__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}


template<bool shadow = false>
__global__
void kernel_render(int num_triangles, TriangleMesh tmesh, 
                   float *z_buffer, float *image, 
                   DistantLightN &dlights, 
                   PointLightN &plights, 
                   ShadowMap shadow_map, 
                   int H, int W, 
                   Matrix4x4 P, 
                   Matrix4x4 M_w2c, 
                   float shift) {
  const int *triangles = tmesh.triangles_;
  const float *vertices = tmesh.vertices_;
  const float *vertex_colors = tmesh.vertex_colors_;

  UTIL_CUDA_KERNEL_LOOP(idx, num_triangles) {
    // triangle prt at idx
    auto triangle_ptr = triangles + (idx * 3);
    auto triangle = to_triangle(vertices, triangle_ptr);
    // World -> Homogeneous Clipping Space
    auto triangle_h = transform(M_w2c, triangle);
    auto triangle_hcp = to_homogeneous_clipping_coords(P, triangle_h);
    auto triangle_hcp_c = conservative(triangle_hcp, shift);

    auto iccond = clipping_condition(triangle_hcp);
    if (!iccond) 
      // TODO: this makes the boundary of an image not rendered.
      return;

    // -> Unit Cube Space -> Raster Space
    auto triangle_uc = to_unit_cube_coords(triangle_hcp);
    auto triangle_uc_c = to_unit_cube_coords(triangle_hcp_c);
    auto triangle_r = to_raster_coords(triangle_uc, H, W);
    auto triangle_r_c = to_raster_coords(triangle_uc_c, H, W);
    // precompute inverse of z
    triangle_r.inv_z();
    triangle_r_c.inv_z();
    // precompute colors
    auto triangle_colors = to_triangle(vertex_colors, triangle_ptr);
    triangle_colors.v0 *= triangle_r.v0.z;
    triangle_colors.v1 *= triangle_r.v1.z;
    triangle_colors.v2 *= triangle_r.v2.z;
    AABB aabb(triangle_r);
    AABBi aabbi(aabb, H, W);
    auto area = edge_function(triangle_r_c);
    auto edges = to_edges(triangle_r_c);
    for (auto y = aabbi.mini.y; y < aabbi.maxi.y; y++) {
      auto py = y + 0.5f;
      for (auto x = aabbi.mini.x; x < aabbi.maxi.x; x++) {
        auto px = x + 0.5f;
        auto p = make_float3(px, py, 1.f);
        auto bc_coords = edge_function(triangle_r_c, p);
        // in/out-test (+ top-left rule)
        auto overlap = inside_test(bc_coords, edges);
        if (!overlap)
          continue;
        bc_coords /= area;
        auto z = 1.0f / sum(make_float3(triangle_r.v0.z, triangle_r.v1.z, triangle_r.v2.z) * bc_coords);
        // depth-test (visibility check)
        auto z_buffer_yx = z_buffer + (y * W) + x;
        atomicMin(z_buffer_yx, z);
        auto z_old = *z_buffer_yx;
        if (z_old < z)
          continue;
        if (shadow == true)
          return;
        auto color = z * weighted_sum(triangle_colors, bc_coords);
        auto normal = normals(triangle);
        
        add_color(image, color, y, x, H, W);
      }
    } // pixels loop
  } // triangles loop
}


int2 to_int2(py::tuple shape) {
  return make_int2(shape[0].cast<int>(), shape[1].cast<int>());
}


void render(Camera camera, Screen screen, 
            TriangleMesh tmesh, 
            std::vector<DistantLight> dlights, 
            std::vector<PointLight> plights, 
            int64_t z_buffer, int64_t image, 
            ShadowMap shadow_map, 
            Matrix4x4 M_w2c, 
            Matrix4x4 P, 
            float shift, 
            bool shadow = false) {
  auto z_buffer_data = reinterpret_cast<float*>(z_buffer);
  auto image_data = reinterpret_cast<float*>(image);
  DistantLightN dlights0(dlights.size());
  for (auto i = 0; i < dlights.size(); i++) {
    dlights0.data_[i] = dlights[i];
  }
  PointLightN plights0(plights.size());
  for (auto i = 0; i < plights.size(); i++) {
    plights0.data_[i] = plights[i];
  }
  auto H = camera.image_height_;
  auto W = camera.image_width_;
  auto num_triangles = tmesh.num_triangles_;
  auto kernel = shadow ? kernel_render<true> : kernel_render<false>;
  UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, num_triangles, 
                                 tmesh, 
                                 z_buffer_data, image_data, 
                                 dlights0, plights0, 
                                 shadow_map, 
                                 H, W, 
                                 P, 
                                 M_w2c, 
                                 shift);
}


PYBIND11_MODULE(rasterizer_cuda_kernel, m) {
  m.doc() = "Rasterizer Module";
  m.def("render", &render, "rasterize triangle mesh.");
}


/*
$ cd ..
$ nvcc -shared --std=c++11 --compiler-options -fPIC -I./ `python3 -m pybind11 --includes` rasterization/rasterizer_conservative.cu -o rasterizer_cuda_kernel`python3-config --extension-suffix`
 */