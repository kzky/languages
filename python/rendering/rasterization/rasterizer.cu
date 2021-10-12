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
                   int H, int W, 
                   Matrix4x4 P, 
                   Matrix4x4 M_w2c, 
                   float shift) {
  const int *triangles = tmesh.triangles_;
  const float *vertices = tmesh.vertices_;
  const float *vertex_colors = tmesh.vertex_colors_;

  UTIL_CUDA_KERNEL_LOOP(idx, num_triangles) {
    // TODO: We can do most of parts related to projection in NN library side
    // triangle prt at idx
    auto triangle_ptr = triangles + (idx * 3);
    auto triangle = to_triangle(vertices, triangle_ptr);
    // World -> Homogeneous Clipping Space
    auto triangle_h = transform(M_w2c, triangle);
    auto triangle_hcp = to_homogeneous_clipping_coords(P, triangle_h);

    auto iccond = clipping_condition(triangle_hcp);
    if (!iccond) 
      // TODO: this makes the boundary of an image not rendered.
      return;

    // -> Unit Cube Space -> Raster Space
    auto triangle_uc = to_unit_cube_coords(triangle_hcp);
    auto triangle_r = to_raster_coords(triangle_uc, H, W);
    // precompute inverse of z
    triangle_r.inv_z();
    // precompute colors
    auto triangle_colors = to_triangle(vertex_colors, triangle_ptr);
    triangle_colors.v0 *= triangle_r.v0.z;
    triangle_colors.v1 *= triangle_r.v1.z;
    triangle_colors.v2 *= triangle_r.v2.z;
    AABB aabb(triangle_r);
    AABBi aabbi(aabb, H, W);
    auto area = edge_function(triangle_r);
    auto edges = to_edges(triangle_r);
    for (auto y = aabbi.mini.y; y < aabbi.maxi.y; y++) {
      auto py = y + 0.5f;
      for (auto x = aabbi.mini.x; x < aabbi.maxi.x; x++) {
        auto px = x + 0.5f;
        auto p = make_float3(px, py, 0.f);
        auto bc_coords = edge_function(triangle_r, p);
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
          continue;
        auto color = z * weighted_sum(triangle_colors, bc_coords);
        add_color(image, color, y, x, H, W);
      }
    } // pixels loop
  } // triangles loop
}


__global__
void kernel_render_with_shadow(int num_triangles, TriangleMesh tmesh, 
                               float *z_buffer, float *image, 
                               ShadowMapN shadow_maps, 
                               int H, int W, 
                               Matrix4x4 P, Matrix4x4 P_inv, 
                               Matrix4x4 M_w2c, Matrix4x4 M_c2w, 
                               float shift) {
  const int *triangles = tmesh.triangles_;
  const float *vertices = tmesh.vertices_;
  const float *vertex_colors = tmesh.vertex_colors_;

  UTIL_CUDA_KERNEL_LOOP(idx, num_triangles) {
    // TODO: We can do most of parts related to projection in NN library side

    // triangle prt at idx
    auto triangle_ptr = triangles + (idx * 3);
    auto triangle = to_triangle(vertices, triangle_ptr);
    // World -> Camera -> Homogeneous Clipping Space
    auto triangle_h = transform(M_w2c, triangle);
    auto triangle_hcp = to_homogeneous_clipping_coords(P, triangle_h);

    auto iccond = clipping_condition(triangle_hcp);
    if (!iccond) 
      // TODO: this makes the boundary of an image not rendered.
      return;

    // -> Unit Cube Space -> Raster Space
    auto triangle_uc = to_unit_cube_coords(triangle_hcp);
    auto triangle_r = to_raster_coords(triangle_uc, H, W);

    // precompute inverse of z
    triangle_r.inv_z();
    // precompute colors
    auto triangle_colors = to_triangle(vertex_colors, triangle_ptr);
    triangle_colors.v0 *= triangle_r.v0.z;
    triangle_colors.v1 *= triangle_r.v1.z;
    triangle_colors.v2 *= triangle_r.v2.z;
    AABB aabb(triangle_r);
    AABBi aabbi(aabb, H, W);
    auto area = edge_function(triangle_r);
    auto edges = to_edges(triangle_r);
    for (auto y = aabbi.mini.y; y < aabbi.maxi.y; y++) {
      auto py = y + 0.5f;
      for (auto x = aabbi.mini.x; x < aabbi.maxi.x; x++) {
        auto px = x + 0.5f;
        auto p = make_float3(px, py, 1.f);
        auto bc_coords = edge_function(triangle_r, p);
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
        auto color = z * weighted_sum(triangle_colors, bc_coords);
        auto normal = normals(triangle);
        //TODO: (x, y) opsition in shadow is somehow weird
        // shadow-test and intensity
        // Raster -> Unit -> Camera -> World
        auto px_uc = px * (2.f / W) - 1.f;
        auto py_uc = 1.f - (2.f / H) * py;
        auto v_uc = make_float4(px_uc, py_uc, z, 1.f);
        auto v_c = transform(P_inv, v_uc);
        v_c = w_divide(v_c);
        auto v_w = transform(M_c2w, v_c);
        auto intensity = 0.f;
        for (auto l = 0; l < shadow_maps.n_; l++) {
          // World -> Light -> HClip -> Unit -> [0, 1]^2
          auto v_s = transform(shadow_maps.data_[l].M_, v_w);
          v_s = transform(shadow_maps.data_[l].P_, v_s);
          v_s.x /= v_s.w;
          v_s.y /= v_s.w;
          v_s.z /= v_s.w;
          v_s.x = (v_s.x + 1.f) * 0.5f;
          v_s.y = (v_s.y + 1.f) * 0.5f;  // z is measured in the unit cube (not raster)
          auto z_buff = lookup_shadow(make_float2(v_s.x, v_s.y), H, W, shadow_maps.data_[l].z_buffer_);
          auto shadow = in_shadow(v_s, H, W, shadow_maps.data_[l].z_buffer_);
          auto vis = !shadow;
          // printf("vis, v_s.x, v_s.y, v_s.z, v_s.w, z_buff = %d, %f.5, %f.5, %f.5, %f.5, %f.5\n", 
          //        vis, v_s.x, v_s.y, v_s.z, v_s.w, z_buff);
          auto dir = to_float3(shadow_maps.data_[l].dlight_.direction_);  // normally, have to transforms
          intensity += vis * shadow_maps.data_[l].dlight_.intensity_ * max(dot(normal, -dir), 0.f);
          color *= intensity;
        }
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
            int64_t z_buffer, int64_t image, 
            Matrix4x4 M_w2c, 
            Matrix4x4 P, 
            float shift) {
  auto z_buffer_data = reinterpret_cast<float*>(z_buffer);
  auto image_data = reinterpret_cast<float*>(image);
  auto H = camera.image_height_;
  auto W = camera.image_width_;
  auto num_triangles = tmesh.num_triangles_;
  UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_render<false>, num_triangles, 
                                 tmesh, 
                                 z_buffer_data, image_data, 
                                 H, W, 
                                 P, 
                                 M_w2c, 
                                 shift);
}

void render_with_shadow(Camera camera, Screen screen, 
                        TriangleMesh tmesh, 
                        std::vector<ShadowMap> shadow_maps, 
                        int64_t z_buffer, int64_t image, 
                        Matrix4x4 M_w2c, Matrix4x4 M_c2w, 
                        Matrix4x4 P, Matrix4x4 P_inv, 
                        float shift) {
  auto z_buffer_data = reinterpret_cast<float*>(z_buffer);
  auto image_data = reinterpret_cast<float*>(image);
  auto H = camera.image_height_;
  auto W = camera.image_width_;
  auto num_triangles = tmesh.num_triangles_;

  ShadowMapN shadow_maps0(shadow_maps.size());

  for (auto i = 0; i < shadow_maps.size(); i++) {
    shadow_maps0.data_[i] = shadow_maps[i];
    UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_render<true>, num_triangles, 
                                   tmesh, 
                                   shadow_maps0.data_[i].z_buffer_, image_data, 
                                   H, W, 
                                   shadow_maps0.data_[i].P_, 
                                   shadow_maps0.data_[i].M_,
                                   shift);
  }
  UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_render_with_shadow, num_triangles, 
                                 tmesh, 
                                 z_buffer_data, image_data, 
                                 shadow_maps0, 
                                 H, W, 
                                 P, P_inv, 
                                 M_w2c, M_c2w, 
                                 shift);

}


PYBIND11_MODULE(rasterizer_cuda_kernel, m) {
  m.doc() = "Rasterizer Module";
  m.def("render", &render, "rasterize triangle mesh.");
  m.def("render", &render_with_shadow, "rasterize triangle mesh.");
}


/*
$ cd ..
$ nvcc -shared --std=c++11 --compiler-options -fPIC -I./ `python3 -m pybind11 --includes` rasterization/rasterizer.cu -o rasterizer_cuda_kernel`python3-config --extension-suffix`
 */