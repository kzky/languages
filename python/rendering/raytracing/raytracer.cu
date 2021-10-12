#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <helper_math.h>
#include <cuda_common.cuh>
#include <render_utils.cuh>
#include <render_utils.hpp>
#include <cfloat>

namespace py = pybind11;
using namespace render_utils;
using namespace render_utils_cuda;


__global__ 
void setup_kernel(int num_pixels, curandState *rstate) {
  UTIL_CUDA_KERNEL_LOOP(idx, num_pixels) {
    curand_init(0, idx, 0, &rstate[idx]);
  }
}


__forceinline__ __device__
bool mt_intersect(float3 orig, float3 dir, 
                  float3 v0, float3 v1, float3 v2, 
                  IsectInfo &isect_info, 
                  float epsilon = 1e-8) {
  // MT method
  auto v0v1 = v1 - v0;
  auto v0v2 = v2 - v0;
  auto pvec = cross(dir, v0v2);
  auto det = dot(v0v1, pvec);
  if (det < epsilon) //  back-face culling
    return false;
  if (abs(det) < epsilon)  // ray is parallel planar
    return false;

  auto inv_det = 1.f / det;
  auto tvec = orig - v0;
  auto u = dot(tvec, pvec) * inv_det;
  if (u < 0 || u > 1)
    return false;

  auto qvec = cross(tvec, v0v1);
  auto v = dot(dir, qvec) * inv_det;
  if (v < 0 || (u + v) > 1)
    return false;
  auto t = dot(v0v2, qvec) * inv_det;
  if (t < 0)
    return false;
  // set
  isect_info.hit = true;
  isect_info.tuv_coords[0] = t;
  isect_info.tuv_coords[1] = u;
  isect_info.tuv_coords[2] = v;
  return true;
}


__forceinline__ __device__ 
IsectInfo intersect(float3 orig, float3 dir, 
                    TriangleMeshN &tmeshes, 
                    const float epsilon) {

  IsectInfo isect_info;
  auto tnear = FLT_MAX;

  for (auto i = 0; i < tmeshes.n_; i++) {
    auto num_triangles = tmeshes.data_[i].num_triangles_;
    auto triangles = tmeshes.data_[i].triangles_;
    auto vertices = tmeshes.data_[i].vertices_;
    for (auto j = 0; j < num_triangles; j++) {
      auto triangle = triangles + j * 3;
      auto v0 = to_coords(vertices + (triangle[0] * 3));
      auto v1 = to_coords(vertices + (triangle[1] * 3));
      auto v2 = to_coords(vertices + (triangle[2] * 3));
      bool hit = mt_intersect(orig, dir, v0, v1, v2, isect_info, epsilon);
      if (!hit) 
        continue;
      auto t = isect_info.tuv_coords[0];
      if (t > tnear)
        continue;
      tnear = t;
      isect_info.idx_obj = i;
      isect_info.idx_near = j;
    }
  }
  return isect_info;
}


template <bool indirect = false>
__forceinline__ __device__
float3 cast_ray(const int idx, curandState *rstate, 
                float3 orig, float3 dir, 
                TriangleMeshN &tmeshes, 
                DistantLightN &dlights, 
                PointLightN &plights, 
                int H, int W, 
                float background_color, float sbias, 
                int depth, int max_depth, int n_indirect_rays) {

  if (depth > max_depth) {
    return make_float3(background_color, background_color, background_color);
  }

  auto isect_info = intersect(orig, dir, tmeshes, 
                              1e-8);
  if (!isect_info.hit) {
    //printf("not hit! (depth = %d)\n", depth);
    return make_float3(background_color, background_color, background_color);
  }

  // point of intersection
  auto k = isect_info.idx_obj;
  auto triangles = tmeshes.data_[k].triangles_;
  auto vertices = tmeshes.data_[k].vertices_;
  auto i = isect_info.idx_near;
  auto tuv = isect_info.tuv_coords;
  auto t = tuv[0];
  auto u = tuv[1];
  auto v = tuv[2];
  auto triangle = triangles + i * 3;
  auto p = orig + t * dir;
  auto v0 = to_coords(vertices + (triangle[0] * 3));
  auto v1 = to_coords(vertices + (triangle[1] * 3));
  auto v2 = to_coords(vertices + (triangle[2] * 3));
  auto normal = normals(v0, v1, v2);
  
  // cast ray again
  auto color = make_float3(0.f, 0.f, 0.f);
  if (tmeshes.data_[k].material_type_ == MaterialType::DIFFUSE) {
    auto lcolor = make_float3(0.f, 0.f, 0.f);
    auto intensity = 0.f;
    auto albedo = tmeshes.data_[k].albedo_;
    auto Kd = tmeshes.data_[k].Kd_;
    auto Ks = tmeshes.data_[k].Ks_;
    // distant light
    for (auto i = 0; i < dlights.n_; i++) {
      auto direction = to_float3(dlights.data_[i].direction_);
      auto lcolor_i = to_float3(dlights.data_[i].color_);
      auto intensity_i = dlights.data_[i].intensity_;
      auto s_isect_info = intersect(p + normal * sbias, -direction,
                                    tmeshes, 
                                    1e-8);
      auto vis = !s_isect_info.hit;
      // diffuse
      auto cos = dot(normal, direction);
      auto intensity_d = vis * albedo / M_PI * intensity_i * max(0.f, cos);
      intensity += intensity_d;
      // specular
      auto r = reflect(direction, normal);
      auto n_specular = tmeshes.data_[k].n_specular_;
      auto intensity_s = vis * intensity_i * pow(max(0.f, dot(r, -dir)), n_specular);
      auto scolor = intensity_s * lcolor_i;
      // light color
      lcolor += Ks * scolor;
    }
    // point light
    for (auto i = 0; i < plights.n_; i++) {
      auto position = to_float3(plights.data_[i].position_);
      auto direction = position - p;
      auto r2 = dot(direction, direction);
      direction = normalize(direction);
      auto lcolor_i = to_float3(plights.data_[i].color_);
      auto intensity_i = plights.data_[i].intensity_ / (4.f * M_PI * r2);
      auto s_isect_info = intersect(p + normal * sbias, direction, 
                                    tmeshes, 
                                    1e-8);
      auto len = length(s_isect_info.tuv_coords[0] * direction);
      auto vis = !s_isect_info.hit || (s_isect_info.hit && (r2 < (len * len)) );
      // diffuse
      auto cos = dot(normal, direction);
      auto intensity_d = vis * (albedo / M_PI * intensity_i * max(0.f, cos));
      intensity += intensity_d;
      // specular
      auto r = reflect(direction, normal);
      auto n_specular = tmeshes.data_[k].n_specular_;
      auto intensity_s = vis * intensity_i * pow(max(0.f, dot(r, -dir)), n_specular);
      auto scolor = intensity_s * lcolor_i;
      // light color
      lcolor += (Ks * scolor);
    }
    // Direct diffuse reflection
    auto vertex_colors = tmeshes.data_[k].vertex_colors_;
    auto c0 = to_coords(vertex_colors + (triangle[0] * 3));
    auto c1 = to_coords(vertex_colors + (triangle[1] * 3));
    auto c2 = to_coords(vertex_colors + (triangle[2] * 3));
    color += Kd * (u * c0 + v * c1 + (1.f - u - v) * c2) * intensity + lcolor;
    //    printf("depth, u, v = %d, %f, %f\n", depth, u, v);

    // Indirect diffuse reflection (diffuse interreflection)
    if (indirect) {
      float pdf = 1 / (2 * M_PI); 
      auto id_color = make_float3(0.f, 0.f, 0.f);
      for (auto i = 0; i < n_indirect_rays; i++) { 
        // sample direction -> create coorinate system -> transform
        auto local_state = rstate[idx];
        auto r1 = curand_uniform(&local_state);
        auto r2 = curand_uniform(&local_state);
        rstate[idx] = local_state;
        auto rdir = sample_direction(r1, r2);
        //printf("rdir.x, rdir.y, rdir.z = %f, %f, %f\n", rdir.x, rdir.y, rdir.z);
        auto Nx = make_float3(0.f, 0.f, 0.f);
        auto Nz = make_float3(0.f, 0.f, 0.f);
        create_coordinate_system(normal, Nx, Nz);
        rdir = transform_by_coordinate_system(rdir, Nx, normal, Nz);
        // cast ray
        id_color += r1 * cast_ray<indirect>(idx, rstate, 
                                            p + rdir * sbias * 10, rdir, tmeshes, dlights, plights, H, W, 
                                            background_color, sbias, 
                                            depth + 1, max_depth, n_indirect_rays) / pdf * (albedo / M_PI);
      }
      // divide by the number of indirect rays
      id_color /= n_indirect_rays;
      //printf("id_color.x, id_color.y, id_color.z = %f, %f, %f\n", id_color.x, id_color.y, id_color.z);
      color += id_color;
    }
    return color;
  } else if (tmeshes.data_[k].material_type_ == MaterialType::REFLECTION) {
    auto rcolor = make_float3(0.f, 0.f, 0.f);    
    auto dir_new = normalize(reflect(dir, normal));
    auto p_new = p + normal * sbias;
    rcolor += cast_ray<indirect>(idx, rstate, 
                       p_new, dir_new, tmeshes, dlights, plights, H, W, background_color, sbias, 
                       depth + 1, max_depth, n_indirect_rays);
    color += 0.5 * rcolor;  // TODO: 0.5 is hyper parameter?
  } else if (tmeshes.data_[k].material_type_ == MaterialType::REFLECTION_AND_REFRACTION) {
    auto ior = tmeshes.data_[k].ior_;
    auto tcolor = make_float3(0.f, 0.f, 0.f);
    auto Kr = fresnel(dir, normal, ior);
    auto outside = dot(dir, normal) < 0;
    auto bias = sbias * normal;
    // refraction (transmission)
    if (Kr < 1) {
      auto dir_new = normalize(refract(dir, normal, ior));
      auto p_new = p + (outside ? -bias : bias);
      tcolor += cast_ray<indirect>(idx, rstate, 
                         p_new, dir_new, tmeshes, dlights, plights, H, W, background_color, sbias, 
                         depth + 1, max_depth, n_indirect_rays);
    }
    // reflection
    auto rcolor = make_float3(0.f, 0.f, 0.f);
    auto dir_new = normalize(reflect(dir, normal));
    auto p_new = p + (outside ? bias : -bias);
    rcolor += cast_ray<indirect>(idx, rstate, 
                       p_new, dir_new, tmeshes, dlights, plights, H, W, background_color, sbias, 
                       depth + 1, max_depth, n_indirect_rays);
    color += Kr * rcolor + (1.f - Kr) * tcolor;
    // color += (1.f - Kr) * tcolor;
  } else {
  }

  return color;
}


// TODO: indirect illumination and environment illumiination
// TODO: accelaration structure
template<bool indirect = false>
__global__
void kernel_render(int num_pixels, 
                   curandState *rstate, 
                   TriangleMeshN tmeshes, 
                   DistantLightN dlights, 
                   PointLightN plights, 
                   float *image, 
                   int H, int W, 
                   float image_aspect_ratio, float scale, 
                   Matrix4x4 M_c2w, 
                   float background_color, float sbias, 
                   int max_depth = 5, 
                   int n_indirect_rays = 0) {
  auto orig = make_float3(M_c2w.data[0][3], M_c2w.data[1][3], M_c2w.data[2][3]);
  UTIL_CUDA_KERNEL_LOOP(idx, num_pixels) {
    // raster to screen
    auto h = idx / W;
    auto w = idx - h * W;
    auto x = (2 * (w + 0.5) / W - 1) * image_aspect_ratio * scale;
    auto y = (1 - 2 * (h + 0.5) / H) * scale;
    auto dir = make_float3(x, y, -1);
    dir = to_world(M_c2w, dir);
    dir = dir - orig;
    dir = normalize(dir);
    // cast ray
    auto color = cast_ray<indirect>(idx, rstate, 
                          orig, dir, 
                          tmeshes, 
                          dlights, 
                          plights, 
                          H, W, 
                          background_color, sbias, 
                          0, max_depth, n_indirect_rays);
    // color
    auto image_0hw = image + (0 * H * W) + (h * W + w);
    auto image_1hw = image + (1 * H * W) + (h * W + w);
    auto image_2hw = image + (2 * H * W) + (h * W + w);
    *image_0hw = color.x * 255.f;
    *image_1hw = color.y * 255.f;
    *image_2hw = color.z * 255.f;
  }
}


int2 to_int2(py::tuple shape) {
  return make_int2(shape[0].cast<int>(), shape[1].cast<int>());
}


void render(Camera camera, Screen screen, 
            std::vector<TriangleMesh> tmeshes, 
            std::vector<DistantLight> dlights, 
            std::vector<PointLight> plights, 
            int64_t image, 
            float background_color, float sbias,
            int max_depth = 5, 
            int n_indirect_rays = 0) {
  TriangleMeshN tmeshes0(tmeshes.size());
  for (auto i = 0; i < tmeshes.size(); i++) {
    tmeshes0.data_[i] = tmeshes[i];
  }
  DistantLightN dlights0(dlights.size());
  for (auto i = 0; i < dlights.size(); i++) {
    dlights0.data_[i] = dlights[i];
  }
  PointLightN plights0(plights.size());
  for (auto i = 0; i < plights.size(); i++) {
    plights0.data_[i] = plights[i];
  }

  auto image_data = reinterpret_cast<float*>(image);

  auto H = camera.image_height_;
  auto W = camera.image_width_;
  auto num_pixels = H * W;
  auto M_c2w = camera.camera_to_world_;
  auto image_aspect_ratio = camera.compute_image_aspect_ratio();
  auto scale = (float)tan(camera.compute_fov(camera.film_aperture_height_) * 0.5 * M_PI / 180);

  // circumvent for the stack overflow in recursion
  size_t pvalue = 0;
  cudaDeviceGetLimit(&pvalue, cudaLimitStackSize);
  printf("default stack size in cuda kernel = %ld\n", pvalue);
  cudaDeviceSetLimit(cudaLimitStackSize, pvalue * 4);
  printf("current stack size in cuda kernel = %ld\n", pvalue * 4);

  auto indirect = n_indirect_rays > 0;

  // RNG for the indirect illumination
  curandState *rstate;
  // Raytrace
  if (indirect) {
    // TODO: array size should be enough the number of threads launched
    cudaMalloc((void **)&rstate, num_pixels * sizeof(curandState));
    UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(setup_kernel, num_pixels, rstate);
    UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_render<true>, num_pixels, 
                                   rstate, 
                                   tmeshes0, 
                                   dlights0, plights0, 
                                   image_data, 
                                   H, W, 
                                   image_aspect_ratio, scale, 
                                   M_c2w, 
                                   background_color, sbias, 
                                   max_depth, n_indirect_rays);
    cudaFree(&rstate);
  } else {
    UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_render<false>, num_pixels, 
                                   rstate, 
                                   tmeshes0, 
                                   dlights0, plights0, 
                                   image_data, 
                                   H, W, 
                                   image_aspect_ratio, scale, 
                                   M_c2w, 
                                   background_color, sbias,
                                   max_depth, n_indirect_rays);
  }

}


using namespace pybind11::literals;

PYBIND11_MODULE(raytracer_cuda_kernel, m) {
  m.doc() = "RayTracer Module";
  m.def("render", &render, "Raytrace triangle mesh.", 
        "camera"_a, "screen"_a, "tmeshes"_a, 
        "dlights"_a=py::list(), "plights"_a=py::list(), 
        "image"_a, 
        "background_color"_a, "sbias"_a, "max_depth"_a=5, "n_indirect_rays"_a=0);
}


/*
$ cd ..
$ nvcc -shared --std=c++11 --compiler-options -fPIC -I./ `python3 -m pybind11 --includes` raytracing/raytracer.cu -o raytracer_cuda_kernel`python3-config --extension-suffix`
 */