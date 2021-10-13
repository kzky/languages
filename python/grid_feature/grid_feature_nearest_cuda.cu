#include <pybind11/pybind11.h>

#include <pybind11/stl.h>
#include <cuda_common.cuh>
#include <helper_math.h>

namespace py = pybind11;

/***
    Forward
 ***/


namespace nearest_by_query {

inline __device__
int3 to_int3(float3 data) {
  return make_int3(data.x, data.y, data.z);
}

inline __device__
float3 to_float3(int3 data) {
  return make_float3(data.x, data.y, data.z);
}

__global__
void kernel_zero(int N, float *data) {
  UTIL_CUDA_KERNEL_LOOP(n, N) {
    data[n] = 0.f;
  }
}

  
template<bool all_nearests = false>
__global__
void kernel_nearest_by_query(int N, float *output, const float *query, const float *feature,
                             int3 grid_sizes, int D,
                             float3 min, float3 max, 
                             bool boundary_check) {
                                 
  auto Gy0 = grid_sizes.y;
  auto Gz0 = grid_sizes.z;
  auto stride_x = Gy0 * Gz0 * D;
  auto stride_y = Gz0 * D;
  auto stride_z = D;
  grid_sizes = grid_sizes - 1;
  
  UTIL_CUDA_KERNEL_LOOP(n, N) {
    auto b = n / D;
    auto d = n - b * D;

    auto querys = make_float3(query[b * 3], query[b * 3 + 1], query[b * 3 + 2]);
  
    // continuous point   
    auto grid_sizes_ = to_float3(grid_sizes);
    auto scales = grid_sizes_ / (max - min);
    auto xyz = (querys - min) * scales;

    // discrete points
    auto xyz0 = floorf(xyz);
    auto xyz1 = xyz0 + 1;

    // scalars
    uint x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;
    uint x1 = xyz1.x, y1 = xyz1.y, z1 = xyz1.z;
    
    // grid features
    auto feature_index = [&](const uint x, const uint y, const uint z) {
      return (x * stride_x) + (y * stride_y) + (z * stride_z + d);
    };
    
    if (all_nearests) {
      auto f000 = feature[feature_index(x0, y0, z0)];
      auto f001 = feature[feature_index(x0, y0, z1)];
      auto f010 = feature[feature_index(x0, y1, z0)];
      auto f011 = feature[feature_index(x0, y1, z1)];
      auto f100 = feature[feature_index(x1, y0, z0)];
      auto f101 = feature[feature_index(x1, y0, z1)];
      auto f110 = feature[feature_index(x1, y1, z0)];
      auto f111 = feature[feature_index(x1, y1, z1)];
      auto bidx = b * (8 * D) + d;
      output[bidx + 0 * D] = f000;
      output[bidx + 1 * D] = f001;
      output[bidx + 2 * D] = f010;
      output[bidx + 3 * D] = f011;
      output[bidx + 4 * D] = f100;
      output[bidx + 5 * D] = f101;
      output[bidx + 6 * D] = f110;
      output[bidx + 7 * D] = f111;
    } else {
      auto f000 = feature[feature_index(x0, y0, z0)];
      output[n] = f000;
    }
    
  }
}


void nearest_by_query(int N, int64_t output_ptr, int64_t query_ptr, int64_t feature_ptr, 
                      std::vector<int> grid_sizes, int D, bool all_nearests, 
                      std::vector<float> min, std::vector<float> max, 
                      bool boundary_check) {
  auto output_buff = reinterpret_cast<float*>(output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);
  auto feature_buff = reinterpret_cast<float*>(feature_ptr);

  auto kernel = all_nearests ? kernel_nearest_by_query<true> : kernel_nearest_by_query<false>;
  UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, 
                                 output_buff, query_buff, feature_buff, 
                                 make_int3(grid_sizes[0], grid_sizes[1], grid_sizes[2]), D, 
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}


/***
    Backward
 ***/

template<bool all_nearests = false>
__global__
void kernel_grad_feature(int N, float *grad_feature, 
                         const float *grad_output, 
                         const float *query, 
                         int3 grid_sizes, int D, 
                         float3 min, float3 max, 
                         bool boundary_check) {
  auto Gy0 = grid_sizes.y;
  auto Gz0 = grid_sizes.z;
  auto stride_x = Gy0 * Gz0 * D;
  auto stride_y = Gz0 * D;
  auto stride_z = D;
  grid_sizes = grid_sizes - 1;
  
  UTIL_CUDA_KERNEL_LOOP(n, N) {
    auto b = n / D;
    auto d = n - b * D;

    auto querys = make_float3(query[b * 3], query[b * 3 + 1], query[b * 3 + 2]);
  
    // continuous point   
    auto grid_sizes_ = to_float3(grid_sizes);
    auto scales = grid_sizes_ / (max - min);
    auto xyz = (querys - min) * scales;

    // discrete points
    auto xyz0 = floorf(xyz);
    auto xyz1 = xyz0 + 1;

    // scalars
    uint x0 = xyz0.x, y0 = xyz0.y, z0 = xyz0.z;
    uint x1 = xyz1.x, y1 = xyz1.y, z1 = xyz1.z;
    
    // gradients
    auto stride = all_nearests ? 8 * D : D;
    auto compute_grad = [&](const uint x, const uint y, const uint z, 
                            const uint i) {
      auto f_idx = x * stride_x + y * stride_y + z * stride_z + d;
      auto o_idx = b * stride + i * D + d;
      atomicAdd(grad_feature + f_idx, grad_output[o_idx]);
    };

    if (all_nearests) {
      compute_grad(x0, y0, z0, 0);
      compute_grad(x0, y0, z1, 1);
      compute_grad(x0, y1, z0, 2);
      compute_grad(x0, y1, z1, 3);
      compute_grad(x1, y0, z0, 4);
      compute_grad(x1, y0, z1, 5);
      compute_grad(x1, y1, z0, 6);
      compute_grad(x1, y1, z1, 7);
    } else {
      compute_grad(x0, y0, z0, 0);
    }
  }
}


void grad_feature(int N, int64_t grad_feature_ptr, 
                  int64_t grad_output_ptr, int64_t query_ptr, 
                  std::vector<int> grid_sizes, int D, bool all_nearests, 
                  std::vector<float> min, std::vector<float> max, 
                  bool boundary_check, bool accum) {
  auto grad_feature_buff = reinterpret_cast<float*>(grad_feature_ptr);
  auto grad_output_buff = reinterpret_cast<float*>(grad_output_ptr);
  auto query_buff = reinterpret_cast<float*>(query_ptr);

  
  if (!accum) {
    auto size = grid_sizes[0] * grid_sizes[1] * grid_sizes[2] * D;
    UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_zero, size, grad_feature_buff);
  }

  auto kernel = all_nearests ? kernel_grad_feature<true> : kernel_grad_feature<false>;
  UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, 
                                 grad_feature_buff, 
                                 grad_output_buff, query_buff, 
                                 make_int3(grid_sizes[0], grid_sizes[1], grid_sizes[2]), D, 
                                 make_float3(min[0], min[1], min[2]),
                                 make_float3(max[0], max[1], max[2]),
                                 boundary_check);
}


}

PYBIND11_MODULE(grid_feature_nearest_cuda, m) {
  m.doc() = "Interpolation by query on grid";
  // forward
  m.def("nearest_by_query", &nearest_by_query::nearest_by_query, "Nearest by query on grid");
  // 1st-order gradient
  m.def("grad_feature", &nearest_by_query::grad_feature, "");
}
