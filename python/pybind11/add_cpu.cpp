#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

void add(int64_t a, int64_t b, int64_t c, 
         py::list shape, py::list stride) {
  auto a_data = reinterpret_cast<float*>(a);
  auto b_data = reinterpret_cast<float*>(b);
  auto c_data = reinterpret_cast<float*>(c);
  auto H = shape[0].cast<int>();
  auto W = shape[1].cast<int>();
  for(int i = 0; i < H; i++) {
    auto a_data_i = a_data + i * stride[0].cast<int>();
    auto b_data_i = b_data + i * stride[0].cast<int>();
    auto c_data_i = c_data + i * stride[0].cast<int>();
    for(int j = 0; j < W; j++) {
      auto a_data_ij = *(a_data_i + j);
      auto b_data_ij = *(b_data_i + j);
      //printf("a[%d, %d] = %f\n", i, j, a_data_ij);
      *(c_data_i + j) = a_data_ij + b_data_ij;
    }
  }
}


PYBIND11_MODULE(add_cpu, m) {
  m.doc() = "add method takes poionters";
  m.def("add", &add, "Add for vector.");
}


/*
import numpy as np
import add_cpu
x = np.random.randn(2, 3).astype(np.float32)
y = np.random.randn(2, 3).astype(np.float32)
z = np.random.randn(2, 3).astype(np.float32)

take_pointer.add(x.ctypes.data,  
                 y.ctypes.data, 
                 z.ctypes.data, 
                 list(x.shape), list([s // x.dtype.itemsize for s in x.strides]))
 */
