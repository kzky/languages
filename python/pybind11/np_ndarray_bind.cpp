#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <numeric>
#include <functional>

namespace py = pybind11;

void print_ndarray(py::array_t<double> ndarray) {
  std::cout << "ndarray.itemsize()" << std::endl;
  std::cout << ndarray.itemsize() << std::endl;

  std::cout << "ndarray.ndim()" << std::endl;
  std::cout << ndarray.ndim() << std::endl;

  std::cout << "ndarray.shape()" << std::endl;
  for (auto i = 0; i < ndarray.ndim(); i++) {
    std::cout << ndarray.shape(i) << std::endl;
  }

  std::cout << "ndarray.size()" << std::endl;
  std::cout << ndarray.size() << std::endl;

  std::cout << "ndarray.ptr" << std::endl;
  auto data = (double*)(ndarray.data());
  auto size = ndarray.size();
  for (int i = 0; i < size; i++) {
    std::cout << *data++ << std::endl;
  }
  return;
  
}


PYBIND11_MODULE(np_ndarray_bind, m) {
  m.doc() = "array_t example working with numpy.";

  m.def("print_ndarray", &print_ndarray, "print numpy ndarray.");
}
