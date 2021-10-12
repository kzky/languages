#include <pybind11/pybind11.h>
#include <iostream>
#include <numeric>
#include <functional>

namespace py = pybind11;

void print_buffer(py::buffer buffer) {
  py::buffer_info info = buffer.request();
  std::cout << "info.itemsize" << std::endl;
  std::cout << info.itemsize << std::endl;

  std::cout << "info.format" << std::endl;
  std::cout << info.format << std::endl;

  std::cout << "info.ndim" << std::endl;
  std::cout << info.ndim << std::endl;

  std::cout << "info.shape" << std::endl;
  for (auto s : info.shape) {
    std::cout << s << std::endl;
  }
  std::cout << "info.strides" << std::endl;
  for (auto s : info.strides) {
    std::cout << s << std::endl;
  }

  std::cout << "info.ptr" << std::endl;
  auto data = (double*)(info.ptr);
  auto size = std::accumulate(info.shape.begin(), info.shape.end(), 1, std::multiplies<py::ssize_t>());
  for (int i = 0; i < size; i++) {
    std::cout << *data++ << std::endl;
  }
  return;
}


PYBIND11_MODULE(python_buffer, m) {
  m.doc() = "python buffer object example working with numpy.";

  m.def("print_buffer", &print_buffer, "print python buffer");
}
