#include <pybind11/pybind11.h>

namespace py = pybind11;

void print_list(py::list data) {
  for (auto i : data) {
    printf("i = %d\n", i.cast<int>());
  }
}


PYBIND11_MODULE(pylist, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
    
  m.def("print_list", &print_list, "A function which takes python list");
}
