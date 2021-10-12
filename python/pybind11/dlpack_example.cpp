#include <pybind11/pybind11.h>
#include <dlpack/dlpack.h> // third-party

namespace py = pybind11;

void to_dlpack(py::capsule dlp) {
  auto dl_mtensor = dlp.get_pointer<DLManagedTensor>();
  auto name = dlp.name();
  printf("name = %s\n", name);
  
  printf("ndim = %d\n", dl_mtensor->dl_tensor.ndim);

  for(int i = 0; i < dl_mtensor->dl_tensor.ndim; i++) {
    printf("s = %d\n", dl_mtensor->dl_tensor.shape[i]);
  }
    
}


PYBIND11_MODULE(dlpack_example, m) {
  m.doc() = "pybind11 dltensor"; // optional module docstring
    
  m.def("to_dlpack", &to_dlpack, "A function which takes python list");
}
