#include <pybind11/pybind11.h>

struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
};

namespace py = pybind11;

PYBIND11_MODULE(example_class, m) {
  m.doc() = "pybind11 class example";

  py::class_<Pet>(m, "PetPy", py::dynamic_attr())
    .def(py::init<const std::string &>())
    .def("set_name", &Pet::setName)
    .def("get_name", &Pet::getName)
    .def_readwrite("name", &Pet::name);
}
