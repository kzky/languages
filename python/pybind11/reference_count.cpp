#include <pybind11/pybind11.h>


class Child {
public:
  Child() {};
};


class Parent {
public:
  Parent(std::shared_ptr<Child> child) : child(child) {}
  
  std::shared_ptr<Child> child;
  
  std::shared_ptr<Child> get_child() {
    return child;
  }

  void show_child_count() {
    printf("child.use_count() = %d\n", child.use_count());
  }
    
};


class Data {
public:
  bool ref_by_python = false;

  Data() {};
};


/* In python, things like...

In [9]: class Data(reference_count.Data):
   ...:     def __init__(self):
   ...:         reference_count.Data.__init__(self)
   ...:         print("__init__")
   ...:         self.ref_py_python = True
   ...:     def __del__(self):
   ...:         print("__del__")
   ...:         self.ref_py_python = False
 */


namespace py = pybind11;


PYBIND11_MODULE(reference_count, m) {
  m.doc() = "pybind11 ";

  py::class_<Child, std::shared_ptr<Child>>(m, "Child")
    .def(py::init<>());

  py::class_<Parent, std::shared_ptr<Parent>>(m, "Parent")
    .def(py::init<std::shared_ptr<Child>>())
    .def("get_child", &Parent::get_child)
    .def("show_child_count", &Parent::show_child_count);

  py::class_<Data, std::shared_ptr<Data>>(m, "Data")
    .def(py::init<>())
    .def_readwrite("ref_py_python", &Data::ref_by_python);

}
