#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <render_utils.hpp>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace render_utils;

PYBIND11_MODULE(render_utils, m) {
  m.doc() = "rendering utilities";

  py::class_<Matrix4x4, std::shared_ptr<Matrix4x4>>(m, "Matrix4x4")
    .def("data", [](const Matrix4x4 &matrix4x4) {
        auto x = py::array_t<float>({4, 4});
        auto r = x.mutable_unchecked<2>();
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            r(i, j) = matrix4x4.data[i][j];
          }
        }
        return x;
      });

  m.def("to_matrix4x4", &to_matrix4x4);
  m.def("from_matrix4x4", &from_matrix4x4);

  py::enum_<FitResolutionGate>(m, "FitResolutionGate")
    .value("FILL", FitResolutionGate::FILL)
    .value("OVER_SCREEN", FitResolutionGate::OVER_SCREEN)
    .export_values();

  py::class_<Screen>(m, "Screen")
    .def(py::init<float, float, float, float>())
    .def_readwrite("left", &Screen::left_)
    .def_readwrite("right", &Screen::right_)
    .def_readwrite("top", &Screen::top_)
    .def_readwrite("bottom", &Screen::bottom_);

  py::class_<Camera>(m, "Camera")
    .def(py::init<>())
    .def(py::init<float, float, float, 
         float, float, 
         int, int, FitResolutionGate>())
    .def_readwrite("focal_length", &Camera::focal_length_)
    .def_readwrite("film_aperture_width", &Camera::film_aperture_width_)
    .def_readwrite("film_aperture_height", &Camera::film_aperture_height_)
    .def_readwrite("z_near", &Camera::z_near_)
    .def_readwrite("z_far", &Camera::z_far_)
    .def_readwrite("image_width", &Camera::image_width_)
    .def_readwrite("image_height", &Camera::image_height_)
    .def_readwrite("k_fit_resolution_gate", &Camera::k_fit_resolution_gate_)
    .def("compute_fov", &Camera::compute_fov)
    .def("compute_film_aspect_ratio", &Camera::compute_film_aspect_ratio)
    .def("compute_image_aspect_ratio", &Camera::compute_image_aspect_ratio)
    .def("compute_screen", &Camera::compute_screen)
    .def("perspective_projection_matrix", &Camera::perspective_projection_matrix)
    .def("orthographic_projection_matrix", &Camera::orthographic_projection_matrix)
    .def_property("camera_to_world", &Camera::get_camera_to_world, &Camera::set_camera_to_world)
    .def_property("world_to_camera", &Camera::get_camera_to_world, &Camera::set_camera_to_world);

  py::enum_<MaterialType>(m, "MaterialType")
    .value("DIFFUSE", MaterialType::DIFFUSE)
    .value("REFLECTION_AND_REFRACTION", MaterialType::REFLECTION_AND_REFRACTION)
    .value("REFLECTION", MaterialType::REFLECTION)
    .export_values();

  py::class_<Object, std::shared_ptr<Object>>(m, "Object")
    .def(py::init<MaterialType>())
    .def_property("material_type", &Object::get_material_type, &Object::set_material_type)
    .def_property("albedo", &Object::get_albedo, &Object::set_albedo)
    .def_property("Kd", &Object::get_Kd, &Object::set_Ks)
    .def_property("Ks", &Object::get_Ks, &Object::set_Ks)
    .def_property("n_specular", &Object::get_n_specular, &Object::set_n_specular)
    .def_property("ior", &Object::get_ior, &Object::set_ior);

  py::class_<TriangleMesh, Object, std::shared_ptr<TriangleMesh>>(m, "TriangleMesh")
    .def(py::init<int, int, int64_t, int64_t, int64_t, MaterialType>())
    .def_readonly("num_triangles", &TriangleMesh::num_triangles_)
    .def_readonly("num_vertices", &TriangleMesh::num_vertices_);

  py::class_<ShadowMap>(m, "ShadowMap")
    .def(py::init<>())
    .def(py::init<int64_t, int, int, DistantLight, Matrix4x4, Matrix4x4>())
    .def_property("W", &ShadowMap::get_W, &ShadowMap::set_W)
    .def_property("H", &ShadowMap::get_H, &ShadowMap::set_H);

  py::class_<Light>(m, "Light")
    .def(py::init([](py::list color, float intensity) {
          float color_[3];
          color_[0] = color[0].cast<float>();
          color_[1] = color[1].cast<float>();
          color_[2] = color[2].cast<float>();
          return std::unique_ptr<Light>(new Light(color_, intensity));
        }))
    .def_property("intensity", &Light::get_intensity, &Light::set_intensity)
    .def_property("color", [](const Light &light){
        py::list ret;
        ret.append(light.color_[0]);
        ret.append(light.color_[1]);
        ret.append(light.color_[2]);
        return ret;
      }, [](Light &light, py::list color) {
        light.color_[0] = color[0].cast<float>();
        light.color_[1] = color[1].cast<float>();
        light.color_[2] = color[2].cast<float>();
      });

  py::class_<DistantLight, Light>(m, "DistantLight")
    .def(py::init([](py::list color, float intensity, py::list direction) {
          float color_[3];
          color_[0] = color[0].cast<float>();
          color_[1] = color[1].cast<float>();
          color_[2] = color[2].cast<float>();
          float direction_[3];
          direction_[0] = direction[0].cast<float>();
          direction_[1] = direction[1].cast<float>();
          direction_[2] = direction[2].cast<float>();
          return std::unique_ptr<DistantLight>(new DistantLight(color_, intensity, direction_));
        }))
    .def_property("direction", [](const DistantLight &light){
        py::list ret;
        ret.append(light.direction_[0]);
        ret.append(light.direction_[1]);
        ret.append(light.direction_[2]);
        return ret;
      }, [](DistantLight &light, py::list direction) {
        light.direction_[0] = direction[0].cast<float>();
        light.direction_[1] = direction[1].cast<float>();
        light.direction_[2] = direction[2].cast<float>();
      });


  py::class_<PointLight, Light>(m, "PointLight")
    .def(py::init([](py::list color, float intensity, py::list position) {
          float color_[3];
          color_[0] = color[0].cast<float>();
          color_[1] = color[1].cast<float>();
          color_[2] = color[2].cast<float>();
          float position_[3];
          position_[0] = position[0].cast<float>();
          position_[1] = position[1].cast<float>();
          position_[2] = position[2].cast<float>();
          return std::unique_ptr<PointLight>(new PointLight(color_, intensity, position_));
        }))
    .def_property("position", [](const PointLight &light){
        py::list ret;
        ret.append(light.position_[0]);
        ret.append(light.position_[1]);
        ret.append(light.position_[2]);
        return ret;
      }, [](PointLight &light, py::list position) {
        light.position_[0] = position[0].cast<float>();
        light.position_[1] = position[1].cast<float>();
        light.position_[2] = position[2].cast<float>();
      });


}

