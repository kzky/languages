#ifndef __RENDER_UTILS_HPP__
#define __RENDER_UTILS_HPP__

#include <math.h>
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace render_utils {

float inch_to_mm = 25.4;


enum FitResolutionGate {
  FILL = 0, 
  OVER_SCREEN = 1
};


class Matrix4x4 {
public:
  float data[4][4];

  Matrix4x4() {};

  Matrix4x4(const Matrix4x4 &M) {
    data[0][0] = M.data[0][0];
    data[0][1] = M.data[0][1];
    data[0][2] = M.data[0][2];
    data[0][3] = M.data[0][3];

    data[1][0] = M.data[1][0];
    data[1][1] = M.data[1][1];
    data[1][2] = M.data[1][2];
    data[1][3] = M.data[1][3];

    data[2][0] = M.data[2][0];
    data[2][1] = M.data[2][1];
    data[2][2] = M.data[2][2];
    data[2][3] = M.data[2][3];

    data[3][0] = M.data[3][0];
    data[3][1] = M.data[3][1];
    data[3][2] = M.data[3][2];
    data[3][3] = M.data[3][3];
  };

};


Matrix4x4 to_matrix4x4(const py::array_t<float> &ndarray) {
  Matrix4x4 M;
  M.data[0][0] = ndarray.at(0, 0);
  M.data[0][1] = ndarray.at(0, 1);
  M.data[0][2] = ndarray.at(0, 2);
  M.data[0][3] = ndarray.at(0, 3);

  M.data[1][0] = ndarray.at(1, 0);
  M.data[1][1] = ndarray.at(1, 1);
  M.data[1][2] = ndarray.at(1, 2);
  M.data[1][3] = ndarray.at(1, 3);

  M.data[2][0] = ndarray.at(2, 0);
  M.data[2][1] = ndarray.at(2, 1);
  M.data[2][2] = ndarray.at(2, 2);
  M.data[2][3] = ndarray.at(2, 3);

  M.data[3][0] = ndarray.at(3, 0);
  M.data[3][1] = ndarray.at(3, 1);
  M.data[3][2] = ndarray.at(3, 2);
  M.data[3][3] = ndarray.at(3, 3);
  
  return M;
}


py::array_t<float> from_matrix4x4(const Matrix4x4 &M) {
  py::array_t<float> ndarray({4, 4});

  ndarray.mutable_at(0, 0) = M.data[0][0];
  ndarray.mutable_at(0, 1) = M.data[0][1];
  ndarray.mutable_at(0, 2) = M.data[0][2];
  ndarray.mutable_at(0, 3) = M.data[0][3];
                  
  ndarray.mutable_at(1, 0) = M.data[1][0];
  ndarray.mutable_at(1, 1) = M.data[1][1];
  ndarray.mutable_at(1, 2) = M.data[1][2];
  ndarray.mutable_at(1, 3) = M.data[1][3];
                  
  ndarray.mutable_at(2, 0) = M.data[2][0];
  ndarray.mutable_at(2, 1) = M.data[2][1];
  ndarray.mutable_at(2, 2) = M.data[2][2];
  ndarray.mutable_at(2, 3) = M.data[2][3];
                  
  ndarray.mutable_at(3, 0) = M.data[3][0];
  ndarray.mutable_at(3, 1) = M.data[3][1];
  ndarray.mutable_at(3, 2) = M.data[3][2];
  ndarray.mutable_at(3, 3) = M.data[3][3];
  
  return ndarray;
}


class Screen {
public:
  float left_ = 0.0f;
  float right_ = 0.0f;
  float top_ = 0.0f;
  float bottom_ = 0.0f;

  Screen() {};

  Screen(float left, float right, float top, float bottom) 
    : left_(left), right_(right), top_(top), bottom_(bottom) {}
  
};


class Camera {
public:
  float focal_length_ = 20;              // in mm 
  float film_aperture_width_ = 0.980;    // 35mm Full Aperture in inches
  float film_aperture_height_ = 0.735;   // 35mm Full Aperture in inches
  float z_near_ = 1;
  float z_far_ = 1000;
  int image_width_ = 640;
  int image_height_ = 480;
  FitResolutionGate k_fit_resolution_gate_ = FitResolutionGate::OVER_SCREEN;

  Matrix4x4 world_to_camera_;
  Matrix4x4 camera_to_world_;

  
  Camera() {};


  Camera(float focal_length, float film_aperture_width, float film_aperture_height, 
         float z_near, float z_far, 
         int image_width, int image_height, FitResolutionGate k_fit_resolution_gate) : 
    focal_length_(focal_length), 
    film_aperture_width_(film_aperture_width), film_aperture_height_(film_aperture_height), 
    z_near_(z_near), z_far_(z_far), 
    image_width_(image_width), image_height_(image_height) {};

  
  float compute_fov(float film_aperture) {
    return 2 * 180 / M_PI * atan((film_aperture * inch_to_mm / 2) / focal_length_);
  }


  float compute_film_aspect_ratio() {
    return 1.0 * film_aperture_width_ / film_aperture_height_;
  }


  float compute_image_aspect_ratio() {
    return 1.0 * image_width_ / image_height_;
  }


  Screen compute_screen() {
    // Scale if necessary
    auto x_scale = 1.0;
    auto y_scale = 1.0;
    auto film_aspect_ratio = this->compute_film_aspect_ratio();
    auto image_aspect_ratio = this->compute_image_aspect_ratio();
    if (k_fit_resolution_gate_ == FitResolutionGate::FILL) {
      if (film_aspect_ratio > image_aspect_ratio)
        x_scale *= image_aspect_ratio / film_aspect_ratio;
      else
        y_scale *= film_aspect_ratio / image_aspect_ratio;
    } else if (k_fit_resolution_gate_ == FitResolutionGate::OVER_SCREEN) {
      if (film_aspect_ratio > image_aspect_ratio)
        y_scale *= film_aspect_ratio / image_aspect_ratio;
      else
        x_scale *= image_aspect_ratio / film_aspect_ratio;
    } else {
      assert(false && "FitResolutionGate is either in [FitResolutionGate.FILL, FitResolutionGate.OVER_SCREEN]");
    }
    // top, right, bottom, left
    auto top = ((film_aperture_height_ * inch_to_mm / 2) / focal_length_) * z_near_;
    auto right = ((film_aperture_width_ * inch_to_mm / 2) / focal_length_) * z_near_;
    top *= y_scale;
    right *= x_scale;
    auto bottom = -top;
    auto left = -right;

    // Screen (Image Plane)
    auto screen = Screen(left=left, right=right, top=top, bottom=bottom);
    return screen;
  }

  
  Matrix4x4 perspective_projection_matrix(const Screen &screen) {
    auto l = screen.left_;
    auto r = screen.right_;
    auto t = screen.top_;
    auto b = screen.bottom_;
    auto n = z_near_;
    auto f = z_far_;

    Matrix4x4 P;
    P.data[0][0] = 2 * n / (r - l);
    P.data[0][1] = 0;
    P.data[0][2] = (r + l) / (r - l);
    P.data[0][3] = 0;

    P.data[1][0] = 0;
    P.data[1][1] = 2 * n / (t - b);
    P.data[1][2] = (t + b) / (t - b);
    P.data[1][3] = 0;

    P.data[2][0] = 0;
    P.data[2][1] = 0;
    P.data[2][2] = -(f + n) / (f - n);
    P.data[2][3] = - (2 * f * n) / (f - n);

    P.data[3][0] = 0;
    P.data[3][1] = 0;
    P.data[3][2] = -1;
    P.data[3][3] = 0;
    
    return P;
  }


  Matrix4x4 orthographic_projection_matrix(const Screen &screen) {
    auto l = screen.left_;
    auto r = screen.right_;
    auto t = screen.top_;
    auto b = screen.bottom_;
    auto n = z_near_;
    auto f = z_far_;

    Matrix4x4 P;
    P.data[0][0] = 2 / (r - l);
    P.data[0][1] = 0;
    P.data[0][2] = 0;
    P.data[0][3] = 0;

    P.data[1][0] = 0;
    P.data[1][1] = 2 / (t - b);
    P.data[1][2] = 0;
    P.data[1][3] = 0;

    P.data[2][0] = 0;
    P.data[2][1] = 0;
    P.data[2][2] = -2 / (f - n);
    P.data[2][3] = 0;

    P.data[3][0] = -(r + l) / (r - l);
    P.data[3][1] = -(t + b) / (t - b);
    P.data[3][2] = -(f + n) / (f - n);
    P.data[3][3] = 1.;
    
    return P;
  }


  void set_camera_to_world(const py::array_t<float> &ndarray) {
    this->set_4x4matrix(ndarray, camera_to_world_);
  }


  Matrix4x4 get_camera_to_world() {
    return camera_to_world_;
  }

  void set_world_to_camera(const py::array_t<float> &ndarray) {
    this->set_4x4matrix(ndarray, world_to_camera_);
  }

  Matrix4x4 get_world_to_camera() {
    return world_to_camera_;
  }


private:
  void set_4x4matrix(const py::array_t<float> &ndarray, Matrix4x4 &M) {
    camera_to_world_.data[0][0] = ndarray.at(0, 0);
    camera_to_world_.data[0][1] = ndarray.at(0, 1);
    camera_to_world_.data[0][2] = ndarray.at(0, 2);
    camera_to_world_.data[0][3] = ndarray.at(0, 3);

    camera_to_world_.data[1][0] = ndarray.at(1, 0);
    camera_to_world_.data[1][1] = ndarray.at(1, 1);
    camera_to_world_.data[1][2] = ndarray.at(1, 2);
    camera_to_world_.data[1][3] = ndarray.at(1, 3);

    camera_to_world_.data[2][0] = ndarray.at(2, 0);
    camera_to_world_.data[2][1] = ndarray.at(2, 1);
    camera_to_world_.data[2][2] = ndarray.at(2, 2);
    camera_to_world_.data[2][3] = ndarray.at(2, 3);

    camera_to_world_.data[3][0] = ndarray.at(3, 0);
    camera_to_world_.data[3][1] = ndarray.at(3, 1);
    camera_to_world_.data[3][2] = ndarray.at(3, 2);
    camera_to_world_.data[3][3] = ndarray.at(3, 3);
  }
};


enum MaterialType {
  DIFFUSE = 0, 
  REFLECTION_AND_REFRACTION = 1, 
  REFLECTION = 2
};


class Light {
public:
  float color_[3];
  float intensity_;

  Light() {};

  Light(float color[3], float intensity) : intensity_(intensity) {
    color_[0] = color[0];
    color_[1] = color[1];
    color_[2] = color[2];
  };

  Light(const Light &light) {
    color_[0] = light.color_[0];
    color_[1] = light.color_[1];
    color_[2] = light.color_[2];
    intensity_ = light.intensity_;
  };

  void set_intensity(float intensity) {
    intensity_ = intensity;
  }

  float get_intensity() {
    return intensity_;
  }

  void set_color(float color[3]) {
    color_[0] = color[0];
    color_[1] = color[1];
    color_[2] = color[2];
  }

  float* get_color() {
    return color_;
  }

  virtual ~Light() {};
};


class DistantLight : public Light {
public:
  float direction_[3];
  
  DistantLight() {};

  DistantLight(float color[3], float intensity, float direction[3]) 
    : Light(color, intensity) {
    auto a = direction[0];
    auto b = direction[1];
    auto c = direction[2];
    auto inv_n = 1.f / sqrt(a * a + b * b + c * c);
    direction_[0] = direction[0] * inv_n;
    direction_[1] = direction[1] * inv_n;
    direction_[2] = direction[2] * inv_n;
  };

  DistantLight(const DistantLight &light) : Light(light) {
    direction_[0] = light.direction_[0];
    direction_[1] = light.direction_[1];
    direction_[2] = light.direction_[2];
  };

  void set_direction(float direction[3]) {
    direction_[0] = direction[0];
    direction_[1] = direction[1];
    direction_[2] = direction[2];
  }

  float* get_direction() {
    return direction_;
  }
  
};


class PointLight : public Light {
public:
  float position_[3];

  PointLight() {};
  
  PointLight(float color[3], float intensity, float position[3]) 
    : Light(color, intensity) {
    position_[0] = position[0];
    position_[1] = position[1];
    position_[2] = position[2];
  };

  PointLight(const PointLight &light) : Light(light) {
    position_[0] = light.position_[0];
    position_[1] = light.position_[1];
    position_[2] = light.position_[2];
  };

  void set_position(float position[3]) {
    position_[0] = position[0];
    position_[1] = position[1];
    position_[2] = position[2];
  }

  float* get_position() {
    return position_;
  }
  
};


class Object {
public:
  MaterialType material_type_;
  float albedo_ = 0.18f;
  // Phong Model
  float Kd_ = 1.f;    // diffuse coefficient
  float Ks_ = 0.08f;  // specular coefficient
  float n_specular_ = 10.f;
  // Index of Refraction
  float ior_ = 1.f;

  Object() {};

  Object(MaterialType material_type) : material_type_(material_type) {};

  ~Object() {};

  void set_material_type(MaterialType material_type) {
    material_type_ = material_type;
  }

  MaterialType get_material_type() {
    return material_type_;
  }

  void set_albedo(float albedo) {
    albedo_ = albedo;
  }

  float get_albedo() {
    return albedo_;
  }

  void set_Kd(float Kd) {
    Kd_ = Kd;
  }

  float get_Kd() {
    return Kd_;
  }

  void set_Ks(float Ks) {
    Ks_ = Ks;
  }

  float get_Ks() {
    return Ks_;
  }

  void set_n_specular(float n_specular) {
    n_specular_ = n_specular;
  }

  float get_n_specular() {
    return n_specular_;
  }

  void set_ior(float ior) {
    ior_ = ior;
  }

  float get_ior() {
    return ior_;
  }

};


class TriangleMesh : public Object {
public:
  // todo: want to use NdArray
  int num_triangles_;
  int num_vertices_;
  int *triangles_;
  float *vertices_;
  float *vertex_colors_ = nullptr;
  float *vertex_uvs_ = nullptr;

  TriangleMesh() : Object() {};
  
  TriangleMesh(int num_triangles, int num_vertices, int64_t triangles, int64_t vertices, int64_t vertex_colors, 
       MaterialType material_type = MaterialType::DIFFUSE) 
    : num_triangles_(num_triangles), num_vertices_(num_vertices),  Object(material_type) {

    triangles_ = reinterpret_cast<int*>(triangles);
    vertices_ = reinterpret_cast<float*>(vertices);
    vertex_colors_ = reinterpret_cast<float*>(vertex_colors);
  };

  TriangleMesh(const TriangleMesh &tmesh) : Object(tmesh){
    num_triangles_ = tmesh.num_triangles_;
    num_vertices_ = tmesh.num_vertices_;
    triangles_ = tmesh.triangles_;
    vertices_ = tmesh.vertices_;
    vertex_colors_ = tmesh.vertex_colors_;
    vertex_uvs_ = tmesh.vertex_uvs_;
  };


};


struct IsectInfo {
public:
  bool hit = false;
  int idx_obj;
  int idx_near;
  float tuv_coords[3];
};


class ShadowMap {
public:
  float *z_buffer_;
  int W_;
  int H_;
  Matrix4x4 M_;
  Matrix4x4 P_;
  DistantLight dlight_;
  
  ShadowMap() {};

  ShadowMap(int64_t z_buffer, int W, int H, DistantLight dlight, Matrix4x4 M, Matrix4x4 P) {
    z_buffer_ = reinterpret_cast<float*>(z_buffer);
    W_ = W;
    H_ = H;
    dlight_ = dlight;
    M_ = M;
    P_ = P;
  };

  ShadowMap(const ShadowMap &self) {
    z_buffer_ = self.z_buffer_;
    W_ = self.W_;
    H_ = self.H_;
    dlight_ = self.dlight_;
    M_ = self.M_;
    P_ = self.P_;
  };


  void set_W(int W) {W_ = W;};
  
  float get_W() {return W_;};

  void set_H(int H) {H_ = H;};
  
  float get_H() {return H_;};


};

}
#endif
