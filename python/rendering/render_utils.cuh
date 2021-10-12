#ifndef __RENDER_UTILS_CUH__
#define __RENDER_UTILS_CUH__

#include <render_utils.hpp>
#include <math.h>
#include <curand_kernel.h>
#include <curand.h>

namespace render_utils_cuda {

using namespace render_utils;

template<typename T>
struct Triangle {

public:
  T v0;
  T v1;
  T v2;

  __device__
  Triangle() {};

  __device__
  Triangle(T v0, T v1, T v2) {
    v0 = v0;
    v1 = v1;
    v2 = v2;
  };

  __device__
  Triangle(const Triangle &obj) {
    v0 = obj.v0;
    v1 = obj.v1;
    v2 = obj.v2;
  };

  __device__
  void inv_z() {
    v0.z = 1.f / v0.z;
    v1.z = 1.f / v1.z;
    v2.z = 1.f / v2.z;
  }

};


template<typename T>
struct Edges {

public:
  T e0;
  T e1;
  T e2;

  __device__
  Edges() {};


  __device__
  Edges(const Triangle<T> &triangle) {
    e0 = triangle.v2 - triangle.v1;
    e1 = triangle.v0 - triangle.v2;
    e2 = triangle.v1 - triangle.v0;
  };


  __device__
  Edges(T e0, T e1, T e2) {
    e0 = e0;
    e1 = e1;
    e2 = e2;
  };

  __device__
  Edges(const Edges &obj) {
    e0 = obj.e0;
    e1 = obj.e1;
    e2 = obj.e2;
  };

};


template<typename T>
struct BarycentricCoords {

public:
  T w0;
  T w1;
  T w2;

  __device__
  BarycentricCoords() {};

  __device__
  BarycentricCoords(T w0, T w1, T w2) {
    w0 = w0;
    w1 = w1;
    w2 = w2;
  };

  __device__
  BarycentricCoords(const BarycentricCoords &obj) {
    w0 = obj.w0;
    w1 = obj.w1;
    w2 = obj.w2;
  };


  __device__
  void devide(T area) {
    w0 /= area;
    w1 /= area;
    w2 /= area;
  }

};


inline __device__
float min3(float a, float b, float c) {
  return min(a, min(b, c));
}


inline __device__
float max3(float a, float b, float c) {
  return max(a, max(b, c));
}


struct AABB {

public:
  float2 min;
  float2 max;

  __device__
  AABB() {};

  __device__
  AABB(const Triangle<float3> &triangle) {
    auto xmin = min3(triangle.v0.x, triangle.v1.x, triangle.v2.x);
    auto xmax = max3(triangle.v0.x, triangle.v1.x, triangle.v2.x);
    auto ymin = min3(triangle.v0.y, triangle.v1.y, triangle.v2.y);
    auto ymax = max3(triangle.v0.y, triangle.v1.y, triangle.v2.y);
    min = make_float2(xmin, ymin);
    max = make_float2(xmax, ymax);
  };

  __device__
  AABB(const AABB &obj) {
    min = obj.min;
    max = obj.max;
  };

};

struct AABBi {

public:
  int2 mini;
  int2 maxi;

  __device__
  AABBi() {};

  __device__
  AABBi(AABB aabb, int H, int W) {
    auto x0 = max(int(floor(aabb.min.x)), 0);
    auto x1 = min(int(floor(aabb.max.x)) + 1, W);
    auto y0 = max(int(floor(aabb.min.y)), 0);
    auto y1 = min(int(floor(aabb.max.y)) + 1, H);
    mini.x = x0;
    mini.y = y0;
    maxi.x = x1;
    maxi.y = y1;
  };
  
};


inline __device__
float4 transform(Matrix4x4 &T, float4 v) {
  float x = T.data[0][0] * v.x + T.data[0][1] * v.y + T.data[0][2] * v.z + T.data[0][3] * v.w;
  float y = T.data[1][0] * v.x + T.data[1][1] * v.y + T.data[1][2] * v.z + T.data[1][3] * v.w;
  float z = T.data[2][0] * v.x + T.data[2][1] * v.y + T.data[2][2] * v.z + T.data[2][3] * v.w;
  float w = T.data[3][0] * v.x + T.data[3][1] * v.y + T.data[3][2] * v.z + T.data[3][3] * v.w;
  return make_float4(x, y, z, w);
}


inline __device__
float4 transform(Matrix4x4 &T, float3 v) {
  float x = T.data[0][0] * v.x + T.data[0][1] * v.y + T.data[0][2] * v.z + T.data[0][3];
  float y = T.data[1][0] * v.x + T.data[1][1] * v.y + T.data[1][2] * v.z + T.data[1][3];
  float z = T.data[2][0] * v.x + T.data[2][1] * v.y + T.data[2][2] * v.z + T.data[2][3];
  return make_float4(x, y, z, 1.f);
}


inline __device__
Triangle<float4> transform(Matrix4x4 &T, Triangle<float3> triangle) {
  Triangle<float4> triangle0;
  triangle0.v0 = transform(T, triangle.v0);
  triangle0.v1 = transform(T, triangle.v1);
  triangle0.v2 = transform(T, triangle.v2);
  return triangle0;
}


inline __device__
float3 to_coords(const float *vp) {
  return make_float3(*vp, *(vp + 1), *(vp + 2));
}



inline __device__
Triangle<float3> to_triangle(const float *vertices, const int *triangle) {
  Triangle<float3> triangle0;
  triangle0.v0 = to_coords(vertices + (triangle[0] * 3));
  triangle0.v1 = to_coords(vertices + (triangle[1] * 3));
  triangle0.v2 = to_coords(vertices + (triangle[2] * 3));
  return triangle0;
}


inline __device__
float4 to_homogeneous_coords(const float *vp) {
  return make_float4(*vp, *(vp + 1), *(vp + 2), 1.f);
}


inline __device__
Triangle<float4> to_homogeneous_coords(const float *vertices, const int *triangle) {
  Triangle<float4> triangle0;
  triangle0.v0 = to_homogeneous_coords(vertices + (triangle[0] * 3));
  triangle0.v1 = to_homogeneous_coords(vertices + (triangle[1] * 3));
  triangle0.v2 = to_homogeneous_coords(vertices + (triangle[2] * 3));
  return triangle0;
}

inline __device__
float4 to_homogeneous_clipping_coords(const Matrix4x4 &P, const float4 &v) {
  float x = P.data[0][0] * v.x + P.data[0][1] * v.y + P.data[0][2] * v.z + P.data[0][3];
  float y = P.data[1][0] * v.x + P.data[1][1] * v.y + P.data[1][2] * v.z + P.data[1][3];
  float z = P.data[2][0] * v.x + P.data[2][1] * v.y + P.data[2][2] * v.z + P.data[2][3];
  float w = P.data[3][0] * v.x + P.data[3][1] * v.y + P.data[3][2] * v.z + P.data[3][3];
  return make_float4(x, y, z, w);
}


inline __device__
Triangle<float4> to_homogeneous_clipping_coords(const Matrix4x4 &P, const Triangle<float4> &triangle) {
  Triangle<float4> triangle0;
  triangle0.v0 = to_homogeneous_clipping_coords(P, triangle.v0);
  triangle0.v1 = to_homogeneous_clipping_coords(P, triangle.v1);
  triangle0.v2 = to_homogeneous_clipping_coords(P, triangle.v2);
  return triangle0;
}


inline __device__
float3 to_unit_cube_coords(const float4 &v) {
  float inv_w = 1.f / v.w;
  return make_float3(v.x * inv_w, v.y * inv_w, v.z * inv_w);
}


inline __device__
Triangle<float3> to_unit_cube_coords(const Triangle<float4> &triangle) {
  Triangle<float3> triangle0;
  triangle0.v0 = to_unit_cube_coords(triangle.v0);
  triangle0.v1 = to_unit_cube_coords(triangle.v1);
  triangle0.v2 = to_unit_cube_coords(triangle.v2);
  return triangle0;
}


inline __device__
float3 to_raster_coords(float3 v, int H, int W) {
  return make_float3((v.x + 1) * 0.5f * W, (1 - v.y) * 0.5f * H, v.z);
}


inline __device__
Triangle<float3> to_raster_coords(const Triangle<float3> &triangle, int H, int W) {
  Triangle<float3> triangle0;
  triangle0.v0 = to_raster_coords(triangle.v0, H, W);
  triangle0.v1 = to_raster_coords(triangle.v1, H, W);
  triangle0.v2 = to_raster_coords(triangle.v2, H, W);
  return triangle0;
}


inline __device__
float3 to_unit_cube_to_raster_coords(const float4 &v, int H, int W) {
  float inv_w = 1.f / v.w;
  auto u = make_float3(v.x * inv_w, v.y * inv_w, v.z * inv_w);
  return make_float3((u.x + 1) * 0.5f * W, (1 - (u.y + 1) * 0.5f) * H, u.z);
}


inline __device__
float4 w_divide(const float4 v) {
  return make_float4(v.x / v.w, v.y / v.w, v.z / v.w, 1.f);
}


inline __device__
bool clipping_condition(const float4 &v) {
  bool cond = true;
  cond &= (v.w > 0);
  cond &= (v.x >= -v.w) && (v.x <= v.w);
  cond &= (v.y >= -v.w) && (v.y <= v.w);
  cond &= (v.z >= -v.w) && (v.z <= v.w);
  return cond;
}


inline __device__
bool clipping_condition(const Triangle<float4> &triangle) {
  bool cond = true;
  cond |= clipping_condition(triangle.v0);
  cond |= clipping_condition(triangle.v1);
  cond |= clipping_condition(triangle.v2);
  return cond;
}


inline __device__
float edge_function(const float2 &a, const float2 &b, const float2 &c) {
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}


inline __device__
float edge_function(const float3 &a, const float3 &b, const float3 &c) {
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}


inline __device__
float edge_function(const Triangle<float3> &triangle) {
  auto a = triangle.v0;
  auto b = triangle.v1;
  auto c = triangle.v2;
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}


inline __device__
float3 edge_function(const Triangle<float3> &triangle, const float3 &pixel) {
  auto v0 = triangle.v0;
  auto v1 = triangle.v1;
  auto v2 = triangle.v2;
  auto w0 = edge_function(v1, v2, pixel);
  auto w1 = edge_function(v2, v0, pixel);
  auto w2 = edge_function(v0, v1, pixel);
  return make_float3(w0, w1, w2);
}


template<bool top_left = true>
inline __device__
bool inside_test(const float3 &bc_coords, const Edges<float3> &edges) {
  const auto w0 = bc_coords.x;
  const auto w1 = bc_coords.y;
  const auto w2 = bc_coords.z;
  const auto e0 = edges.e0;
  const auto e1 = edges.e1;
  const auto e2 = edges.e2;
  auto overlap = true;
  overlap &= (w0 == 0.f) ? ((e0.y == 0 && e0.x > 0) || e0.y > 0) : w0 > 0.f;
  overlap &= (w1 == 0.f) ? ((e1.y == 0 && e1.x > 0) || e1.y > 0) : w1 > 0.f;
  overlap &= (w2 == 0.f) ? ((e2.y == 0 && e2.x > 0) || e2.y > 0) : w2 > 0.f;
  return overlap;      
}


inline __device__
Edges<float3> to_edges(const Triangle<float3> &triangle) {
  Edges<float3> edges(triangle);
  return edges;
}



inline __device__
float sum(float3 v) {
  return v.x + v.y + v.z;
}


inline __device__
float3 weighted_sum(const Triangle<float3> triangle, const float3 &bc_coords) {
  auto v = triangle.v0 * bc_coords.x + triangle.v1 * bc_coords.y + triangle.v2 * bc_coords.z;
  return v;
}


inline __device__
void add_color(float *image, const float3 &color, const int y, const int x, const int H, const int W) {
  auto image_0hw = image + (0 * H * W) + (y * W) + x;
  auto image_1hw = image + (1 * H * W) + (y * W) + x;
  auto image_2hw = image + (2 * H * W) + (y * W) + x;
  *image_0hw = color.x * 255.f;
  *image_1hw = color.y * 255.f;
  *image_2hw = color.z * 255.f;
}


inline __device__
float3 to_world(const Matrix4x4 &M_c2w, const float3 &v) {
  float x = M_c2w.data[0][0] * v.x + M_c2w.data[0][1] * v.y + M_c2w.data[0][2] * v.z + M_c2w.data[0][3];
  float y = M_c2w.data[1][0] * v.x + M_c2w.data[1][1] * v.y + M_c2w.data[1][2] * v.z + M_c2w.data[1][3];
  float z = M_c2w.data[2][0] * v.x + M_c2w.data[2][1] * v.y + M_c2w.data[2][2] * v.z + M_c2w.data[2][3];
  return make_float3(x, y, z);
}


inline __device__
float3 compute_normal_and_bias(float4 u, float4 v) {
  //TODO: can we work directly on clipping space?
  auto u2 = make_float2(u.x, u.y);
  auto v2 = make_float2(v.x, v.y);
  auto n = make_float2(v2.y - u2.y, u2.x - v2.x); // n = R90 * (u - v)
  n = normalize(n);
  auto d = -dot(n, make_float2(u2.x, u2.y));
  return make_float3(n.x, n.y, d);
}


inline __device__
Triangle<float4> conservative(Triangle<float4> triangle, float shift = 1) {
  Triangle<float4> ctriangle = triangle;
  // (a, b, c) homogeneous coords (slope, bias)
  auto n0 = compute_normal_and_bias(triangle.v1, triangle.v2);
  auto n1 = compute_normal_and_bias(triangle.v2, triangle.v0);
  auto n2 = compute_normal_and_bias(triangle.v0, triangle.v1);
  auto c = make_float2(shift, shift);
  n0.z -= dot(c, fabs(make_float2(n0.x, n0.y)));
  n1.z -= dot(c, fabs(make_float2(n1.x, n1.y)));
  n2.z -= dot(c, fabs(make_float2(n2.x, n2.y)));
  // (x, y, z) homogeneous coords, so devide at last
  auto v0_xyz = cross(n1, n2);
  auto v1_xyz = cross(n2, n0);
  auto v2_xyz = cross(n0, n1);
  ctriangle.v0.x = v0_xyz.x / v0_xyz.z;
  ctriangle.v0.y = v0_xyz.y / v0_xyz.z;
  ctriangle.v1.x = v1_xyz.x / v1_xyz.z;
  ctriangle.v1.y = v1_xyz.y / v1_xyz.z;
  ctriangle.v2.x = v2_xyz.x / v2_xyz.z;
  ctriangle.v2.y = v2_xyz.y / v2_xyz.z;
  return ctriangle;
}


inline __device__
float3 normals(float3 v0, float3 v1, float3 v2) {
  return normalize(cross(v2 - v0, v1 - v0));
}

inline __device__
float3 normals(Triangle<float3> triangle) {
  return normalize(cross(triangle.v2 - triangle.v0, triangle.v1 - triangle.v0));
}


// inline __device__
// float3 reflect(float3 normal, float3 ilight) {
//   return 2.f * dot(normal, ilight) * normal - ilight;
// }


inline __device__
void swap(float &a, float &b) {
  float tmp = a;
  a = b;
  b = tmp;
}


inline __device__
float fresnel(float3 i, float3 n, float ior) {
  auto cosi = clamp(dot(i, n), -1.f, 1.f);
  auto etai = 1.f; // TODO: have to track the refraction index of the incidnet ray generally
  auto etat = ior;
  if (cosi > 0)
    swap(etai, etat);
  // Snell's law
  auto sint = etai / etat * sqrt(max(0.f, 1.f - cosi * cosi));
  // total reflection
  if (sint >= 1) {
    auto Kr = 1;
    return Kr;
  }
  auto cost = sqrt(max(0.f, 1.f - sint * sint));
  cosi = abs(cosi);
  auto Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
  auto Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
  auto Kr = (Rs * Rs + Rp * Rp) / 2.f;
  return Kr;
}


inline __device__
float3 refract(float3 i, float3 n, float ior) {
  auto cosi = clamp(dot(i, n), -1.f, 1.f);
  auto etai = 1.f; // TODO: have to track the refraction index of the incidnet ray generally
  auto etat = ior;
  if (cosi < 0) {
    cosi = -cosi;
  } else {
    swap(etai, etat);
    n = -n;
  }
  auto eta = etai / etat;
  auto k = 1.f - eta * eta * (1.f - cosi * cosi);
  return k < 0.f ? make_float3(0.f, 0.f, 0.f) : eta * i + (eta * cosi - sqrt(k)) * n;
}


inline __device__
float lookup_shadow(float2 uv, int H, int W, float *shadow_map) {
  // coords
  auto u = clamp(uv.x * W, 0.f, float(W - 1));
  auto v = clamp(uv.y * H, 0.f, float(H - 1));
  int u0 = int(floor(u));
  int v0 = int(floor(v));
  int u1 = min(u0 + 1, W - 1);
  int v1 = min(v0 + 1, H - 1);
  // coefs
  float p0 = u - u0;
  float q0 = v - v0;
  float p1 = 1 - p0;
  float q1 = 1 - q0;
  // values
  auto z_u0v0 = *(shadow_map + (v0 * W) + u0);
  auto z_u0v1 = *(shadow_map + (v1 * W) + u0);
  auto z_u1v0 = *(shadow_map + (v0 * W) + u1);
  auto z_u1v1 = *(shadow_map + (v1 * W) + u1);
  // interp
  auto z = (z_u0v0 * p1 * q1) + (z_u0v1 * p1 * q0) + (z_u1v0 * p0 * q1) + (z_u1v1 * p0 * q0);
  return z;
}


inline __device__
bool in_shadow(float4 v, int H, int W, float* shadow_map) {
  auto z = lookup_shadow(make_float2(v.x, v.y), H, W, shadow_map);
  return z < v.z;
}


class TriangleMeshN {
public:
  TriangleMesh data_[10];
  int n_;
  TriangleMeshN(int n) : n_(n) {};
};


class DistantLightN {
public:
  DistantLight data_[10];
  int n_;
  
  DistantLightN(int n) : n_(n) {};
};


class PointLightN {
public:
  PointLight data_[10];
  int n_;
  
  PointLightN(int n) : n_(n) {};
};


class ShadowMapN {
public:
  ShadowMap data_[10];
  int n_;
  ShadowMapN(int n) : n_(n) {};
};


inline __device__
float3 to_float3(float *data) {
  return make_float3(data[0], data[1], data[2]);
};


inline __device__
float3 sample_direction(const float r1, const float r2) {
  // sample direction on the hemisphere
  auto sin_theta = sqrtf(1.f - r1 * r1);
  auto phi = 2.f * M_PI * r2;
  auto x = sin_theta * cosf(phi);
  auto z = sin_theta * sinf(phi);
  auto sample = make_float3(x, r1, z);
  return normalize(sample);
};


inline __device__
void create_coordinate_system(const float3 &N, float3 &Nx, float3 &Nz) {
  if (fabs(N.x) > fabs(N.y))  {
    auto inv_norm = 1.f / sqrtf(N.x * N.x + N.z * N.z);
    Nx = make_float3(N.z, 0.f, -N.x) * inv_norm;
    Nx.x = N.z * inv_norm;
    Nx.z = -N.x * inv_norm;
  } else {
    auto inv_norm = 1.f / sqrtf(N.y * N.y + N.z * N.z); 
    Nx = make_float3(0.f, -N.z, N.y) * inv_norm;
    Nx.y = -N.z * inv_norm;
    Nx.z = N.y * inv_norm;
  }
  auto _Nz = cross(N, Nx);
  Nz.x = _Nz.x;
  Nz.y = _Nz.y;
  Nz.z = _Nz.z;
};


inline __device__
float3 transform_by_coordinate_system(const float3 &rdir, 
                                      const float3 &Nx, const float3 &Ny, const float3 &Nz) {
  auto x = Nx.x * rdir.x + Ny.x * rdir.y + Nz.x * rdir.z;
  auto y = Nx.y * rdir.x + Ny.y * rdir.y + Nz.y * rdir.z;
  auto z = Nx.z * rdir.x + Ny.z * rdir.y + Nz.z * rdir.z;
  return normalize(make_float3(x, y, z));
}

}

#endif