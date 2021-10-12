// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include <vector>

__global__
void cuda_round(float *xf, int *xi)
{
  
  xi[threadIdx.x] = round(xf[threadIdx.x]);
  // auto x = xf[threadIdx.x];
  // auto tmp = floor(abs(x) + 0.5);
  // xi[threadIdx.x] = tmp ? tmp : -tmp;
}

int main()
{
  std::vector<float> xf_vec{-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
  std::vector<int> xi_vec(xf_vec.size());
  
  const int fsize = xf_vec.size() * sizeof(float);
  const int isize = xi_vec.size() * sizeof(int);
  float *xf_vec_dev;
  int *xi_vec_dev;
  cudaMalloc((void**)&xf_vec_dev, fsize);
  cudaMalloc((void**)&xi_vec_dev, isize);

  // copy to device
  cudaMemcpy(xf_vec_dev, xf_vec.data(), fsize, cudaMemcpyHostToDevice);
  cudaMemcpy(xi_vec_dev, xi_vec.data(), isize, cudaMemcpyHostToDevice);

  // cuda round
  cuda_round<<<1, xf_vec.size()>>>(xf_vec_dev, xi_vec_dev);

  // copy to host
  cudaMemcpy(xi_vec.data(), xi_vec_dev, isize, cudaMemcpyDeviceToHost);

  printf("before/after cuda round\n");
  for (int i=0; i<xi_vec.size(); i++) {
    printf("val[%d] = %f, %d\n", i, xf_vec[i], xi_vec[i]);
  }

  cudaFree(xf_vec_dev);
  cudaFree(xi_vec_dev);

  return EXIT_SUCCESS;
}
