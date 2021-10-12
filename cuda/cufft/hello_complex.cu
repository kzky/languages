#include <cufft.h>
#include <cstdio>
#include <cmath>

int main() {
  // Init in cpu
  float *inp_data_cpu = (float*)malloc(sizeof(float) * 2);
  for(int i=0; i<2; i++) {
    inp_data_cpu[i] = i;
  }

  // Copy to device and allocate cuda memory and set data
  cufftComplex *data;
  cudaMalloc((void**)&data, sizeof(cufftComplex) * 1);
  cudaMemcpy(data, inp_data_cpu, sizeof(cufftComplex) * 1, cudaMemcpyHostToDevice);

  // Copy back to cpu
  float *out_data_cpu = (float*)malloc(sizeof(float) * 2);
  cudaMemcpy(out_data_cpu, data, sizeof(cufftComplex) * 1, cudaMemcpyDeviceToHost);
  for(int i=0; i<2; i++) {
    printf("out_data_cpu[%d] = %f\n", i, out_data_cpu[i]);
  }
}
