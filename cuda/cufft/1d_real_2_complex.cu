#include <cufft.h>
#include <cstdio>
#include <cmath>

#define NX 256   // Number of data points of the signal
#define BATCH 1  // Batch size

int main() {
  // Array size in bytes
  int size_inp = sizeof(float) * NX * BATCH;
  int size_out =size_inp / 2 + 1;
  printf("size_inp = %d\n", size_inp);
  printf("size_out = %d\n", size_out);

  // Plan of cufft
  cufftHandle plan;
  cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH);  // TODO: error check

  // Data in GPU
  cufftReal *inp_data_device;
  cudaMalloc((void**)&inp_data_device, size_inp);
  cufftComplex *out_data_device;
  cudaMalloc((void**)&out_data_device, size_out);

  // Data in CPU
  float *inp_data_cpu = (float*)malloc(size_inp);
  float *out_data_cpu = (float*)malloc(size_out);

  // Sign curve
  const float pi = std::acos(-1);
  float freq = 32;
  for (int i = 0; i < NX * BATCH; i++) {
    inp_data_cpu[i] = std::sin(2. * pi * i / freq);
//    printf("inp_data_cpu[%d] = %f\n", i, inp_data_cpu[i]);
  }

  // Copy cpu to device
  cudaMemcpy(inp_data_device, inp_data_cpu, size_inp, cudaMemcpyHostToDevice);

  // Execute FFT 1d
  cufftExecR2C(plan, inp_data_device, out_data_device);

  // Copy back device to cpu
  cudaMemcpy(out_data_cpu, out_data_device, size_out, cudaMemcpyDeviceToHost);


  cufftDestroy(plan);
  cudaFree(inp_data_device);
  cudaFree(out_data_device);
  free(inp_data_cpu);
  free(out_data_cpu);
  
  
}
