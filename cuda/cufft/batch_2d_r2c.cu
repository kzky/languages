#include <cufft.h>
#include <cufftXt.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <random>

#define BATCH 4  // Batch size
#define NX 8   // Number of data points of the signal
#define NY 4   // Number of data points of the signal


int main() {
  
  // Array size in bytes
  int size_inp = sizeof(float) * BATCH * NX * NY;
  int size_out = sizeof(cufftComplex) * BATCH * NX * (NY/2 + 1);
  printf("size_inp = %d\n", size_inp);
  printf("size_out = %d\n", size_out);

  // Create plan
  cufftHandle plan;
  cufftCreate(&plan);

  // Arguments for cufftXtMakePlanMany
  int rank = 2;
  std::vector<long long int> n = {NX, NY};
  std::vector<long long int> inembed = {NX, NY};
  long long int istride = 1;
  long long int idist = NX * NY;
  cudaDataType input_type = CUDA_R_32F;
  std::vector<long long int> onembed = {NX, NY/2 + 1};
  long long int ostride = 1;
  long long int odist = NX * (NY/2 + 1);
  cudaDataType output_type = CUDA_C_32F;
  long long int batch = BATCH;
  size_t work_size = 0;
  cudaDataType execution_type = CUDA_C_32F;  // cudaDataType is correct

  // Make Plan
  printf("work_size (before) = %lu\n", work_size);
  cufftXtMakePlanMany(plan,
                      rank, n.data(),
                      inembed.data(), istride, idist, input_type,
                      onembed.data(), ostride, odist, output_type,
                      batch, &work_size, execution_type);
  printf("work_size (after) = %lu\n", work_size);

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
  // std::random_device seed_gen;
  // std::mt19937 engine(seed_gen());
  // std::uniform_real_distribution<> dist1(-1.0, 1.0);
  for (int i = 0; i < BATCH; i++) {
    for (int j = 0; j < NX; j++) {
      for (int k = 0; k < NY; k++) {
        int idx = k + j * NY + i * NX * NY;
        inp_data_cpu[idx] = 0.6 * std::sin(2. * pi * i / freq) + 0.4 * std::cos(2. * pi * i / freq);
        //inp_data_cpu[idx] = (float)dist1(engine);
      }
    }
  }

  // Copy cpu to device
  cudaMemcpy(inp_data_device, inp_data_cpu, size_inp, cudaMemcpyHostToDevice);

  // Execute FFT 2d
  cufftExecR2C(plan, inp_data_device, out_data_device);

  // Copy back device to cpu
  cudaMemcpy(out_data_cpu, out_data_device, size_out, cudaMemcpyDeviceToHost);

  // Show results
  for (int i=0; i < BATCH*NX*(NY/2+1); i++) {
    printf("out_data_cpu[%d] = %f\n", i, out_data_cpu[i]);
  }

  cufftDestroy(plan);
  cudaFree(inp_data_device);
  cudaFree(out_data_device);
  free(inp_data_cpu);
  free(out_data_cpu);


}
