#ifndef __UTIL_CUDA_COMMON_HPP__
#define __UTIL_CUDA_COMMON_HPP__

#define UTIL_CUDA_KERNEL_CHECK() UTIL_CUDA_CHECK(cudaGetLastError())

/**
Check CUDA error for synchronous call
cudaGetLastError is used to clear previous error happening at "condition".
*/
#define UTIL_CUDA_CHECK(condition)                                             \
  {                                                                            \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      cudaGetLastError();                                                      \
      printf("(%s) failed with \"%s\" (%s).",                                  \
             #condition, cudaGetErrorString(error),                            \
             cudaGetErrorName(error));                                         \
    }                                                                          \
  }

enum {
  CUDA_WARP_SIZE = 32,
  CUDA_WARP_MASK = 0x1f,
  CUDA_WARP_BITS = 5,
};

/** ceil(N/D) where N and D are integers */
#define UTIL_CEIL_INT_DIV(N, D)                                                \
  ((static_cast<int>(N) + static_cast<int>(D) - 1) / static_cast<int>(D))

/** Default num threads */
#define UTIL_CUDA_NUM_THREADS 256

/** Max number of blocks per dimension*/
//#define UTIL_CUDA_MAX_BLOCKS 65536
#define UTIL_CUDA_MAX_BLOCKS 1

/** Block size */
#define UTIL_CUDA_GET_BLOCKS(num) UTIL_CEIL_INT_DIV(num, UTIL_CUDA_NUM_THREADS)

/** Get an appropriate block size given a size of elements.

    The kernel is assumed to contain a grid-strided loop.
 */
inline int cuda_get_blocks_by_size(int size) {
  if (size == 0)
    return 0;
  const int blocks = UTIL_CUDA_GET_BLOCKS(size);
  const int inkernel_loop = UTIL_CEIL_INT_DIV(blocks, UTIL_CUDA_MAX_BLOCKS);
  const int total_blocks = UTIL_CEIL_INT_DIV(blocks, inkernel_loop);
  return total_blocks;
}

#define UTIL_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, ...)                      \
  {                                                                            \
    (kernel)<<<cuda_get_blocks_by_size(size), UTIL_CUDA_NUM_THREADS>>>(        \
        (size), __VA_ARGS__);                                                  \
    UTIL_CUDA_KERNEL_CHECK();                                                  \
  }

/** Cuda grid-strided loop */
#define UTIL_CUDA_KERNEL_LOOP(idx, num)                                        \
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (num);           \
       idx += blockDim.x * gridDim.x)

#endif
