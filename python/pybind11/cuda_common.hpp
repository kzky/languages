#ifndef __NBLA_CUDA_COMMON_HPP__
#define __NBLA_CUDA_COMMON_HPP__

#define NBLA_CUDA_KERNEL_CHECK() NBLA_CUDA_CHECK(cudaGetLastError())

/**
Check CUDA error for synchronous call
cudaGetLastError is used to clear previous error happening at "condition".
*/
#define NBLA_CUDA_CHECK(condition)                                             \
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
#define NBLA_CEIL_INT_DIV(N, D)                                                \
  ((static_cast<int>(N) + static_cast<int>(D) - 1) / static_cast<int>(D))

/** Default num threads */
#define NBLA_CUDA_NUM_THREADS 512

/** Max number of blocks per dimension*/
#define NBLA_CUDA_MAX_BLOCKS 65536

/** Block size */
#define NBLA_CUDA_GET_BLOCKS(num) NBLA_CEIL_INT_DIV(num, NBLA_CUDA_NUM_THREADS)

/** Get an appropriate block size given a size of elements.

    The kernel is assumed to contain a grid-strided loop.
 */
inline int cuda_get_blocks_by_size(int size) {
  if (size == 0)
    return 0;
  const int blocks = NBLA_CUDA_GET_BLOCKS(size);
  const int inkernel_loop = NBLA_CEIL_INT_DIV(blocks, NBLA_CUDA_MAX_BLOCKS);
  const int total_blocks = NBLA_CEIL_INT_DIV(blocks, inkernel_loop);
  return total_blocks;
}

#define NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, ...)                      \
  {                                                                            \
    (kernel)<<<cuda_get_blocks_by_size(size), NBLA_CUDA_NUM_THREADS>>>(        \
        (size), __VA_ARGS__);                                                  \
    NBLA_CUDA_KERNEL_CHECK();                                                  \
  }

/** Cuda grid-strided loop */
#define NBLA_CUDA_KERNEL_LOOP(idx, num)                                        \
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (num);           \
       idx += blockDim.x * gridDim.x)

#endif
