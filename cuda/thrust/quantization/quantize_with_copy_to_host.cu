#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <time.h>
#include <cstdlib>

int main(int argc, char *argv[])
{
  // Settings
  const int N = atoi(argv[1]);
  const float p = 0.07;
  int threshold = (int)(N * p);
  printf("N: %d\n", N);

  // Container
  thrust::device_vector<float> data(N);
  thrust::sequence(data.begin(), data.end());
  thrust::device_vector<int> indices(N);
  // this can be determined in this code only.
  // in practice, memory pool is usually used.
  // and get the data of `size` from the begining of pool.
  thrust::host_vector<int> values(threshold); 

  // Find out the indices
  clock_t start0 = clock();
  thrust::device_vector<int>::iterator end = thrust::copy_if(thrust::make_counting_iterator(0),
                                                             thrust::make_counting_iterator(N),
                                                             data.begin(),
                                                             indices.begin(),
                                                             thrust::placeholders::_1 < threshold);
  int size = end - indices.begin();
  printf("Size: %d\n", size);
  indices.resize(size);
  
  // Fetch corresponding values
  thrust::copy(thrust::make_permutation_iterator(data.begin(), indices.begin()),
               thrust::make_permutation_iterator(data.end(), indices.end()),
               values.begin());
  clock_t stop0 = clock();
  double elapsed = (double)(stop0 - start0) * 1000.0 / CLOCKS_PER_SEC;
  printf("ElapsedTime: %.3f [ms]\n", elapsed);
  
  //std::cout << "indices: ";
  //thrust::copy(indices.begin(), indices.end(), std::ostream_iterator<int>(std::cout, " "));
  //std::cout << std::endl;

  //std::cout << "values: ";
  //thrust::copy(values.begin(), values.end(), std::ostream_iterator<int>(std::cout, " "));
  //std::cout << std::endl;

  return 0;
}
