#include <iostream>
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <time.h>
#include <curand.h>
#include <cstdlib>

/*Need $ nvcc -lcurand sort.cu -o sort.out*/


void GPU_fill_rand(float *data, int N) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, data, N);
}

void thrust_sort(int n, int randomseed)
{
	std::cout << "Start thrust_sort\n";

	// Create random vector
	thrust::device_vector<float> d_vector(n);
	clock_t start0 = clock();
	GPU_fill_rand(thrust::raw_pointer_cast(&d_vector[0]), n);
	cudaDeviceSynchronize();
	clock_t stop0 = clock();
	double elapsed = (double)(stop0 - start0) * 1000.0 / CLOCKS_PER_SEC;
	printf("ElapsedTime(Generate random number): %.3f [ms]\n", elapsed);

	// Sort
	start0 = clock();
	thrust::sort(thrust::device, d_vector.begin(), d_vector.end());
	cudaDeviceSynchronize();
	stop0 = clock();
	elapsed = (double)(stop0 - start0) * 1000.0 / CLOCKS_PER_SEC;
	printf("ElapsedTime(Sort): %.3f [ms]\n", elapsed);

	std::cout << "\nDone.\n";
}


int main(int argc, char *argv[]) {
	int n = atoi(argv[1]);
	int randomseed = time(NULL);

	thrust_sort(n, randomseed);

	return 0;
}
