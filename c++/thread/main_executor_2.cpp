#include "executor_2.hpp"
#include <cstdio>
#include <memory>
#include <chrono>


int main(int argc, char *argv[])
{
	// Executor
	int pool_size = 1;

	ThreadPool<Result> thread_pool(pool_size);

  int x = 0;
	auto f = thread_pool.submit([&]() {  x += 1; return Result();});


  // join
	f->get();

  printf("x = %d\n", x);

	thread_pool.shutdown();
		 
	return 0;
}



