#include "executor.hpp"
#include <cstdio>

class SumTask {

private:
	int n_;

public:
	SumTask(int n) {
		n_ = n;
	}
	
	int operator()() const {
		int sum = 0;
		for (int i = 0; i < n_; i++) {
			sum += i;
		}
		return sum;
	}
};


int main(int argc, char *argv[])
{
	// Tasks
	SumTask task(100000);

	// Executor
	int pool_size = 4;

	//TODO: template specification
	ThreadPool<SumTask, int> thread_pool(pool_size);

	std::future<int> f = thread_pool.submit(task);

	printf("%d\n", f.get());

	thread_pool.shutdown();
		 
	return 0;
}



