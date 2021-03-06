#include "executor.hpp"
#include <cstdio>
#include <memory>
#include <chrono>

template<typename R>
class SumTask : public Task<R> {

private:
	int n_;

public:
	SumTask(R n) {
		n_ = n;
	}

  using Task<R>::Task;
	
	R operator()() const {
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
	SumTask<int> task(100);

	// Executor
	int pool_size = 4;

	ThreadPool<SumTask<int>> thread_pool(pool_size);

	std::shared_ptr<std::future<int>> f = thread_pool.submit(task);

	std::this_thread::sleep_for(std::chrono::seconds(1));

	printf("%d\n", f->get());

	thread_pool.shutdown();
		 
	return 0;
}



