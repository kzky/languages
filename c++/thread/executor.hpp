#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <thread>


template<typename T>
class BlockingQueue {
private:
	std::mutex mutex_;
	std::condition_variable cond_;
	std::queue<T> queue_;

public:
	BlockingQueue();
	~BlockingQueue();
	void push(T const &task);
	T pop();
};

template<typename T>
class ExecutorPool {
private:
	BlockingQueue<T> queue_;
	int pool_size_;
public:
	ExecutorPool(int pool_size);
	~ExecutorPool();
	std::future<T> submit(T task);
	void shutdown();
};
