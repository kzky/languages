/*
	Minimal Examples of ExecutorPool
 */

#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <thread>
#include <utility>
#include <memory>

template<typename T, typename R>
class BlockingQueue {
private:
	std::mutex mutex_;
	std::condition_variable cond_;
	std::queue<T> queue_;

public:
	BlockingQueue();
	~BlockingQueue();
	void push(std::pair<T, std::promise<R>> &&item);
	std::pair<T, std::promise<R>> pop();
};

template<typename T, typename R>
class ExecutorPool {
private:
	int pool_size_;
	BlockingQueue<T, R> queue_;
	std::vector<std::thread> thread_pool_;
	bool is_shutdown_;
public:
	ExecutorPool(int pool_size);
	~ExecutorPool();
	std::future<R> submit(T const &task);
	void shutdown();
};
