/*
	Minimal Examples of ThreadPool
 */

#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <thread>
#include <utility>
#include <memory>

//TODO: can address one type: T<R>?
template<typename T, typename R>
class BlockingQueue {
private:
	std::mutex mutex_;
	std::condition_variable cond_;
	std::queue<std::pair<T, std::shared_ptr<std::promise<R>>>> queue_;

public:
	BlockingQueue();
	~BlockingQueue();
	void push(std::pair<T, std::shared_ptr<std::promise<R>>> &item);
	std::pair<T, std::shared_ptr<std::promise<R>>> pop();
};

template<typename T, typename R>
class ThreadPool {
private:
	int pool_size_;
	BlockingQueue<T, R> queue_;
	std::vector<std::thread> thread_pool_;
	bool is_shutdown_;
public:
	ThreadPool(int pool_size);
	~ThreadPool();
	std::shared_ptr<std::future<R>> submit(T const &task);
	void shutdown();
};

#include "executor_impl.hpp"
