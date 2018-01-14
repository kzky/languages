#include "executor.hpp"

template<typename T, typename R>
BlockingQueue<T, R>::BlockingQueue() {
}

template<typename T, typename R>
BlockingQueue<T, R>::~BlockingQueue() {
}

template<typename T, typename R>
void BlockingQueue<T, R>::push(std::pair<T, std::promise<R>> &&item) {
	std::unique_lock<std::mutex> mlock(mutex_);
	queue_.push(item);
	mlock.unlock();
	cond_.notify_one();
}

template<typename T, typename R>
std::pair<T, std::promise<R>> BlockingQueue<T, R>::pop() {
	std::unique_lock<std::mutex> mlock(mutex_);
	cond_.wait(mlock, [=] { return !this->queue_.empty(); });
	auto item = queue_.front();
	queue_.pop();
	return item;
}


template<typename T, typename R>
ThreadPool<T, R>::ThreadPool(int pool_size): pool_size_(pool_size) {
	// Create thread pool
	for (int i = 0; i < pool_size_; i++) {
		std::thread t([&] {
				while (true) {
					auto item = this->queue_.pop();
					R result = item.first();
					item.second.set_value(result);
				}
			});
		thread_pool_.push_back(t);
	}
}

template<typename T, typename R>
ThreadPool<T, R>::~ThreadPool() {
}
	
template<typename T, typename R>
std::future<R> ThreadPool<T, R>::submit(T const &task) {
	if (is_shutdown_) {
		return nullptr; // TODO: handle in a better way.
	}
	std::promise<R> p;
	std::future<R> f = p.get_future();
	std::pair<R, T> item = std::make_pair(std::move(task), std::move(p));
	queue_.push(item);
	return f;
}

template<typename T, typename R>
void ThreadPool<T, R>::shutdown() {
	is_shutdown_ = true;
	// Wait taks completed
	for (auto t : thread_pool_) {
		t.join();
	}
}


