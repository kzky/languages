#include <executor.hpp>

template<typename T>
BlockingQueue<T>::BlockingQueue() {
}

template<typename T>
BlockingQueue<T>::~BlockingQueue() {
}

template<typename T>
void BlockingQueue<T>::push(T const &task) {
	std::unique_lock<std::mutex> mlock(mutex_);
	queue_.push(task);
	mlock.unlock();
	cond_.notify_one();
}

template<typename T>
T BlockingQueue<T>::pop() {
	std::unique_lock<std::mutex> mlock(mutex_);
	cond_.wait(mlock, [=] { return !this->queue_.empty(); });
	auto item = queue_.front();
	queue_.pop();
}
