template<typename T>
BlockingQueue<T>::BlockingQueue() {
}

template<typename T>
BlockingQueue<T>::~BlockingQueue() {
}

template<typename T>
void BlockingQueue<T>::push(T &item) {
	std::unique_lock<std::mutex> mlock(mutex_);
	queue_.push(item);
	mlock.unlock();
	cond_.notify_one();
}

template<typename T>
T BlockingQueue<T>::pop() {
	std::unique_lock<std::mutex> mlock(mutex_);
	cond_.wait(mlock, [=] { return !this->queue_.empty(); });
	auto item = queue_.front();
	queue_.pop(); // delete promise here if item is reference.
	return item;
}


template<template<typename R> typename T, typename R>
ThreadPool<T<R>>::ThreadPool(int pool_size): pool_size_(pool_size) {
	// Create thread pool
	for (int i = 0; i < pool_size_; i++) {
		std::thread t([&, this] {
				while (true) {
					auto item = this->queue_.pop();
					R result = item.first();
					item.second->set_value(result);
				}
			});
		thread_pool_.push_back(std::move(t));
	}
}

template<template<typename R> typename T, typename R>
ThreadPool<T<R>>::~ThreadPool() {
}
	
template<template<typename R> typename T, typename R>
std::shared_ptr<std::future<R>> ThreadPool<T<R>>::submit(T<R> const &task) {
	//TODO: handle when shutdowning.
	std::shared_ptr<std::promise<R>> p_ptr = std::make_shared<std::promise<R>>();
	std::future<R> f = p_ptr->get_future();
	auto f_ptr = std::make_shared<std::future<R>>(std::move(f));
	std::pair<T<R>, std::shared_ptr<std::promise<R>>> item = std::make_pair(task, p_ptr);
	queue_.push(item);
	return f_ptr;
}

template<template<typename R> typename T, typename R>
void ThreadPool<T<R>>::shutdown() {
	//TODO: Send End Message
	
	// Wait tasks completed
	for (auto &t : thread_pool_) {
		t.join();
	}
}


