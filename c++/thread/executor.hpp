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

template<typename T>
class BlockingQueue {
private:
	std::mutex mutex_;
	std::condition_variable cond_;
	std::queue<T> queue_;

public:
	BlockingQueue();
	~BlockingQueue();
	void push(T &item);
	T pop();
};


// ThreadPool Traits
template<typename R>
class ThreadPool;

template<template<typename> typename T, typename R>
class ThreadPool<T<R>> {
private:
	int pool_size_;
	BlockingQueue<std::pair<T<R>, std::shared_ptr<std::promise<R>>>> queue_;
	std::vector<std::thread> thread_pool_;
	bool is_shutdown_;
public:
	ThreadPool(int pool_size);
	~ThreadPool();
	std::shared_ptr<std::future<R>> submit(T<R> const &task);
	void shutdown();
};

#include "executor_impl.hpp"
