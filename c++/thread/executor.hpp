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

template<typename R>
class Task {
  std::string msg_;

public:
  Task() {};
  Task(std::string msg) {
    msg_ = msg; 
  }
  std::string get_msg() {
    return msg_;
  }
};

// ThreadPool Traits
template<typename R, typename enalbe = void>
class ThreadPool;

template<template<typename> typename T, typename R>
class ThreadPool<T<R>, typename std::enable_if<std::is_base_of<Task<R>, T<R>>::value>::type> {
private:
  int pool_size_;
  BlockingQueue<std::pair<T<R>, std::shared_ptr<std::promise<R>>>> queue_;
  std::vector<std::thread> threads_;
  bool is_shutdown_;
public:
  ThreadPool(int pool_size);
  ~ThreadPool();
  std::shared_ptr<std::future<R>> submit(T<R> task);
  void shutdown();
};

#include "executor_impl.hpp"
