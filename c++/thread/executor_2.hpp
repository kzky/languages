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

class Result {
};

class StopResult : public Result {
};

extern void* enabler;

//template<typename R>
template<typename R, typename std::enable_if<std::is_base_of<Result, R>::value>::type*>
class ThreadPool {
private:
  int pool_size_;
  BlockingQueue<std::pair<std::function<R()>, std::shared_ptr<std::promise<R>>>> queue_;
  std::vector<std::thread> threads_;
  bool is_shutdown_;
public:
  ThreadPool(int pool_size);
  ~ThreadPool();
  std::shared_ptr<std::future<R>> submit(std::function<R()> func);
  void shutdown();
};

#include "executor_impl_2.hpp"
