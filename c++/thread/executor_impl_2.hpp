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

template<typename R>
ThreadPool<R, typename std::enable_if<std::is_base_of<Result, R>::value>::type* = nullptr>::ThreadPool(int pool_size): pool_size_(pool_size) {
//ThreadPool<R, enabler>::ThreadPool(int pool_size): pool_size_(pool_size) {
  // Create thread pool
  for (int i = 0; i < pool_size_; i++) {
    std::thread t([&, this] {
        while (true) {
          auto item = this->queue_.pop();
          R result = item.first();  // call function object.
          item.second->set_value(result);

          // stop
          bool cond = typeid(result) == typeid(StopResult);
          printf("result = %s\n", typeid(result).name());
          printf("stop_result = %s\n", typeid(StopResult).name());
          printf("cond = %d\n", cond);
          if (cond) {
            break;
          }
        }
      });
    threads_.push_back(std::move(t));
  }
}

template<typename R>
ThreadPool<R, typename std::enable_if<std::is_base_of<Result, R>::value>::type* = nullptr>::~ThreadPool() {
}

template<typename R>
std::shared_ptr<std::future<R>> ThreadPool<R, typename std::enable_if<std::is_base_of<Result, R>::value>::type* = nullptr>::submit(std::function<R()> func) {
  std::shared_ptr<std::promise<R>> p_ptr = std::make_shared<std::promise<R>>();
  std::future<R> f = p_ptr->get_future();
  auto f_ptr = std::make_shared<std::future<R>>(std::move(f));
  std::pair<std::function<R()>, std::shared_ptr<std::promise<R>>> item = std::make_pair(func, p_ptr);
  queue_.push(item);
  return f_ptr;
}

template<typename R>
void ThreadPool<R, typename std::enable_if<std::is_base_of<Result, R>::value>::type* = nullptr>::shutdown() {
  // Send stop messages
  for (int i = 0; i < pool_size_; i++) {
    std::function<R()> stop = []() {return StopResult();};
    this->submit(stop);
  }

  // Wait tasks completed
  for (auto &t : threads_) {
    t.join();
  }
}


