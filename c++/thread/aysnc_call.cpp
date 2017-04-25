#include <cstdio>
#include <iostream>
#include <future>
#include <vector>

int task(int task_id) {
  auto id = std::hash<std::thread::id>()(std::this_thread::get_id());
  std::cout << "thread_id=" << id << std::endl;
  printf("task_id=%d\n", task_id);

  int n = 100000000;
  for (int i=0; i < n; i++) {
    // wait for long loop
  }

  return id;
}

int main(int argc, char *argv[])
{

  int n = 10;
  std::vector<std::future<int> > futures;

  // call async function
  for (int i=0; i < n; i++) {
    auto f = std::async(std::launch::async, task, i);
    futures.push_back(std::move(f));
  }

  // get result
  for (auto &f : futures) {
    int i = f.get();
    printf("taks_id=%d finished\n", i);
  }

  return 0;
}


