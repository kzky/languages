#include <mutex>
#include <condition_variable>
#include <queue>


template<typename T>
class BlockingQueue {
private:
	std::mutex mutex_;
	std::condition_variable cond_;
	std::queue<T> queue_;

public:
	BlockingQueue();
	~BlockingQueue();
	void push(T const &task);
	T pop();
};
