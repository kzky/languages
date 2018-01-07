#include <cstdio>
#include <memory>
#include <thread>


int main(int argc, char *argv[])
{
	int x = 10;

	auto t = std::make_shared<std::thread>([&]{ x++; });
	t->join();

	// std::thread t([&]{ x++; });
	// t.join();

	printf("%d\n", x);
	

  return 0;
}


