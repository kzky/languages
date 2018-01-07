#include <stdio.h>
#include <string>

/*
	Hello World of Capture of Lambda
 */


int main(int argc, char *argv[])
{
	int x = 10;
	printf("x = %d\n", x);

	// capture by reference
	auto f = [&]() {x = 100;};
	f();
	printf("x = %d (x was changed by reference capture)\n", x);

	// capture by copy
	auto g = [=]() {
		// x = 10; // error: read-only variable
		int y = x + 1;
		printf("y = %d (x was used by copy capture)\n", y);
	};
	g();

	// capture by copy but with mutable modifier
	auto g1 = [=]() mutable {
		x++;
		printf("x = %d (x was changed by copy capture with mutable modifier)\n", x);
	};
	g1();

	// capture by reference and copy
	int a = 1;
	int b = 2;
	printf("a = %d\n", a);
	auto h = [&a, b](int c) {
		auto d = (a + b) * c;
		a = d;
	};
	h(5);
	printf("a = %d p(a was usd and changed by reference capture)\n", a);

	return 0;
}

