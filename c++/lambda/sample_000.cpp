#include <stdio.h>
#include <string>

/*
	Hello World of Lambda
 */


// Functor object like this is created in the lambda function.
struct Plus
{
	int operator()(int a, int b) {
		return a + b;
	}
};


int main(int argc, char *argv[])
{
	// Lambda 
	auto plus = [](int a, int b) {return a + b;};

	int result = plus(10, 100);
	printf("%d\n", result);
	
	Plus plus_ftor;
	result = plus_ftor(10, 1000);
	printf("%d\n", result);

	return 0;
}




