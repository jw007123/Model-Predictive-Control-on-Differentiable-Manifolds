#ifdef _WIN32
	#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include <thread>

#include "Quadcopter.cpp"
#include "BicycleModel.cpp"
#include "SolverTests.cpp"

int main()
{
	printf("Running solver unit tests...\n\n");

	SolverTests solverTests;
	if (!solverTests.Run())
	{
		std::this_thread::sleep_for(std::chrono::seconds(5));
		return -1;
	}

	std::this_thread::sleep_for(std::chrono::seconds(5));
	return 0;
}
