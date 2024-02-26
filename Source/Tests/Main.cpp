#ifdef _WIN32
	#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "Quadcopter.cpp"
#include "BicycleModel.cpp"

int main()
{
	NonLinearSolver<QuadcopterState, 9, 4, 10>::CreateOptions quadSolverOptions;
	quadSolverOptions = NonLinearSolver<QuadcopterState, 9, 4, 10>::CreateOptions::PrecalculateAll;

	Quadcopter quadSolver(quadSolverOptions);

	return 0;
}
