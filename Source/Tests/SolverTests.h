#pragma once

#include "Quadcopter.h"
#include "BicycleModel.h"

class SolverTests
{
public:
	bool Run();

private:
	enum Test : u8
	{
		UnconstrainedBicycleModel = 0,
		ConstrainedBicycleModel	  = 1,
		UnconstrainedQuadcopter   = 2,
		ConstrainedQuadcopter     = 3,
		Num
	};

	const char* TestStrings[Test::Num] =
	{
		"Unconstrained Bicycle Model",
		"Constrained Bicycle Model",
		"Unconstrained Quadcopter",
		"Constrained Quadcopter"
	};

	bool TestUnconstrainedBicycleModel();
	bool TestConstrainedBicycleModel();
	bool TestUnconstrainedQuadcopter();
	bool TestConstrainedQuadcopter();
};
