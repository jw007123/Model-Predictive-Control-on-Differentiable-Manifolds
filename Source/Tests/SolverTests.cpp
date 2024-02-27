#include "SolverTests.h"

bool SolverTests::Run()
{
	using namespace std::chrono;

	std::function<bool()> TestFuncs[Test::Num] =
	{
		std::bind(&SolverTests::TestUnconstrainedBicycleModel, this),
		std::bind(&SolverTests::TestConstrainedBicycleModel,   this),
		std::bind(&SolverTests::TestUnconstrainedQuadcopter,   this),
		std::bind(&SolverTests::TestConstrainedQuadcopter,     this)
	};

	const time_point<system_clock> allStartPt = system_clock::now();
	u32 testsPassed							  = 0;

	for (u32 i = 0; i < Test::Num; ++i)
	{
		const time_point<system_clock> testStartPt = system_clock::now();
		const bool testPassed					   = TestFuncs[i]();
		const time_point<system_clock> testEndPt   = system_clock::now();
		const milliseconds testTime				   = duration_cast<milliseconds>(testEndPt - testStartPt);

		if (testPassed)
		{
			printf("%s: Passed in %zu[ms]\n\n", TestStrings[i], testTime.count());
		}
		else
		{
			printf("%s: Failed in %zu[ms]\n\n", TestStrings[i], testTime.count());
		}

		testsPassed += testPassed;
	}

	const time_point<system_clock> allEndPt = system_clock::now();
	const milliseconds allTime				= duration_cast<milliseconds>(allEndPt - allStartPt);

	if (testsPassed == Test::Num)
	{
		printf("All tests passed in %zu[ms]!\n\n", allTime.count());
	}
	else
	{
		printf("Some tests failed in %zu[ms]!\n\n", allTime.count());
	}

	return (testsPassed == Test::Num);
}

bool SolverTests::TestUnconstrainedBicycleModel()
{
	// Constants
	constexpr const f64 wheelRadius = 2.0;
	constexpr const f64 dT			= 0.1;
	constexpr const u32 lookaheadN  = 25;
	constexpr const u32 sequenceN   = 400;

	// Create solver with precalculated flags and set the wheel radius
	BicycleModel::CreateOptions biSolverOptions;
	biSolverOptions = (BicycleModel::CreateOptions)((u8)BicycleModel::CreateOptions::PrecalculateAll | (u8)BicycleModel::CreateOptions::IgnoreUBounds);
	BicycleModel biSolver(biSolverOptions);
	biSolver.SetWheelRadius(wheelRadius);

	// Allocate storage for reference arrays. rStates[0] is the origin due to ctor()
	std::vector<BicycleModelState> referenceStates;
	referenceStates.resize(sequenceN);
	std::vector<Eigen::Vector2d> referenceControls;
	referenceControls.resize(sequenceN);

	// Generate a reference sequence of states and controls using the BicycleModel dynamics
	{
		u32 cnt = 0;
		for (u32 i = 0; i < (sequenceN - 1); ++i)
		{
			// Accelerate and deccelerate
			referenceControls[i](0) = cnt >= 10 ? -0.1 : 0.1;

			// Keep a constant turning angle
			referenceControls[i](1) = cnt >= 10 ? -0.25 : 0.25;

			// Determine new state
			const Eigen::Vector4d fxkuk = biSolver.f(referenceStates[i], referenceControls[i]);
			referenceStates[i + 1]		= biSolver.ApplyDelta(referenceStates[i], fxkuk, dT);

			// Update tracker var
			cnt = (cnt == 19) ? 0 : (cnt + 1);
		}
	}
	
	// Determine that, if we start at each reference state, we receive the reference control back
	BicycleModelState currentState = referenceStates[0];
	std::array<Eigen::Vector2d, lookaheadN> optimalControls;
	for (u32 i = 0; i < ((sequenceN - 1) - lookaheadN); ++i)
	{
		// Obtain optimal set for this window
		if (!biSolver.Solve(optimalControls.data(),
						    currentState,
							referenceStates.data() + i,
							referenceControls.data() + i,
							dT))
		{
			return false;
		}

		// Apply control to state
		const Eigen::Vector4d fxkuk = biSolver.f(currentState, optimalControls[0]);
		currentState				= biSolver.ApplyDelta(currentState, fxkuk, dT);

		// If we never deviate (~0.5m-ish) from the path, then our MPC is working as expected because
		// we start from the same position
		if (!BicycleModel::IsApprox(currentState, referenceStates[i + 1], (0.25 * 0.25), (0.1 * 0.1)))
		{
			return false;
		}
	}

	return true;
}

bool SolverTests::TestConstrainedBicycleModel()
{
	// Constants
	constexpr const f64 wheelRadius = 2.0;
	constexpr const f64 dT			= 0.1;
	constexpr const u32 lookaheadN  = 25;
	constexpr const u32 sequenceN   = 400;

	// Create solver with precalculated flags and set the wheel radius
	BicycleModel::CreateOptions biSolverOptions;
	biSolverOptions = BicycleModel::CreateOptions::PrecalculateAll;
	BicycleModel biSolver(biSolverOptions);
	biSolver.SetWheelRadius(wheelRadius);

	// Allocate storage for reference arrays. rStates[0] is the origin due to ctor()
	std::vector<BicycleModelState> referenceStates;
	referenceStates.resize(sequenceN);
	std::vector<Eigen::Vector2d> referenceControls;
	referenceControls.resize(sequenceN);

	// Generate a reference sequence of states and controls using the BicycleModel dynamics
	{
		u32 cnt = 0;
		for (u32 i = 0; i < (sequenceN - 1); ++i)
		{
			// Accelerate and deccelerate
			referenceControls[i](0) = cnt >= 10 ? -0.1 : 0.1;

			// Keep a constant turning angle
			referenceControls[i](1) = cnt >= 10 ? -0.25 : 0.25;

			// Determine new state
			const Eigen::Vector4d fxkuk = biSolver.f(referenceStates[i], referenceControls[i]);
			referenceStates[i + 1]		= biSolver.ApplyDelta(referenceStates[i], fxkuk, dT);

			// Update tracker var
			cnt = (cnt == 19) ? 0 : (cnt + 1);
		}
	}
	
	// Determine that, if we start at each reference state, we receive the reference control back
	BicycleModelState currentState = referenceStates[0];
	std::array<Eigen::Vector2d, lookaheadN> optimalControls;
	for (u32 i = 0; i < ((sequenceN - 1) - lookaheadN); ++i)
	{
		// Obtain optimal set for this window
		if (!biSolver.Solve(optimalControls.data(),
						    currentState,
							referenceStates.data() + i,
							referenceControls.data() + i,
							dT))
		{
			return false;
		}

		// Apply control to state
		const Eigen::Vector4d fxkuk = biSolver.f(currentState, optimalControls[0]);
		currentState				= biSolver.ApplyDelta(currentState, fxkuk, dT);

		// If we never deviate (~0.5m-ish) from the path, then our MPC is working as expected because
		// we start from the same position
		if (!BicycleModel::IsApprox(currentState, referenceStates[i + 1], (0.25 * 0.25), (0.1 * 0.1)))
		{
			return false;
		}
	}

	return true;
}

bool SolverTests::TestUnconstrainedQuadcopter()
{
	// Constants
	constexpr const f64 wheelRadius = 2.0;
	constexpr const f64 dT			= 0.1;
	constexpr const u32 lookaheadN  = 25;
	constexpr const u32 sequenceN   = 400;

	// Create solver with precalculated flags and set the wheel radius
	Quadcopter::CreateOptions quadSolverOptions;
	quadSolverOptions = (Quadcopter::CreateOptions)((u8)Quadcopter::CreateOptions::PrecalculateAll | (u8)Quadcopter::CreateOptions::IgnoreUBounds);
	Quadcopter quadSolver(quadSolverOptions);

	// Allocate storage for reference arrays. rStates[0] is the origin due to ctor()
	std::vector<QuadcopterState> referenceStates;
	referenceStates.resize(sequenceN);
	std::vector<Eigen::Vector4d> referenceControls;
	referenceControls.resize(sequenceN);

	// Generate a reference sequence of states and controls using the BicycleModel dynamics
	{
		u32 cnt = 0;
		for (u32 i = 0; i < (sequenceN - 1); ++i)
		{
			// Accelerate and deccelerate
			referenceControls[i](0) = cnt >= 10 ? -0.1 : 0.1;

			// Do loops because... Why not?
			referenceControls[i](1) = 0.1;

			// No roll
			referenceControls[i](2) = 0.0;

			// Keep a constant turning angle
			referenceControls[i](3) = cnt >= 10 ? -0.25 : 0.25;

			// Determine new state
			const Eigen::Vector<f64, 9> fxkuk = quadSolver.f(referenceStates[i], referenceControls[i]);
			referenceStates[i + 1]	     	  = quadSolver.ApplyDelta(referenceStates[i], fxkuk, dT);

			// Update tracker var
			cnt = (cnt == 19) ? 0 : (cnt + 1);
		}
	}
	
	// Determine that, if we start at each reference state, we receive the reference control back
	QuadcopterState currentState = referenceStates[0];
	std::array<Eigen::Vector4d, lookaheadN> optimalControls;
	for (u32 i = 0; i < ((sequenceN - 1) - lookaheadN); ++i)
	{
		// Obtain optimal set for this window
		if (!quadSolver.Solve(optimalControls.data(),
						      currentState,
							  referenceStates.data() + i,
							  referenceControls.data() + i,
							  dT))
		{
			return false;
		}

		// Apply control to state
		const Eigen::Vector<f64, 9> fxkuk = quadSolver.f(currentState, optimalControls[0]);
		currentState					  = quadSolver.ApplyDelta(currentState, fxkuk, dT);

		// If we never deviate (~0.5m-ish) from the path, then our MPC is working as expected because
		// we start from the same position
		if (!Quadcopter::IsApprox(currentState, referenceStates[i + 1], (0.25 * 0.25), (0.1 * 0.1)))
		{
			return false;
		}
	}

	return true;
}

bool SolverTests::TestConstrainedQuadcopter()
{
	// Constants
	constexpr const f64 wheelRadius = 2.0;
	constexpr const f64 dT			= 0.1;
	constexpr const u32 lookaheadN  = 25;
	constexpr const u32 sequenceN   = 400;

	// Create solver with precalculated flags and set the wheel radius
	Quadcopter::CreateOptions quadSolverOptions;
	quadSolverOptions = Quadcopter::CreateOptions::PrecalculateAll;
	Quadcopter quadSolver(quadSolverOptions);

	// Allocate storage for reference arrays. rStates[0] is the origin due to ctor()
	std::vector<QuadcopterState> referenceStates;
	referenceStates.resize(sequenceN);
	std::vector<Eigen::Vector4d> referenceControls;
	referenceControls.resize(sequenceN);

	// Generate a reference sequence of states and controls using the BicycleModel dynamics
	{
		u32 cnt = 0;
		for (u32 i = 0; i < (sequenceN - 1); ++i)
		{
			// Accelerate and deccelerate
			referenceControls[i](0) = cnt >= 10 ? -0.1 : 0.1;

			// Do loops because... Why not?
			referenceControls[i](1) = 0.1;

			// No roll
			referenceControls[i](2) = 0.0;

			// Keep a constant turning angle
			referenceControls[i](3) = cnt >= 10 ? -0.25 : 0.25;

			// Determine new state
			const Eigen::Vector<f64, 9> fxkuk = quadSolver.f(referenceStates[i], referenceControls[i]);
			referenceStates[i + 1]	     	  = quadSolver.ApplyDelta(referenceStates[i], fxkuk, dT);

			// Update tracker var
			cnt = (cnt == 19) ? 0 : (cnt + 1);
		}
	}
	
	// Determine that, if we start at each reference state, we receive the reference control back
	QuadcopterState currentState = referenceStates[0];
	std::array<Eigen::Vector4d, lookaheadN> optimalControls;
	for (u32 i = 0; i < ((sequenceN - 1) - lookaheadN); ++i)
	{
		// Obtain optimal set for this window
		if (!quadSolver.Solve(optimalControls.data(),
						      currentState,
							  referenceStates.data() + i,
							  referenceControls.data() + i,
							  dT))
		{
			return false;
		}

		// Apply control to state
		const Eigen::Vector<f64, 9> fxkuk = quadSolver.f(currentState, optimalControls[0]);
		currentState					  = quadSolver.ApplyDelta(currentState, fxkuk, dT);

		// If we never deviate (~0.5m-ish) from the path, then our MPC is working as expected because
		// we start from the same position
		if (!Quadcopter::IsApprox(currentState, referenceStates[i + 1], (0.25 * 0.25), (0.1 * 0.1)))
		{
			return false;
		}
	}

	return true;
}
