#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include "Literals.h"
#include "NonLinearSolver.h"

struct BicycleModelState
{
	Eigen::Vector2d p;
	f64				v;
	Eigen::Matrix2d R;

	BicycleModelState();
};

class BicycleModel : public NonLinearSolver<BicycleModelState, 4, 2, 10>
{
public:
	/// Inherit ctor from base class
	using NonLinearSolver<BicycleModelState, 4, 2, 10>::NonLinearSolver;

	/// Convenience
	typedef NonLinearSolver<BicycleModelState, 4, 2, 10>::CreateOptions CreateOptions;

	/// Virtuals: functions
	Eigen::Vector<f64, 4> f(const BicycleModelState& a_, const Control& b_)											  const;
	BicycleModelState     ApplyDelta(const BicycleModelState& a_, const Eigen::Vector<f64, 4>& delta_, const f64 dT_) const;

	/// Helper function for whether two states are approximately equal
	static bool IsApprox(const BicycleModelState& a_, const BicycleModelState& b_, const f64 eulTolSq_, const f64 rotTolSq_);

	/// Generic setter...
	void SetWheelRadius(const f64 wheelRadiusM_);

protected:
	/// Virtuals: constants
	Eigen::Matrix<f64, 4, 4> CalculateQZero()   const;
	Eigen::Matrix<f64, 2, 2> CalculatePZero()   const;
	Eigen::Matrix<f64, 2, 2> CalculateUBounds() const;

	/// Virtuals: functions
	Eigen::Vector<f64, 4>	 BoxMinus(const BicycleModelState& a_, const BicycleModelState& b_)	  const;
	Eigen::Matrix<f64, 4, 4> dFdxk(const BicycleModelState& a_, const Control& b_, const f64 dT_) const;
	Eigen::Matrix<f64, 4, 2> dFduk(const BicycleModelState& a_, const Control& b_, const f64 dT_) const;

private:
	f64 wheelRadiusM;

	/// SO(2) Log function (7) https://arxiv.org/pdf/2106.15233.pdf
	static f64 Log(const Eigen::Matrix2d& R_);

	/// SO(2) Exp function (7) https://arxiv.org/pdf/2106.15233.pdf
	static Eigen::Matrix2d Exp(const f64& w_);
};
