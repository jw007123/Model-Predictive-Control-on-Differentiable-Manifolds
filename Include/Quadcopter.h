#pragma once

#include "Literals.h"
#include "NonLinearSolver.h"

struct QuadcopterState
{
	Eigen::Vector3d pI;
	Eigen::Vector3d vI;
	Eigen::Matrix3d R;

	QuadcopterState();
};

class Quadcopter : public NonLinearSolver<QuadcopterState, 9, 4, 10>
{
public:
	/// Inherit ctor from base class
	using NonLinearSolver<QuadcopterState, 9, 4, 10>::NonLinearSolver;

	/// Convenience
	typedef NonLinearSolver<QuadcopterState, 9, 4, 10>::CreateOptions CreateOptions;

	/// Virtuals: functions
	Eigen::Vector<f64, 9> f(const QuadcopterState& a_, const Control& b_)											const;
	QuadcopterState	      ApplyDelta(const QuadcopterState& a_, const Eigen::Vector<f64, 9>& delta_, const f64 dT_) const;

	/// Helper function for whether two states are approximately equal
	static bool IsApprox(const QuadcopterState& a_, const QuadcopterState& b_, const f64 eulTolSq_, const f64 rotTolSq_);

protected:
	/// Virtuals: constants
	Eigen::Matrix<f64, 9, 9> CalculateQZero()   const;
	Eigen::Matrix<f64, 4, 4> CalculatePZero()   const;
	Eigen::Matrix<f64, 4, 2> CalculateUBounds() const;

	/// Virtuals: functions
	Eigen::Vector<f64, 9>	 BoxMinus(const QuadcopterState& a_, const QuadcopterState& b_)		const;
	Eigen::Matrix<f64, 9, 9> dFdxk(const QuadcopterState& a_, const Control& b_, const f64 dT_)	const;
	Eigen::Matrix<f64, 9, 4> dFduk(const QuadcopterState& a_, const Control& b_, const f64 dT_)	const;

private:
	/// SO(3) Log function https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf
	static Eigen::Vector3d Log(const Eigen::Matrix3d& R_);

	/// SO(3) Exp function https://arwilliams.github.io/so3-exp.pdf
	static Eigen::Matrix3d Exp(const Eigen::Vector3d& w_);

	/// A as in (44) https://arxiv.org/pdf/2106.15233.pdf
	static Eigen::Matrix3d A(const Eigen::Vector3d& w_);
};
