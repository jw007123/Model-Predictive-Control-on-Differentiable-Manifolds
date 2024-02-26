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
	using NonLinearSolver<QuadcopterState, 9, 4, 20>::NonLinearSolver;

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

	/// State equation f in (33c) https://arxiv.org/pdf/2106.15233.pdf
	static Eigen::Vector<f64, 9> f(const QuadcopterState& a_, const Control& b_);
};
