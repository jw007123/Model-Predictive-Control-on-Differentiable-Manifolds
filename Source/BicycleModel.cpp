#include "BicycleModel.h"

BicycleModelState::BicycleModelState()
{
	p.setZero();
	v = 0.0;
	R.setIdentity();
}

void BicycleModel::SetWheelRadius(const f64 wheelRadiusM_)
{
	wheelRadiusM = wheelRadiusM_;
}

Eigen::Matrix<f64, 4, 4> BicycleModel::CalculateQZero() const
{
	return Eigen::Matrix<f64, 4, 4>::Identity();
}

Eigen::Matrix<f64, 2, 2> BicycleModel::CalculatePZero() const
{
	return (Eigen::Matrix<f64, 2, 2>::Identity() * 5.0);
}

Eigen::Matrix<f64, 2, 2> BicycleModel::CalculateUBounds() const
{
	constexpr const f64 pi = 3.14159265358979323846;

	Eigen::Matrix<f64, 2, 2> U;
	U.col(0) = Eigen::Vector<f64, 2>(-3.0, -pi * 0.25);
	U.col(1) = Eigen::Vector<f64, 4>( 3.0,  pi * 0.25);

	return U;
}

Eigen::Vector<f64, 4> BicycleModel::BoxMinus(const BicycleModelState& a_, const BicycleModelState& b_) const
{
	// For R^n, boxminus = -n
	// For SO(2), boxminus = Log(a^-1 * b)
	// Use manifold composition of boxminus to obtain result
	Eigen::Vector<f64, 4> aBoxMinusB;
	aBoxMinusB.head<2>(0) = b_.p - a_.p;
	aBoxMinusB(2)		  = b_.v - a_.v;
	aBoxMinusB(3)		  = Log(a_.R.inverse() * b_.R);

	return aBoxMinusB;
}

Eigen::Matrix<f64, 4, 4> BicycleModel::dFdxk(const BicycleModelState& a_, const Control& b_, const f64 dT_) const
{
	// Determine dTf
	const Eigen::Vector<f64, 4> dTf = dT_ * f(a_, b_);

	// Determine Gx = I_n BS 1 BS 1, where BS is the block sum. I.e. the identity
	const Eigen::Matrix<f64, 4, 4> Gx(Eigen::Matrix<f64, 4, 4>::Identity());

	// Likewise for Gf. NOTE(IF): These could be removed, but are kept in for readability sake
	const Eigen::Matrix<f64, 4, 4> Gf(Eigen::Matrix<f64, 4, 4>::Identity());

	Eigen::Matrix<f64, 4, 4> dfddx;
	/*
	*
	*		CALCULATE
	* 
	*/

	// Fxk = Gx + dT * Gf * (df/ddx)_{d = 0}
	return (Gx + (dT_ * Gf * dfddx));
}

Eigen::Matrix<f64, 4, 2> BicycleModel::dFduk(const BicycleModelState& a_, const Control& b_, const f64 dT_) const
{
	// Determine dTf
	const Eigen::Vector<f64, 4> dTf = dT_ * f(a_, b_);

	// Determine Gf = I_n BS 1 BS 1, where BS is the block sum. I.e. the identity
	const Eigen::Matrix<f64, 4, 4> Gf(Eigen::Matrix<f64, 4, 4>::Identity());

	Eigen::Matrix<f64, 4, 4> dfddu;
	/*
	*
	*		CALCULATE
	*
	*/

	// Fuk = dT * Gf * (df/ddu)_{d = 0}
	return (dT_ * Gf * dfddu);
}

f64 BicycleModel::Log(const Eigen::Matrix2d& R_)
{
	return std::atan2(R_(2, 1), R_(1, 1));
}

Eigen::Matrix2d BicycleModel::Exp(const f64& w_)
{
	const f64 cw = std::cos(w_);
	const f64 sw = std::sin(w_);

	Eigen::Matrix2d R;
	R << cw, -sw,
		 sw,  cw;

	return R;
}

Eigen::Vector<f64, 4> BicycleModel::f(const BicycleModelState& a_, const Control& b_) const
{
	Eigen::Vector<f64, 4> fk;
	fk(0) = a_.v * std::cos(b_(1));
	fk(1) = a_.v * std::sin(b_(1));
	fk(2) = b_(0);
	fk(3) = (a_.v / wheelRadiusM) * std::tan(b_(1));

	return fk;
}
