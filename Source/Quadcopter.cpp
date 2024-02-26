#include "Quadcopter.h"

QuadcopterState::QuadcopterState()
{
	pI.setZero();
	vI.setZero();
	R.setIdentity();
}

Eigen::Matrix<f64, 9, 9> Quadcopter::CalculateQZero() const
{
	return Eigen::Matrix<f64, 9, 9>::Identity();
}

Eigen::Matrix<f64, 4, 4> Quadcopter::CalculatePZero() const
{
	return (Eigen::Matrix<f64, 4, 4>::Identity() * 5.0);
}

Eigen::Matrix<f64, 4, 2> Quadcopter::CalculateUBounds() const
{
	Eigen::Matrix<f64, 4, 2> U;
	U.col(0) = Eigen::Vector<f64, 4>(-19.62, -1.0, -1.0, -1.0);
	U.col(1) = Eigen::Vector<f64, 4>( 19.62,  1.0,  1.0,  1.0);

	return U;
}

Eigen::Vector<f64, 9> Quadcopter::BoxMinus(const QuadcopterState& a_, const QuadcopterState& b_) const
{
	// For R^n, boxminus = -n
	// For SO(3), boxminus = Log(a^-1 * b)
	// Use manifold composition of boxminus to obtain result
	Eigen::Vector<f64, 9> aBoxMinusB;
	aBoxMinusB.tail<3>(0) = (b_.pI - a_.pI);
	aBoxMinusB.tail<3>(3) = (b_.vI - a_.vI);
	aBoxMinusB.tail<3>(6) = Log(a_.R.inverse() * b_.R);

	return aBoxMinusB;
}

Eigen::Matrix<f64, 9, 9> Quadcopter::dFdxk(const QuadcopterState& a_, const Control& b_, const f64 dT_) const
{

}

Eigen::Matrix<f64, 9, 4> Quadcopter::dFduk(const QuadcopterState& a_, const Control& b_, const f64 dT_) const
{
	// Determine dTf
	const Eigen::Vector<f64, 9> dTf = dT_ * f(a_, b_);

	// Determine Gf = Gf(dt * f) = I_n f[0, 2] BS I_n f[3, 5] BS (A (dt f[6, 8]))^T
	// where BS is the block sum. Hence, Gf is a 9x9 matrix

	// Determine first two blocks of Gf over R^n
	Eigen::Matrix3d Gf0(Eigen::Matrix3d::Identity());
	Gf0.diagonal() = dTf.tail<3>(0);
	Eigen::Matrix3d Gf1(Eigen::Matrix3d::Identity());
	Gf1.diagonal() = dTf.tail<3>(3);

	// Determine final block over SO(3) and use result in Table I
	Eigen::Matrix3d Gf2 = A(dTf.tail<3>(6));
	Gf2.transposeInPlace();

	// Compose as block sum with off-diagonal values as 0
	Eigen::Matrix<f64, 9, 9> Gf(Eigen::Matrix<f64, 9, 9>::Zero());
	Gf.block<3, 3>(0, 0) = Gf0;
	Gf.block<3, 3>(3, 3) = Gf1;
	Gf.block<3, 3>(6, 6) = Gf2;
		 
	// Determine (df/ddu)_{ddu = 0}
	Eigen::Matrix<f64, 9, 4> dfddu(Eigen::Matrix<f64, 9, 4>::Zero());
	dfddu.block<3, 1>(3, 0) = -(a_.R * Eigen::Vector3d::UnitZ());
	dfddu.block<3, 3>(6, 1) = Eigen::Matrix3d::Identity();

	// Fuk = dT * Gf * (df/ddu)_{ddu = 0}
	return dT_ * Gf * dfddu;
}

Eigen::Vector3d Quadcopter::Log(const Eigen::Matrix3d& R_)
{
	// Determine sin(t) and t
	const f64 traceR = R_.trace();
	const f64 sinT   = 0.5 * std::sqrt((3.0 - traceR) * (1.0 + traceR));
	const f64 t		 = std::asin(sinT);

	// Early return for case t ~= 0. Using f32 as small_num rather than actual eps
	if (std::abs(t) < std::numeric_limits<f32>::epsilon())
	{
		return Eigen::Vector3d::Zero();
	}

	// Base case
	const Eigen::Vector3d v(R_(3, 2) - R_(2, 3), R_(1, 3) - R_(3, 1), R_(2, 1) - R_(1, 2));
	return (t / (2.0 * sinT)) * v;
}

Eigen::Matrix3d Quadcopter::Exp(const Eigen::Vector3d& w_)
{
	// Convert w to skew(w)
	Eigen::Matrix3d skewW;
	skewW << 0.0,   -w_.z(), w_.y(),
			 w_.z(), 0.0,   -w_.x(),
			-w_.y(), w_.x(), 0.0;

	// Determine ||w|| and compute Exp coefficients via TS (thanks Wolfram)
	const f64 normW  = w_.norm();
	const f64 normW2 = normW * normW;
	const f64 normW4 = normW2 * normW2;
	const f64 e0     = 1.0 - (normW2 / 6.0) + (normW4 / 120.0);
	const f64 e1	 = 0.5 - (normW2 / 24.0) + (normW4 / 720.0);

	// Final eq
	const Eigen::Matrix3d skewW2 = skewW * skewW;
	return Eigen::Matrix3d::Identity() + e0 * skewW + e1 * skewW2;
}

Eigen::Matrix3d Quadcopter::A(const Eigen::Vector3d& w_)
{
	// Convert w to skew(w)
	Eigen::Matrix3d skewW;
	skewW << 0.0,   -w_.z(), w_.y(),
			 w_.z(), 0.0,   -w_.x(),
			-w_.y(), w_.x(), 0.0;

	// Determine ||w|| and compute A coefficients via TS
	const f64 normW  = w_.norm();
	const f64 normW2 = normW * normW;
	const f64 normW4 = normW2 * normW2;
	const f64 e0	 = (1.0 / 2.0) - (normW2 / 24.0) + (normW4 / 720.0);
	const f64 e1	 = (1.0 / 6.0) - (normW2 / 120.0) + (normW4 / 5040.0);

	// Final eq
	const Eigen::Matrix3d skewW2 = skewW * skewW;
	return Eigen::Matrix3d::Identity() + e0 * skewW + e1 * skewW2;
}

Eigen::Vector<f64, 9> Quadcopter::f(const QuadcopterState& a_, const Control& b_)
{
	Eigen::Vector<f64, 9> fk;
	fk.tail<3>(0) = a_.vI;
	fk.tail<3>(3) = Eigen::Vector3d(0.0, 0.0, -9.81) - (b_(0) * a_.R * Eigen::Vector3d::UnitZ());
	fk.tail<3>(6) = b_.tail<3>(1);

	return fk;
}
