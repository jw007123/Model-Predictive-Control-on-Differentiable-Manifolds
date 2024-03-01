#pragma once

#include "Literals.h"
#include "NonLinearSolver.h"

namespace ModelPredictiveControl
{
	struct QuadcopterState
	{
		Eigen::Vector3d pI;
		Eigen::Vector3d vI;
		Eigen::Matrix3d R;

		QuadcopterState();
	};

	template <u32 L>
	class Quadcopter : public NonLinearSolver<QuadcopterState, 9, 4, L>
	{
	public:
		/// Inherit ctor from base class
		using NonLinearSolver<QuadcopterState, 9, 4, L>::NonLinearSolver;

		/// Convenience
		using CreateOptions = typename NonLinearSolver<QuadcopterState, 9, 4, L>::CreateOptions;
		using Control	    = typename NonLinearSolver<QuadcopterState, 9, 4, L>::Control;

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

	template <u32 L>
	Eigen::Vector<f64, 9> Quadcopter<L>::f(const QuadcopterState& a_, const Control& b_) const
	{
		constexpr const f64 g = 9.80665;

		// State equation f in (33c) https://arxiv.org/pdf/2106.15233.pdf
		Eigen::Vector<f64, 9> fk;
		fk.segment<3>(0) = a_.vI;
		fk.segment<3>(3) = Eigen::Vector3d(0.0, 0.0, -g) - (b_(0) * a_.R * Eigen::Vector3d::UnitZ());
		fk.segment<3>(6) = b_.segment<3>(1);

		return fk;
	}

	template <u32 L>
	QuadcopterState	Quadcopter<L>::ApplyDelta(const QuadcopterState& a_, const Eigen::Vector<f64, 9>& delta_, const f64 dT_) const
	{
		// For Euclidean components of a_, OP := +_n. For SO(3) R comp, OP := x * Exp(R)
		QuadcopterState b = a_;
		b.pI += (dT_ * delta_.segment<3>(0));
		b.vI += (dT_ * delta_.segment<3>(3));
		b.R  *= (dT_ * Exp(delta_.segment<3>(6)));

		return b;
	}

	template <u32 L>
	bool Quadcopter<L>::IsApprox(const QuadcopterState& a_, const QuadcopterState& b_, const f64 eulTolSq_, const f64 rotTolSq_)
	{
		// Use correct L2 norm for Euclidean spaces and SO(3) space
		if ((a_.pI - b_.pI).squaredNorm() > eulTolSq_)
		{
			return false;
		}
		if ((a_.vI - b_.vI).squaredNorm() > eulTolSq_)
		{
			return false;
		}

		const Eigen::Vector3d Lab = Log(a_.R.inverse() * b_.R);
		if (Lab.squaredNorm() > rotTolSq_)
		{
			return false;
		}

		return true;
	}

	template <u32 L>
	Eigen::Matrix<f64, 9, 9> Quadcopter<L>::CalculateQZero() const
	{
		return Eigen::Matrix<f64, 9, 9>::Identity();
	}

	template <u32 L>
	Eigen::Matrix<f64, 4, 4> Quadcopter<L>::CalculatePZero() const
	{
		return (Eigen::Matrix<f64, 4, 4>::Identity() * 5.0);
	}

	template <u32 L>
	Eigen::Matrix<f64, 4, 2> Quadcopter<L>::CalculateUBounds() const
	{
		constexpr const f64 g = 9.80665;

		Eigen::Matrix<f64, 4, 2> U;
		U.col(0) = Eigen::Vector<f64, 4>(-2.0 * g, -1.0, -1.0, -1.0);
		U.col(1) = Eigen::Vector<f64, 4>(2.0 * g, 1.0, 1.0, 1.0);

		return U;
	}

	template <u32 L>
	Eigen::Vector<f64, 9> Quadcopter<L>::BoxMinus(const QuadcopterState& a_, const QuadcopterState& b_) const
	{
		// For R^n, boxminus = -n
		// For SO(3), boxminus = Log(a^-1 * b)
		// Use manifold composition of boxminus to obtain result
		Eigen::Vector<f64, 9> aBoxMinusB;
		aBoxMinusB.segment<3>(0) = (b_.pI - a_.pI);
		aBoxMinusB.segment<3>(3) = (b_.vI - a_.vI);
		aBoxMinusB.segment<3>(6) = Log(a_.R.inverse() * b_.R);

		return aBoxMinusB;
	}

	template <u32 L>
	Eigen::Matrix<f64, 9, 9> Quadcopter<L>::dFdxk(const QuadcopterState& a_, const Control& b_, const f64 dT_) const
	{
		// Determine dTf
		const Eigen::Vector<f64, 9> dTf = dT_ * f(a_, b_);

		// Determine Gx = I_n BS I_n BS Exp(-dt f[6, 8]), where BS is the block sum.
		// Calculate block over SO(3) using result in Table I
		const Eigen::Matrix3d Gx2 = Exp(-dTf.segment<3>(6));

		// Compose Gx as the full block sum with off-diagonal elements set to 0.
		Eigen::Matrix<f64, 9, 9> Gx(Eigen::Matrix<f64, 9, 9>::Identity());
		Gx.block<3, 3>(6, 6) = Gx2;

		// Do the same for Gf using the same reasoning as in dFduk
		Eigen::Matrix3d Gf2 = A(dTf.segment<3>(6));
		Gf2.transposeInPlace();

		// Obtain Gf
		Eigen::Matrix<f64, 9, 9> Gf(Eigen::Matrix<f64, 9, 9>::Identity());
		Gf.block<3, 3>(6, 6) = Gf2;

		// Determine (df/ddx)_{d = 0}. First, obtain skew(UnitZ)
		Eigen::Matrix3d skewUnitZ;
		skewUnitZ << 0.0, -1.0, 0.0,
					 1.0,  0.0, 0.0,
					 0.0,  0.0, 0.0;

		// Write out dfddx = (34a)
		Eigen::Matrix<f64, 9, 9> dfddx(Eigen::Matrix<f64, 9, 9>::Zero());
		dfddx.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
		dfddx.block<3, 3>(3, 6) = (b_(0) * a_.R * skewUnitZ);

		// Fxk = Gx + dT * Gf * (df/ddx)_{d = 0}
		return (Gx + (dT_ * Gf * dfddx));
	}

	template <u32 L>
	Eigen::Matrix<f64, 9, 4> Quadcopter<L>::dFduk(const QuadcopterState& a_, const Control& b_, const f64 dT_) const
	{
		// Determine dTf
		const Eigen::Vector<f64, 9> dTf = dT_ * f(a_, b_);

		// Determine Gf = I_n BS I_n BS (A (dt f[6, 8]))^T, where BS is the block sum.
		// Calculate block over SO(3) using result in Table I
		Eigen::Matrix3d Gf2 = A(dTf.segment<3>(6));
		Gf2.transposeInPlace();

		// Compose as block sum with off-diagonal values as 0. First two blocks are simply identity,
		// so can be left alone
		Eigen::Matrix<f64, 9, 9> Gf(Eigen::Matrix<f64, 9, 9>::Identity());
		Gf.block<3, 3>(6, 6) = Gf2;

		// Determine (df/ddu)_{d = 0}
		Eigen::Matrix<f64, 9, 4> dfddu(Eigen::Matrix<f64, 9, 4>::Zero());
		dfddu.block<3, 1>(3, 0) = -(a_.R * Eigen::Vector3d::UnitZ());
		dfddu.block<3, 3>(6, 1) = Eigen::Matrix3d::Identity();

		// Fuk = dT * Gf * (df/ddu)_{d = 0}
		return (dT_ * Gf * dfddu);
	}

	template <u32 L>
	Eigen::Vector3d Quadcopter<L>::Log(const Eigen::Matrix3d& R_)
	{
		// Determine sin(t) and t
		const f64 traceR = R_.trace();
		const f64 sinT   = 0.5 * std::sqrt((3.0 - traceR) * (1.0 + traceR));
		const f64 t      = std::asin(sinT);

		// Early return for case t ~= 0. Using f32 as small_num rather than actual eps
		if (std::abs(t) < std::numeric_limits<f32>::epsilon())
		{
			return Eigen::Vector3d::Zero();
		}

		// Base case
		const Eigen::Vector3d v(R_(2, 1) - R_(1, 2), R_(0, 2) - R_(2, 0), R_(1, 0) - R_(0, 1));
		return ((t / (2.0 * sinT)) * v);
	}

	template <u32 L>
	Eigen::Matrix3d Quadcopter<L>::Exp(const Eigen::Vector3d& w_)
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
		const f64 e1     = 0.5 - (normW2 / 24.0) + (normW4 / 720.0);

		// Final eq
		const Eigen::Matrix3d skewW2 = skewW * skewW;
		return (Eigen::Matrix3d::Identity() + (e0 * skewW) + (e1 * skewW2));
	}

	template <u32 L>
	Eigen::Matrix3d Quadcopter<L>::A(const Eigen::Vector3d& w_)
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
		return (Eigen::Matrix3d::Identity() + (e0 * skewW) + (e1 * skewW2));
	}
}
