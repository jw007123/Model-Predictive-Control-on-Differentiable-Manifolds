#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include "Literals.h"
#include "NonLinearSolver.h"

namespace ModelPredictiveControl
{
	struct BicycleModelState
	{
		Eigen::Vector2d p;
		f64				v;
		Eigen::Matrix2d R;

		BicycleModelState();
	};

	template <u32 L>
	class BicycleModel : public NonLinearSolver<BicycleModelState, 4, 2, L>
	{
	public:
		/// Inherit ctor from base class
		using NonLinearSolver<BicycleModelState, 4, 2, L>::NonLinearSolver;

		/// Convenience
		using CreateOptions = typename NonLinearSolver<BicycleModelState, 4, 2, L>::CreateOptions;
		using Control	    = typename NonLinearSolver<BicycleModelState, 4, 2, L>::Control;

		/// Virtuals: functions
		Eigen::Vector<f64, 4> f(const BicycleModelState& a_, const Control& b_)											  const;
		BicycleModelState     ApplyDelta(const BicycleModelState& a_, const Eigen::Vector<f64, 4>& delta_, const f64 dT_) const;

		/// Helper function for whether two states are approximately equal
		static bool IsApprox(const BicycleModelState& a_, const BicycleModelState& b_, const f64 eulTolSq_, const f64 rotTolSq_);

		/// Generic setter...
		void SetWheelBase(const f64 wheelBaseM_);

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
		f64 wheelBaseM;

		/// SO(2) Log function (7) https://arxiv.org/pdf/2106.15233.pdf
		static f64 Log(const Eigen::Matrix2d& R_);

		/// SO(2) Exp function (7) https://arxiv.org/pdf/2106.15233.pdf
		static Eigen::Matrix2d Exp(const f64& w_);
	};

	template <u32 L>
	Eigen::Vector<f64, 4> BicycleModel<L>::f(const BicycleModelState& a_, const Control& b_) const
	{
		const f64 w = Log(a_.R);

		Eigen::Vector<f64, 4> fk;
		fk(0) = a_.v * std::cos(w);
		fk(1) = a_.v * std::sin(w);
		fk(2) = b_(0);
		fk(3) = (a_.v / wheelBaseM) * std::tan(b_(1));

		return fk;
	}

	template <u32 L>
	BicycleModelState BicycleModel<L>::ApplyDelta(const BicycleModelState& a_, const Eigen::Vector<f64, 4>& delta_, const f64 dT_) const
	{
		// For Euclidean components of a_, OP := +_n. For SO(2) R comp, OP := x * Exp(R)
		BicycleModelState b = a_;
		b.p				   += (dT_ * delta_.segment<2>(0));
		b.v				   += (dT_ * delta_(2));
		b.R				   *= (dT_ * Exp(delta_(3)));

		return b;
	}

	template <u32 L>
	bool BicycleModel<L>::IsApprox(const BicycleModelState& a_, const BicycleModelState& b_, const f64 eulTolSq_, const f64 rotTolSq_)
	{
		// Use correct L2 norm for Euclidean spaces and SO(2) space
		if ((a_.p - b_.p).squaredNorm() > eulTolSq_)
		{
			return false;
		}
		if (((a_.v - b_.v) * (a_.v - b_.v)) > eulTolSq_)
		{
			return false;
		}

		const f64 Lab = Log(a_.R.inverse() * b_.R);
		if ((Lab * Lab) > rotTolSq_)
		{
			return false;
		}

		return true;
	}

	template <u32 L>
	void BicycleModel<L>::SetWheelBase(const f64 wheelBaseM_)
	{
		wheelBaseM = wheelBaseM_;
	}

	template <u32 L>
	Eigen::Matrix<f64, 4, 4> BicycleModel<L>::CalculateQZero() const
	{
		return Eigen::Matrix<f64, 4, 4>::Identity();
	}

	template <u32 L>
	Eigen::Matrix<f64, 2, 2> BicycleModel<L>::CalculatePZero() const
	{
		return (Eigen::Matrix<f64, 2, 2>::Identity() * 5.0);
	}

	template <u32 L>
	Eigen::Matrix<f64, 2, 2> BicycleModel<L>::CalculateUBounds() const
	{
		constexpr const f64 pi = 3.14159265358979323846;

		Eigen::Matrix<f64, 2, 2> U;
		U.col(0) = Eigen::Vector<f64, 2>(-3.0, -pi * 0.25);
		U.col(1) = Eigen::Vector<f64, 2>(3.0, pi * 0.25);

		return U;
	}

	template <u32 L>
	Eigen::Vector<f64, 4> BicycleModel<L>::BoxMinus(const BicycleModelState& a_, const BicycleModelState& b_) const
	{
		// For R^n, boxminus = -n
		// For SO(2), boxminus = Log(a^-1 * b)
		// Use manifold composition of boxminus to obtain result
		Eigen::Vector<f64, 4> aBoxMinusB;
		aBoxMinusB.segment<2>(0) = b_.p - a_.p;
		aBoxMinusB(2)			 = b_.v - a_.v;
		aBoxMinusB(3)			 = Log(a_.R.inverse() * b_.R);

		return aBoxMinusB;
	}

	template <u32 L>
	Eigen::Matrix<f64, 4, 4> BicycleModel<L>::dFdxk(const BicycleModelState& a_, const Control& b_, const f64 dT_) const
	{
		// Precalc for dfddx
		const f64 c1 = std::cos(Log(a_.R));
		const f64 s1 = std::sin(Log(a_.R));

		// Determine Gx = I_n BS 1 BS 1, where BS is the block sum. I.e. the identity
		const Eigen::Matrix<f64, 4, 4> Gx(Eigen::Matrix<f64, 4, 4>::Identity());

		Eigen::Matrix<f64, 4, 4> dfddx(Eigen::Matrix<f64, 4, 4>::Zero());
		dfddx(0, 2) = c1;
		dfddx(1, 2) = s1;
		dfddx(0, 3) = -a_.v * s1;
		dfddx(1, 3) = a_.v * c1;
		dfddx(3, 2) = (1.0 / wheelBaseM) * std::tan(b_(1));

		// Fxk = Gx + dT * Gf * (df/ddx)_{d = 0}. Gf is the identity and can be ignored
		return (Gx + (dT_ * dfddx));
	}

	template <u32 L>
	Eigen::Matrix<f64, 4, 2> BicycleModel<L>::dFduk(const BicycleModelState& a_, const Control& b_, const f64 dT_) const
	{
		// Precalc for dfddu
		const f64 c1 = std::cos(b_(1));

		Eigen::Matrix<f64, 4, 2> dfddu(Eigen::Matrix<f64, 4, 2>::Zero());
		dfddu(2, 0) = 1.0;
		dfddu(3, 1) = (a_.v / wheelBaseM) * (1.0 / (c1 * c1));

		// Fuk = dT * Gf * (df/ddu)_{d = 0}. Gf is the identity and can be ignored
		return (dT_ * dfddu);
	}

	template <u32 L>
	f64 BicycleModel<L>::Log(const Eigen::Matrix2d& R_)
	{
		return std::atan2(R_(1, 0), R_(0, 0));
	}

	template <u32 L>
	Eigen::Matrix2d BicycleModel<L>::Exp(const f64& w_)
	{
		const f64 cw = std::cos(w_);
		const f64 sw = std::sin(w_);

		Eigen::Matrix2d R;
		R << cw, -sw,
			 sw,  cw;

		return R;
	}
}
