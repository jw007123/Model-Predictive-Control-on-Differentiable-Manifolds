#pragma once

#include <iostream>
#include <functional>
#include <vector>
#include <cassert>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "osqp.h"

#include "Literals.h"

/// N:= Dim(State); M:= Dim(Control); L:= Lookahead
template <typename State, u32 N, u32 M, u32 L>
class NonLinearSolver
{
public:
	typedef Eigen::Vector<f64, M> Control;

	enum class CreateOptions : u8
	{
		None				= 0,				       // Assumes nothing
		PrecalculateQ		= 1,					   // Assumes CalculateQZero is constant at run-time
		PrecalculateP		= 1 << 1,				   // Assumes CalculatePZero is constant at run-time
		PrecalculateUBounds = 1 << 2,				   // Assumes CalculateUBounds is constant at run-time
		PrecalculateAll		= 1 | (1 << 1) | (1 << 2), // Assumes all functions are constant at run-time
		IgnoreUBounds		= 1 << 3				   // Assumes CalculateUBounds = [-inf, inf]
	};

	NonLinearSolver(const CreateOptions options_);

	/// State transition function
	virtual Eigen::Vector<f64, N> f(const State& a_, const Control& b_) const = 0;

	/// Update operator for x_{k + 1} = xk (+) dT * f(xk, uk)
	virtual State ApplyDelta(const State& a_, const Eigen::Vector<f64, N>& delta_, const f64 dT_) const = 0;

	/// Given x0, xk^d and uk^d, calculates uk^opt. Assumes all ptrs own at least L x sizeof(T) bytes
	bool Solve(Control* const ukopt_, const State& x0_, const State* const xkd_, const Control* const ukd_, const f64 dT_);

protected:
	/// this() and/or Solve() Constants
	virtual Eigen::Matrix<f64, N, N> CalculateQZero()   const = 0;
	virtual Eigen::Matrix<f64, M, M> CalculatePZero()   const = 0;
	virtual Eigen::Matrix<f64, M, 2> CalculateUBounds() const = 0;

	/// State functions
	virtual Eigen::Vector<f64, N>    BoxMinus(const State& a_, const State& b_)				  const = 0;
	virtual Eigen::Matrix<f64, N, N> dFdxk(const State& a_, const Control& b_, const f64 dT_) const = 0;
	virtual Eigen::Matrix<f64, N, M> dFduk(const State& a_, const Control& b_, const f64 dT_) const = 0;

private:
	/// Required for class to work. See osqp_configure for OSQPInt typedef
	static_assert(sizeof(OSQPFloat) == sizeof(f64));
	static_assert(sizeof(OSQPInt) == sizeof(Eigen::SparseMatrix<f64>::StorageIndex));

	std::vector<Eigen::Matrix<f64, N, N>> FxScratch;      // Length L
	std::vector<Eigen::Matrix<f64, N, M>> FuScratch;      // Length L
	Eigen::MatrixX<f64>					  matAScratch;    // x = A^-1 * b
	Eigen::VectorX<f64>					  vecBScratch;    // x = A^-1 * b
	bool								  firstSolveCall; // Default false
	const CreateOptions					  options;

	Eigen::MatrixX<f64>      matM;	// [N * L, M * L]
	Eigen::MatrixX<f64>      matH;	// [N * L, N]
	Eigen::MatrixX<f64>      matQ;	// [N * L, N * L]
	Eigen::MatrixX<f64>	     matP;  // [M * L, M * L]
	Eigen::MatrixX<f64>		 matdU; // [M * L, 2]
	Eigen::MatrixX<f64>		 matpU;	// [M * L, 1]
	Eigen::Matrix<f64, M, 2> matU;

	/// Optionally calculates matQ, matP and matU
	void CalculateConstants(const bool calcQ_, const bool calcP_, const bool calcU_);

	/// Solves an unconstrained QP problem analytically
	void SolveUnconstrainedQP(Control* const  ukopt_, const Eigen::Vector<f64, N>& dx0_);

	/// Solves a constrained QP problem via OSQP
	bool SolveConstrainedQP(Control* const ukopt_, const Eigen::Vector<f64, N>& dx0_);

	/// Helper function to add to larger matrices
	template <u32 R, u32 C>
	static void PushAToB(const Eigen::Matrix<f64, R, C>& a_, Eigen::MatrixX<f64>& b_, const u32 bRowStart_, const u32 bColStart_);

	/// Helper function to convert from Eigen::MatrixX<f64> to OSQPCscMatrix
	static void EigenToOSQPCsc(const Eigen::MatrixX<f64>& eigen_, OSQPCscMatrix& csc_);
};

template <typename State, u32 N, u32 M, u32 L>
NonLinearSolver<State, N, M, L>::NonLinearSolver(const NonLinearSolver::CreateOptions options_) :
											     options(options_)
{
	// Allocate scratch for Fu/Fx
	FxScratch.resize(L);
	FuScratch.resize(L);

	// Resize matM and matH here rather than Solve() due to repeated calls of latter
	matM.resize(N * L, M * L);
	matH.resize(N * L, N);

	// Resize matQ, mapP and matdU here too as, regardless of const-ness, their size doesn't change
	matQ.resize(N * L,  N * L);
	matP.resize(M * L,  M * L);
	matdU.resize(M * L, 2);

	// matpU is used in OSQP as is just a vector of ones
	matpU.resize(M * L, 1);
	matpU.setOnes();

	// Resize scratch used in the final QP solve
	matAScratch.resize(M * L, M * L);
	vecBScratch.resize(M * L);

	// Set all uninitialised 'mat' matrices to 0 in order to satisfy a few assumptions made later
	matM.setZero();
	matH.setZero();
	matQ.setZero();
	matP.setZero();
	matdU.setZero();

	// Used to determine when to call CalculateConstants()
	firstSolveCall = true;
}

template <typename State, u32 N, u32 M, u32 L>
bool NonLinearSolver<State, N, M, L>::Solve(Control* const ukopt_, const State& x0_, const State* const xkd_, const Control* const ukd_, const f64 dT_)
{
	if (firstSolveCall)
	{
		// Always calculate on the first call to Solve()
		CalculateConstants(true, true, true);
	}
	else
	{
		// On later calls, this is now dependent on CreateOptions
		CalculateConstants(!((u8)options & (u8)CreateOptions::PrecalculateQ),
						   !((u8)options & (u8)CreateOptions::PrecalculateP),
					       !((u8)options & (u8)CreateOptions::PrecalculateUBounds) && !((u8)options & (u8)CreateOptions::IgnoreUBounds));
	}

	// No longer on first call to Solve()
	firstSolveCall = false;

	// Determine dx0
	const Eigen::Vector<f64, N> dx0 = BoxMinus(x0_, xkd_[0]);

	// Fill Fx and Fu buffers to avoid recalculating Jacobians
	for (u32 i = 0; i < L; ++i)
	{
		// dxk = 0 => xk = xkd. Likewise for uk
		FxScratch[i] = dFdxk(xkd_[i], ukd_[i], dT_);
		FuScratch[i] = dFduk(xkd_[i], ukd_[i], dT_);
	}

	// Calculate matH
	Eigen::Matrix<f64, N, N> dFdxkMult(Eigen::Matrix<f64, N, N>::Identity());
	for (u32 i = 0; i < L; ++i)
	{
		dFdxkMult *= FxScratch[i];

		// Add Mult(Fxk)^T to matH
		const Eigen::Matrix<f64, N, N> dFdxkMultT = dFdxkMult.transpose();
		PushAToB(dFdxkMultT, matH, i * N, 0);
	}

	// Calculate matM. Most of matrix is zeroes, so default to zero
	matM.setZero();
	for (u32 i = 0; i < L; ++i)
	{
		dFdxkMult.setIdentity();

		// i32 to avoid wrap-around at j = 0 and --j >= 0 check
		for (i32 j = i; j >= 0; --j)
		{
			// Add Mult(Fxk) * Fuk to matM
			const Eigen::Matrix<f64, N, M> dFdxkMultFuk = dFdxkMult * FuScratch[j];
			PushAToB(dFdxkMultFuk, matM, i * N, j * M);

			// For i = 1, j = 0: dFdxkMult = FxScratch[1 - 0] = FxScratch[1] as required
			dFdxkMult *= FxScratch[i - j];
		}
	}

	// Solve for dU and place in ukopt_. Can determine dU exactly when the IgnoreUBounds flag is enabled
	bool error = false;
	if ((u8)options & (u8)CreateOptions::IgnoreUBounds)
	{
		SolveUnconstrainedQP(ukopt_, dx0);
	}
	else
	{
		// Determine dU = [dUmin, duMax]
		for (u32 i = 0; i < L; ++i)
		{
			matdU.block<M, 1>(i * M, 0) = matU.col(0) - ukd_[i];
			matdU.block<M, 1>(i * M, 1) = matU.col(1) - ukd_[i];
		}

		error = SolveConstrainedQP(ukopt_, dx0);
	}

	// If solving was OK, determine Uopt via Uopt = dU + Ud
	if (!error)
	{
		for (u32 i = 0; i < L; ++i)
		{
			ukopt_[i] = ukopt_[i] + ukd_[i];
		}
	}

	return !error;
}

template <typename State, u32 N, u32 M, u32 L>
void NonLinearSolver<State, N, M, L>::CalculateConstants(const bool calcQ_, const bool calcP_, const bool calcU_)
{
	if (calcQ_)
	{
		const Eigen::Matrix<f64, N, N> matQZero = CalculateQZero();
		for (u32 i = 0; i < L; ++i)
		{
			PushAToB(matQZero, matQ, i * N, i * N);
		}
	}

	if (calcP_)
	{
		const Eigen::Matrix<f64, M, M> matPZero = CalculatePZero();
		for (u32 i = 0; i < L; ++i)
		{
			PushAToB(matPZero, matP, i * M, i * M);
		}
	}

	if (calcU_)
	{
		matU = CalculateUBounds();
	}
}

template <typename State, u32 N, u32 M, u32 L>
template <u32 R, u32 C>
void NonLinearSolver<State, N, M, L>::PushAToB(const Eigen::Matrix<f64, R, C>& a_, Eigen::MatrixX<f64>& b_, const u32 bRowStart_, const u32 bColStart_)
{
	b_.block<R, C>(bRowStart_, bColStart_) = a_;
}

template <typename State, u32 N, u32 M, u32 L>
void NonLinearSolver<State, N, M, L>::EigenToOSQPCsc(const Eigen::MatrixX<f64>& eigen_, OSQPCscMatrix& csc_)
{
	// MatrixX -> SparseMatrix
	Eigen::SparseMatrix<f64> sparse;
	sparse = eigen_.sparseView();

	// SparseMatrix must be in compressed form
	sparse.makeCompressed();

	// Use helper function rather than setting values ourselves
	csc_set_data(&csc_, sparse.innerSize(), sparse.outerSize(), sparse.nonZeros(),
				 (OSQPFloat*)sparse.valuePtr(),
				 (OSQPInt*)sparse.innerIndexPtr(),
				 (OSQPInt*)sparse.outerIndexPtr());
}

template <typename State, u32 N, u32 M, u32 L>
void NonLinearSolver<State, N, M, L>::SolveUnconstrainedQP(Control* const ukopt_, const Eigen::Vector<f64, N>& dx0_)
{
	// Need to split operation up due to Eigen aliasing
	matAScratch  = matM.transpose();
	matAScratch *= matQ * matM;
	vecBScratch  = (matM.transpose()) * matQ * matH * dx0_ * f64(-1.0);

	// Solve for dU https://eigen.tuxfamily.org/dox/classEigen_1_1CompleteOrthogonalDecomposition.html
	Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixX<f64>> exactSolver(matAScratch);
	const Eigen::Matrix<f64, M * L, 1> dU = exactSolver.solve(vecBScratch);

	// Place dU in vector
	for (u32 i = 0; i < L; ++i)
	{
		ukopt_[i] = dU.block<M, 1>(i * M, 0);
	}
}

template <typename State, u32 N, u32 M, u32 L>
bool NonLinearSolver<State, N, M, L>::SolveConstrainedQP(Control* const ukopt_, const Eigen::Vector<f64, N>& dx0_)
{
	// Need to split operation up due to Eigen aliasing
	matAScratch  = matM.transpose();
	matAScratch *= matQ * matM;
	vecBScratch  = (matM.transpose()) * matQ * matH * dx0_ * f64(-1.0);

	// osqp assumes matA is symmetric and only wants the upper-triangular coeffs.
	// To make sure te EigenToOSQPCsc gives just these, zero the lower-triangular coeffs
	for (u32 i = 0; i < matAScratch.rows(); ++i)
	{
		for (u32 j = 0; j < i; ++j)
		{
			matAScratch(i, j) = 0.0;
		}
	}
	assert(matAScratch.isUpperTriangular());

	// Convert to Eigen sparse and then csc
	OSQPCscMatrix cscA;
	OSQPCscMatrix cscpU;
	EigenToOSQPCsc(matAScratch, cscA);
	EigenToOSQPCsc(matpU,		cscpU);

	// Setup osqp vars
	OSQPSolver*	 osqpSolver;
	OSQPSettings osqpSettings;
	OSQPInt		 osqpExitFlag;

	// Uses settings as in osqp_api_constants
	osqp_set_default_settings(&osqpSettings);

	// Setup solving and check for errors
	const OSQPFloat* uMin = (OSQPFloat*)matdU.col(0).data();
	const OSQPFloat* uMax = (OSQPFloat*)matdU.col(1).data();
	OSQPInt solveErrors   = 0;
	osqpExitFlag		  = osqp_setup(&osqpSolver, &cscA, vecBScratch.data(), &cscpU, uMin, uMax, M * L, M * L, &osqpSettings);
	if (!osqpExitFlag)
	{
		solveErrors = osqp_solve(osqpSolver);

		// Pass solution across
		for (u32 i = 0; i < L; ++i)
		{
			for (u32 j = 0; j < M; ++j)
			{
				ukopt_[i](j) = osqpSolver->solution->x[i * M + j];
			}
		}
	}

	// Cleanup
	osqp_cleanup(osqpSolver);

	return (osqpExitFlag == 0) && (solveErrors == 0);
}
