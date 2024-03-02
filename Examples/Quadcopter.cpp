#include "Quadcopter.h"

namespace ModelPredictiveControl
{
	QuadcopterState::QuadcopterState()
	{
		pI.setZero();
		vI.setZero();
		R.setIdentity();
	}
}
