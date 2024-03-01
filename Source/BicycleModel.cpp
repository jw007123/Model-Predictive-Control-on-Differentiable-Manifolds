#include "BicycleModel.h"

namespace ModelPredictiveControl
{
	BicycleModelState::BicycleModelState()
	{
		p.setZero();
		v = 0.0;
		R.setIdentity();
	}
}
