#include "predict_model.h"
#include "lstm_model.h"

PredictModel* PredictModel::createModel()
{
	PredictModel* pm = new LstmModel();
	if(pm->init())
	{
		return pm;
	}
	destroyModel(pm);
	return nullptr;
}

void PredictModel::destroyModel(PredictModel* pm)
{
	delete pm;
}
