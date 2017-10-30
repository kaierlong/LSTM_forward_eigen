#ifndef _LSTM_MODEL_H_
#define _LSTM_MODEL_H_

#include <Eigen/Dense>
#include "predict_model.h"

class LstmModel: public PredictModel
{
private:
	Eigen::MatrixXf embed;
	
	std::vector<Eigen::MatrixXf> W;	
	std::vector<Eigen::VectorXf> b;	

	std::vector<Eigen::VectorXf> c;
	std::vector<Eigen::VectorXf> h;
	
	Eigen::MatrixXf softmax_W;		
	Eigen::VectorXf softmax_b;

	int vocab_size;
	int embed_size;
	int num_hidden_layer;
	std::vector<int> layer_size;

	void computeRecurrentLayer(int layer, int input_id);

public:
	LstmModel(){};
	virtual bool init();
	virtual void predict(std::vector<std::vector<int>>& data, std::vector<std::vector<std::pair<int, float>>>& res, int K);
	virtual ~LstmModel(){};
};

#endif
