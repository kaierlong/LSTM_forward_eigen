#include <iostream>
#include <stdexcept>
#include <algorithm> 
#include <float.h>
#include <chrono>
#include <ctime>
#include <math.h>
#include "lstm_model.h"

using namespace Eigen;

float my_sigmoid(float a)
{
	return 1/(1+exp(-a));
}

float my_tanh(float a)
{
	return tanhf(a);
}

VectorXf softmax(VectorXf& vec)
{
	float max = vec.maxCoeff();
	vec = (vec.array() - max).exp();
	float sum = vec.sum();
	VectorXf res = vec.array() / sum;
	return res;
}

RowVectorXf softmax(RowVectorXf& vec)
{
	float max = vec.maxCoeff();
	vec = (vec.array() - max).exp();
	float sum = vec.sum();
	RowVectorXf res = vec.array() / sum;
	return res;
}

MatrixXf softmax(MatrixXf& mat)
{
	RowVectorXf col_max = mat.colwise().maxCoeff();
	MatrixXf res = (mat.array().rowwise() - col_max.array()).exp();
	RowVectorXf col_sum = res.colwise().sum();
	res = res.array().rowwise() / col_sum.array();
	return res;
}

void topK(RowVectorXf vec, std::vector<std::pair<int, float>>& res, int K)
{
	for(int i = 0; i < K; i++)
	{
		RowVectorXf::Index max_ind;
		float max_score = vec.maxCoeff(&max_ind);
		vec(max_ind) = -FLT_MAX;
		std::pair<int, float> p = std::make_pair(max_ind, max_score);
		res.push_back(p);
	}
}

void topK(VectorXf vec, std::vector<std::pair<int, float>>& res, int K)
{
	for(int i = 0; i < K; i++)
	{
		VectorXf::Index max_ind;
		float max_score = vec.maxCoeff(&max_ind);
		vec(max_ind) = -FLT_MAX;
		std::pair<int, float> p = std::make_pair(max_ind, max_score);
		res.push_back(p);
	}
}

void goToDelimiter(int delim, FILE *fi)
{
	int ch=0;
	while (ch != delim) {
		ch = fgetc(fi);
		if (feof(fi)) {
			std::cout << "Unexpected end of file" << std::endl;
			throw std::exception();
		}
	}
}

bool loadWeight(const char *file_name, MatrixXf& mat, int row, int col)
{
	FILE *pFile = fopen(file_name, "rb");
	if(pFile == NULL) 
	{
		std::cout << "Model file is not found!" << std::endl;
		return false;
	}

	float fl;
	mat.resize(row, col);
	for(int r = 0; r < row; r++)
	{
		for(int c = 0; c < col; c++)
		{
			fread(&fl, 4, 1, pFile);
			if (std::isnan(fl))
			{
				printf("result is nan \n");
				return false;
			}
			mat(r, c) = fl;
		}
	}
	mat.transposeInPlace();
	fclose(pFile);
	return true;
}

bool loadWeight(const char *file_name, VectorXf& vec, int size)
{
	FILE *pFile = fopen(file_name, "rb");
	if(pFile == NULL) 
	{
		std::cout << "Model file is not found!" << std::endl;
		return false;
	}

	float fl;
	vec.resize(size);
	for(int i = 0; i < size; i++)
	{
		fread(&fl, 4, 1, pFile);
		if (std::isnan(fl))
		{
			printf("result is nan \n");
			return false;
		}
		vec(i) = fl;			
	}
	fclose(pFile);
	return true;
}

bool LstmModel::init()
{	
	FILE *pFile = fopen("model_config", "rb");
	if(pFile == NULL) 
	{
		std::cout << "Model file is not found!" << std::endl;
		return false;
	}
		
	try 
	{	
		goToDelimiter(':', pFile);
		fscanf(pFile, "%d", &vocab_size);

		goToDelimiter(':', pFile);
		fscanf(pFile, "%d", &embed_size);

		goToDelimiter(':', pFile);
		fscanf(pFile, "%d", &num_hidden_layer);
			
		int layersize;
		for(int layer = 0; layer < num_hidden_layer; layer++)
		{
			goToDelimiter(':', pFile);
			fscanf(pFile, "%d", &layersize);
			layer_size.push_back(layersize);
		}
	} catch(const std::exception& e) {
		return false;
	}
	fclose(pFile);
	
	loadWeight("./embedding", embed, vocab_size, embed_size);	
	
	W = std::vector<MatrixXf>(num_hidden_layer);
	b = std::vector<VectorXf>(num_hidden_layer);
	c = std::vector<VectorXf>(num_hidden_layer);
	h = std::vector<VectorXf>(num_hidden_layer);

	for(int layer = 0; layer < num_hidden_layer; layer++)
	{
		int	nRows = layer_size[layer] + (layer == 0 ? embed_size : layer_size[layer-1]);
		int nCols= layer_size[layer];
	
		loadWeight("./kernel", W[layer], nRows, 4*nCols);	
		loadWeight("./bias", b[layer], 4*nCols);	
	}
	
	loadWeight("./softmax_w", softmax_W, layer_size[num_hidden_layer-1], embed_size);	
	loadWeight("./softmax_b", softmax_b, embed_size);	
	
	return true;
}

void LstmModel::computeRecurrentLayer(int layer, int input_id)
{
	if(layer == 0)
	{
		int hidden_size = layer_size[layer];
		int nCols1 = 4*hidden_size;
		VectorXf input = embed.col(input_id);
		VectorXf concats(4*hidden_size);
#pragma omp parallel
{
        	int num_threads = omp_get_num_threads();
        	int tid = omp_get_thread_num();
        	int n_per_thread = nCols1 / num_threads;
        	if ((n_per_thread * num_threads < nCols1)) n_per_thread++;
        	int start = tid * n_per_thread;
        	int len = n_per_thread;
        	if (tid + 1 == num_threads) len = nCols1 - start;
        	if(start < nCols1)
			concats.segment(start, len) = W[layer].block(start, 0, len, embed_size+hidden_size) * (VectorXf(embed_size + hidden_size) << input, h[layer]).finished() + b[layer].segment(start, len);
}
		VectorXf i = concats.segment(0, hidden_size);
		VectorXf j = concats.segment(hidden_size, hidden_size);
		VectorXf f = concats.segment(hidden_size*2, hidden_size);
		VectorXf o = concats.segment(hidden_size*3, hidden_size);

		c[layer] = c[layer].array() * f.unaryExpr(std::ptr_fun(my_sigmoid)).array() + i.unaryExpr(std::ptr_fun(my_sigmoid)).array() * j.unaryExpr(std::ptr_fun(my_tanh)).array();
		h[layer] = c[layer].unaryExpr(std::ptr_fun(my_tanh)).array() * o.unaryExpr(std::ptr_fun(my_sigmoid)).array();
	}
	else
	{
		// This model has only one layer
	}	
}

void LstmModel::predict(std::vector<std::vector<int>>& data, std::vector<std::vector<std::pair<int, float>>>& res, int K)
{
	for(int i = 0; i < data.size(); i++)
	{
		for(int layer = 0; layer < num_hidden_layer; layer++)
		{
			h[layer].resize(layer_size[layer]);
			h[layer].setZero();
			c[layer].resize(layer_size[layer]);
			c[layer].setZero();
		}
		
		for(int step = 0; step < data[i].size(); step++)
		{
			for(int layer = 0; layer < num_hidden_layer; layer++)
			{
				computeRecurrentLayer(layer, data[i][step]);
			}
		}
		VectorXf logits = softmax_W * h[num_hidden_layer-1] + softmax_b;
		int nCols3 = vocab_size;
		RowVectorXf temp(vocab_size);
		RowVectorXf logits_T = logits.transpose();
#pragma omp parallel
{
        	int num_threads = omp_get_num_threads();
        	int tid = omp_get_thread_num();
        	int n_per_thread = nCols3 / num_threads;
        	if ((n_per_thread * num_threads < nCols3)) n_per_thread++;
        		int start = tid * n_per_thread;
        	int len = n_per_thread;
        	if (tid + 1 == num_threads) len = nCols3 - start;
        	if(start < nCols3)
			temp.segment(start, len) = logits_T * embed.block(0, start, embed_size, len);
}
		RowVectorXf probs = softmax(temp);
		topK(probs, res[i], K);
	}
}
