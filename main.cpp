#include "util.h"
#include "predict_model.h"

#include <chrono>
#include <ctime>
#include <ratio>
#include <iostream>
#include <thread>
#include <Eigen/Dense>
int main(int argc, char **argv)
{
	Eigen::initParallel();	
	std::string vocab_path = "vocab_file";
	std::unordered_map<std::string,int> vocab2id;
	std::vector<std::string> id2vocab;
	readVocab(vocab_path, vocab2id, id2vocab);
	std::cout << "finish reading vocab_file" << std::endl;	
	
	// batch_size : # of sentences per batch
	int batch_size = 1;
	// num_step : # of words per query
	int max_input_len = 1;
		// topN : # of answers per query
	int topN = 3;
	
	// myPredictor would be nullptr if the following failed!
	auto myPredictor = PredictModel::createModel();
	if(myPredictor == nullptr)
		exit(1);
	std::cout << "finish loading model" << std::endl;	
		
	std::ifstream file("small_data_file");
	float total_time = 0;	
	int total_iter = 100;
	int iter = 0;
	std::vector<std::vector<int>> data = readBatchFromFile(file, vocab2id, max_input_len, batch_size);
	std::vector<std::vector<std::pair<int, float>>> res(data.size());
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();	
	while(iter < total_iter) 
	{
		myPredictor->predict(data, res, topN);
		/*
		std::cout << "input: " << std::endl;
		for(int i = 0; i < data[0].size(); i++)
		{
			std::cout << id2vocab[data[0][i]];
		}
		std::cout << std::endl;
		for(int i = 0; i < topN; i++)
		{
			std::cout << i << "th answer: " << id2vocab[res[0][i].first] << " with prob: " << res[0][i].second << std::endl;
		}
		*/
		iter++;
	}
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time_span = t2 - t1;
	
	total_time += time_span.count();
	std::cout<<"total_time:"<<total_time<<std::endl;
	
	std::cout << "num sentence: " << total_iter << " average time per sentence: " << total_time/total_iter << std::endl;
	
	PredictModel::destroyModel(myPredictor);
	
	return 0;
}
