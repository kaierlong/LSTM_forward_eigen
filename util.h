#ifndef _UTIL_H_
#define _UTIL_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>

void readVocab(std::string vocab_path, std::unordered_map<std::string,int>& vocab2id, std::vector<std::string>& id2vocab);

void readWordid2rc(std::string wordid2rc_path, std::unordered_map<int, std::pair<int, int>>& wordid2rc);

std::vector<std::vector<int>> readBatchFromFile(std::istream& file, std::unordered_map<std::string,int>& vocab2id, int max_input_len, int batch_size);
#endif

