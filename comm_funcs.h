//#pragma once
#ifndef RANDOM_NUM_H
#define RANDOM_NUM_H
#include <math.h>
#include <random>

float gaussrand_t(float MO, float sko);
void gen_array(const float N, const float M, const int Z, float *array);
int8_t Parser(std::string& input,const std::string* possible_values_list, uint8_t num_of_possible);
std::vector<std::string>& Get_words(std::string& str, std::vector<std::string>& arr,std::string delim);

//Функции активации
float SIGMOID(float S);
float ReLU(float S);
float Linear(float S);
float SoftMax(float S);
//Производные функций активации
float DSIGMIOD(float S);

#endif // RANDOM_NUM_H
