//#pragma once
#ifndef RANDOM_NUM_H
#define RANDOM_NUM_H
#include <math.h>
#include <random>

 double gaussrand_t( double MO,  double sko);
void gen_array(const  double N, const  double M, const int Z,  double *array);
int8_t Parser(std::string& input,const std::string* possible_values_list, uint8_t num_of_possible);
std::vector<std::string>& Get_words(std::string& str, std::vector<std::string>& arr,std::string delim);

//Функции активации
 double SIGMOID_f( double S);
 double ReLU_f( double S);
 double Linear_f( double S);
 double SoftMax_f( double S);
//Производные функций активации
 double DSIGMOID( double S);
 double DReLU(double S);
 double DLinear(double S);
 double DSoftMax(double S);

#endif // RANDOM_NUM_H
