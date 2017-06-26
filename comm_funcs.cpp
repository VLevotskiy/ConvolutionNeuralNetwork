#ifndef CONVOLUTIONNN_H
#define CONVOLUTIONNN_H
#include <memory.h>
#include <stdlib.h>
#include <iostream>
#include "comm_funcs.h"

 double gaussrand_t( double MO,  double sko)
{
     double sum = 0, x;
    for (int i = 0; i<25; i++)
        sum += 1.0*rand() / RAND_MAX;
    x = (sqrt(2.0)*(sko)*(sum - 12.5)) / 1.99661 + MO;

    return x;
}

void gen_array(const  double N, const  double M, const int Z,  double *array)
{
     double average = (N + M) / 2.;
     double sigma = (average - N) / 3.;

    for (int i = 0; i<Z; i++) {
         double new_value = gaussrand_t(average, sigma);

         //есть вероятность (0.3%) что сгенерированное число выйдет за нужный нам диапазон
        while (new_value < N || new_value > M)
            new_value = gaussrand_t(average, sigma);

        array[i] = new_value;
    }
}

int8_t Parser(std::string& input,const std::string* possible_values_list, uint8_t num_of_possible) {
    int8_t command_num = -1;
    for (int i = 0; i < num_of_possible; i++){
        size_t first_word_ptr = input.find(possible_values_list[i]);

        if (first_word_ptr != std::string::npos) {
            if (first_word_ptr == 0) {
                command_num = i;
                break;
            }
        }
    }
    return command_num;
}

std::vector<std::string>& Get_words(std::string& str, std::vector<std::string>& arr,std::string delim){
    size_t prev = 0;
    size_t next;
    size_t delta = delim.length();

    while( ( next = str.find( delim, prev ) ) != std::string::npos ){
      arr.push_back( str.substr( prev, next-prev ) );
      prev = next + delta;
    }
    arr.push_back( str.substr( prev ) );
    return arr;
}

 double SIGMOID( double S){
   return 1/(1+exp(-S));
}
//Производные функций активации
 double DSIGMIOD( double S) {
    //return (1-(SIGMOID(S))*(SIGMOID(S)));
    return SIGMOID(S) * (1 - SIGMOID(S));
}

 double ReLU( double S) {
    if (S < 0) return 0;
    else return S;
}

 double Linear( double S) {
    return S;
}

 double SoftMax( double S){
    return S;
}
/*void pooling( double* input, const size_t width,const size_t height,const unsigned char el_size, const unsigned char step );

void convolution( double* input,const uint16_t input_width,const uint16_t input_height,  double* mask, size_t el_width, size_t el_height){

     double out[input_height * input_width];
    memset(out, 0, sizeof(int) * input_width*input_height);

    int start_w_offset = (int)el_width * 0.5;
    int start_h_offset = (int)el_height * 0.5;

    for (int i =0; i <  input_height; i++){           //cтроки
        int temp_h = i - start_h_offset;
        for(int j = 0; j  < input_width; j++) {       //столбцы
            int temp_w = j - start_w_offset;
            for (int m = 0; m <  el_height; m++){   //строки маски
                for (int n = 0; n < el_width; n++){  //столбцы маски
                    int h_pos = temp_h + m;
                    int w_pos = temp_w + n;
                    if (h_pos < 0) continue;//h_pos = m;
                    if (w_pos < 0) continue;//w_pos = n;
                    if (h_pos >= input_height)continue;// h_pos = input_height-1-m + start_h_offset;
                    if (w_pos >= input_width) continue;//w_pos = input_width-1-n + start_w_offset;

                    out[i * input_height + j] += input[h_pos * input_height + w_pos] * mask[m * el_height + n];
                }
            }
        }
    }

    for (uint16_t i = 0; i <input_height * input_width; i++){
        if (out[i] <0) out[i] = 0;
    }



    pooling(out,input_width,input_height,2,2);
}

void pooling( double* input, const size_t width,const size_t height,const unsigned char el_size, const unsigned char step ){
    unsigned int new_w = width/el_size, new_h = height/el_size;
    if (width % el_size != 0) new_w +=1;
    if (height % el_size != 0) new_h +=1;
     double* out = ( double*)malloc(new_w*new_h*sizeof( double));
    for (int i =0, out_i =0; i < height; i+=step, out_i++) {
        for (int j = 0,out_j = 0; j < width; j+=step, out_j++){

             double max = input[i * width + j];
            for (int m = 0; m < el_size;m++){
                for (int n = 0; n < el_size;n++) {
                    const int  pos = (i+m)*width + j + n;
                    if(j+n >= width || i+m >=height){break;}
                    if (input[pos] > max) {
                        max = input[pos];
                    }
                }
            }

            out[out_i * new_w + out_j] = max;
        }
    }
    for (int i = 0; i < new_h; i++){
        for (int j = 0; j < new_w; j++) {
            std::cout << out[i*new_h + j];
        }
    }
}*/



#endif // CONVOLUTIONNN_H
