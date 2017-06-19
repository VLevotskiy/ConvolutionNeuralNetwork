#ifndef LAYER_H
#define LAYER_H
#include "neuron.h"
#include <vector>
#include <string>
#include <stdint.h>
#include <math.h>

//Функции активации
float SIGMOID(float S){
   return 1/(1+exp(-S));
}
//Производные функций активации
float DSIGMIOD(float S) {
    return (1-(SIGMOID(S))*(SIGMOID(S)));
}

float ReLU(float S) {
    if (S < 0) return 0;
    else return S;
}

float SoftMax(float){
    return 0;
}

enum Layer_type{FullConnected=1,Convolution,Pooling};
const int NUM_OF_LAYERS_TYPES = 3;
const int NUM_OF_ACTIVATION_FUNCS = 3;
const std::string list_of_layers_types[NUM_OF_LAYERS_TYPES] = {"FullConnected", "Convolution", "Pooling"};
const std::string list_of_activation_funcs[NUM_OF_ACTIVATION_FUNCS] = {"Sigmoid", "ReLU", "SoftMax"};

//Класс слоя. Хранит информацию о слое сети
class Layer {
private:
    std::vector<Neuron> neurons;
    unsigned int layer_size;
    Layer* prev_layer;
    Layer_type type;
    float (*activation_func)(float);
public:
    Layer(int n, Layer* prev, std::string &Activation_function);
    unsigned int Size() const;// { return layer_size;}
    std::vector<Neuron>& Get_neurons() const;
    Layer* Get_Prev() const;
    Layer_type Get_type() const;
    virtual void Calculate = 0;
};

class FullConnected_Layer : public Layer{
public:
    FullConnected_Layer(int n, Layer* prev, std::string &Activation_function);
    void Calculate();
};

class Convolution_Layer : public Layer {
private:
    /*unsigned char num_of_masks;
    uint16_t structural_width;
    uint16_t structural_height;
    uint8_t step_size;*/

    void convolution(float* input,const size_t input_width,const size_t input_height, float* mask, size_t el_width, size_t el_height);
public:
    Convolution_Layer(int n, Layer* prev, std::string& Activation_function, uint8_t el_width, uint8_t el_height, uint8_t num_of_masks);
    void Calculate();
};

class Pooling_Layer : public Layer {
    unsigned char num_of_masks;
    uint16_t structural_width;
    uint16_t structural_height;
    uint8_t step_size;

    void pooling(float* input, const size_t width,const size_t height,const unsigned char el_size, const unsigned char step );
public:
    Pooling_Layer(int n, Layer* prev,std::string& Activation_function, uint16_t input_width, uint16_t input_height, uint8_t step_size, uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks);
    void Calculate();
};

#endif // LAYER_H
