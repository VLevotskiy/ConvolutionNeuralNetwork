#ifndef LAYER_H
#define LAYER_H
#include "neuron.h"
#include <vector>
#include <string>
#include <stdint.h>
#include <math.h>
#include <memory>


enum Layer_type{Input=0, FullConnected=1, Convolution=2, Pooling=3};
const int NUM_OF_LAYERS_TYPES = 3;
const int NUM_OF_ACTIVATION_FUNCS = 4;
const std::string list_of_layers_types[NUM_OF_LAYERS_TYPES] = {"FullConnected", "Convolution", "Pooling"};
const std::string list_of_activation_funcs[NUM_OF_ACTIVATION_FUNCS] = {"Sigmoid", "ReLU", "SoftMax", "Linear"};

//Класс слоя. Хранит информацию о слое сети
class Layer {
protected:
    std::vector<Neuron> neurons;
    unsigned int layer_size;
    std::shared_ptr<Layer>  prev_layer;
    Layer_type type;
    float (*activation_func)(float);

    void Create_neurons();
public:
    Layer() {
        neurons.clear();
        layer_size =0;
        activation_func = nullptr;
    }

    Layer(unsigned int n, std::shared_ptr<Layer>& prev, std::string &Activation_function, Layer_type type_);
    unsigned int Size() const;// { return layer_size;}
    std::vector<Neuron>* Get_neurons();
    std::shared_ptr<Layer> Get_Prev() const;
    Layer_type Get_type() const;
    virtual void Calculate(){ }
};

class FullConnected_Layer : public Layer{
public:
    FullConnected_Layer(unsigned int n, std::shared_ptr<Layer> prev, std::string &Activation_function);
    void Calculate();
};

class Convolution_Layer : public Layer {
private:
    uint16_t height;
    uint16_t width;
    uint16_t mask_size;
    /*unsigned char num_of_masks;
    uint16_t structural_width;
    uint16_t structural_height;
    uint8_t step_size;*/

    void convolution(float* input,const size_t input_width,const size_t input_height, float* mask, size_t el_width, size_t el_height);
public:
    Convolution_Layer(std::shared_ptr<Layer> prev,std::string& Activation_function,\
                      uint16_t input_height,uint16_t input_width,\
                      uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks);
    void Calculate();
};

class Pooling_Layer : public Layer {
    unsigned char num_of_masks;
    uint16_t mask_size;
    uint16_t structural_width;
    uint16_t structural_height;
    uint8_t step_size;

    void pooling(float* input, const size_t width,const size_t height,const unsigned char el_size, const unsigned char step );
public:
    Pooling_Layer(std::shared_ptr<Layer> prev,std::string& Activation_function, uint16_t input_width, uint16_t input_height, uint8_t step_size, uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks);
    void Calculate();
};

class Input_Layer: public Layer{
public:
    Input_Layer(unsigned int layer_size_);
};

#endif // LAYER_H
