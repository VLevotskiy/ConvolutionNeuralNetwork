#ifndef LAYER_H
#define LAYER_H
#include "neuron.h"
#include <vector>
#include <stdint.h>

enum Layer_type{FullConnected=1,Convolution,Pooling};
const int NUM_OF_LAYERS_TYPES = 3;
const std::string list_of_layers_types[3] = {"FullConnected", "Convolution", "Pooling"};

//Класс слоя. Хранит информацию о слое сети
class Layer {
private:
    std::vector<Neuron> neurons;
    unsigned int layer_size;
    Layer* prev_layer;
    Layer_type type;
public:
    Layer(int n, Layer* prev,Layer_type t);
    //void Set_Layer_Type(Layer_type t, unsigned int step_size = 0, unsigned int el_width = 0, unsigned int el_height = 0, unsigned char num_of_masks = 0, size_t input_height = 0, size_t input_width = 0);
    unsigned int Size() const;// { return layer_size;}
    std::vector<Neuron>& Get_neurons() const;
    Layer* Get_Prev() const;
    Layer_type Get_type() const;
    virtual void Calculate = 0;
};

class FullConnected_Layer : public Layer{
public:
    FullConnected_Layer();
    void Calculate();
};

class Convolution_Layer : public Layer {
private:
    unsigned char num_of_masks;
    uint16_t structural_width;
    uint16_t structural_height;
    uint8_t step_size;

    void convolution(float* input,const size_t input_width,const size_t input_height, float* mask, size_t el_width, size_t el_height);
public:
    Convolution_Layer();
    void Calculate();
};

class Pooling_Layer : public Layer {
    unsigned char num_of_masks;
    uint16_t structural_width;
    uint16_t structural_height;
    uint8_t step_size;

    void pooling(float* input, const size_t width,const size_t height,const unsigned char el_size, const unsigned char step );
public:
    Pooling_Layer();
    void Calculate();
};

#endif // LAYER_H
