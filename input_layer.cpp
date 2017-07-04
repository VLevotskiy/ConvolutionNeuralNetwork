#include "input_layer.h"

Input_Layer::Input_Layer(unsigned int layer_size_) :Layer() {
    type = Input;
    layer_size = layer_size_;
    Create_neurons();
}

void Input_Layer::Fill_layer(std::vector< double>& data) {
    if (layer_size <= 0) throw std::runtime_error("Input_layer::Fill_layer. Layer is empty");
    if (layer_size < data.size()) throw std::runtime_error("Input_layer::Fill_layer. Input_vector large then layer_size");

    for (int i = 0; i < data.size();i++) {
        neurons[i].Set_value(data[i]);
    }
}

void Input_Layer::Calculate(){
    return;
}

