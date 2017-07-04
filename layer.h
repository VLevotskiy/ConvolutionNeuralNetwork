#ifndef LAYER_H
#define LAYER_H
#include "neuron.h"
#include <vector>
#include <string>
#include <stdint.h>
#include <math.h>
#include <memory>


enum Layer_type{Input=0, FullConnected=1, Convolution=2, Pooling=3};
enum Activation_funcs {Sigmoid=0, ReLU, SoftMax, Linear};
const int NUM_OF_LAYERS_TYPES = 4;
const int NUM_OF_ACTIVATION_FUNCS = 4;
const std::string list_of_layers_types[NUM_OF_LAYERS_TYPES] = {"FullConnected", "Convolution", "Pooling", "Input"};
const std::string list_of_activation_funcs[NUM_OF_ACTIVATION_FUNCS] = {"Sigmoid", "ReLU", "SoftMax", "Linear"};

//Класс слоя. Хранит информацию о слое сети
class Layer {
protected:
    std::vector<Neuron> neurons;
    unsigned int layer_size;
    std::shared_ptr<Layer>  prev_layer;
    Layer_type type;
    Activation_funcs activation_func;
    double (*D_activation_func)( double);

    void Create_neurons();
public:
    Layer() {
        neurons.clear();
        layer_size =0;
        activation_func = Linear;
    }
    Layer(unsigned int n, std::shared_ptr<Layer> prev, Activation_funcs Activation_function, Layer_type type_);

    unsigned int Size() const;
    //std::vector<Neuron> &Get_neurons();
    std::shared_ptr<std::vector<Neuron>> Get_neurons();
    std::shared_ptr<Layer> Get_Prev() const;
    Layer_type Get_type() const;
    Activation_funcs Get_activation_func() const { return activation_func; }
    void Update_weights(const double training_rate,const double inert_coeff);
    void Activate_Layer();
    std::vector<Neuron>* Get_neurons_p() { return &neurons;}

    virtual void Calculate() = 0;
    virtual void Back_Propagation(std::shared_ptr<Layer>& Next_layer) =0;
    virtual void Back_Propagation(const std::vector<double>& actual_values) =0;
};

#endif // LAYER_H
