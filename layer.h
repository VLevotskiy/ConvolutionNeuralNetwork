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
    std::vector<Neuron> &Get_neurons();
    std::shared_ptr<Layer> Get_Prev() const;
    Layer_type Get_type() const;
    Activation_funcs Get_activation_func() const { return activation_func; }
    void Update_weights(const double training_rate,const double inert_coeff);
    void Activate_Layer();

    virtual void Calculate() = 0;
    virtual void Back_Propagation(std::shared_ptr<Layer>& Next_layer) =0;
    virtual void Back_Propagation(const std::vector<double>& actual_values) =0;
};

//Полносвязный слой. Каждый нейрон слоя соединен с каждым нейроном предыдущего слоя.
class FullConnected_Layer : public Layer{
public:
    FullConnected_Layer(unsigned int n, std::shared_ptr<Layer> prev, Activation_funcs Activation_function);
    void Calculate();
    void Back_Propagation(std::shared_ptr<Layer>& Next_layer);
    void Back_Propagation(const std::vector< double>& actual_values);
};


//Сверточный слой. Выполнятеся свертка нейронов предыдущего слоя с заданным весами структурным элементом
class Convolution_Layer : public Layer {
private:
    uint16_t height;
    uint16_t width;
    uint16_t mask_size;

    void convolution( double* input,const size_t input_width,const size_t input_height,  double* mask, size_t el_width, size_t el_height);
public:
    Convolution_Layer(std::shared_ptr<Layer> prev, Activation_funcs Activation_function, \
                      uint16_t input_height, uint16_t input_width, \
                      uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks);
    void Calculate();
    void Back_Propagation(std::shared_ptr<Layer>& Next_layer){}
    void Back_Propagation(const std::vector<double>& actual_values){}
};

//слой выбирает наибольшее значение нейрона среди нейронов, попавших в структурный элмент определенного размера.
class Pooling_Layer : public Layer {
    unsigned char num_of_masks;
    uint16_t mask_size;
    uint16_t structural_width;
    uint16_t structural_height;
    uint8_t step_size;

    void pooling( double* input, const size_t width,const size_t height,const unsigned char el_size, const unsigned char step );
public:
    Pooling_Layer(std::shared_ptr<Layer> prev,Activation_funcs Activation_function, uint16_t input_width, uint16_t input_height, uint8_t step_size, uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks);
    void Calculate();
    void Back_Propagation(std::shared_ptr<Layer>& Next_layer){}
    void Back_Propagation(const std::vector<double>& actual_values){}
};

//Входной слой. Хранит данные поданые на вход. Не выполняет никаких вычислений.
class Input_Layer: public Layer{
public:
    Input_Layer(unsigned int layer_size_);
    void Calculate();
    void Back_Propagation(std::shared_ptr<Layer>& Next_layer){}
    void Back_Propagation(const std::vector<double>& actual_values){}
    void Fill_layer(std::vector< double>& data);
};

#endif // LAYER_H
