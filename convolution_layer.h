#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H
#include "layer.h"
#include "comm_funcs.h"

//Сверточный слой. Выполнятеся свертка нейронов предыдущего слоя с заданным весами структурным элементом
class Convolution_Layer : public Layer {
private:
    uint16_t mask_size;
    uint16_t num_of_masks;
    std::vector<std::vector<Neuron>> sublayers;

    void convolution( double* input,const size_t input_width,const size_t input_height,  double* mask, size_t el_width, size_t el_height);
public:
    Convolution_Layer(std::shared_ptr<Layer> prev, Activation_funcs Activation_function, \
                      uint16_t input_height, uint16_t input_width, \
                      uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks);
    void Calculate();
    void Back_Propagation(std::shared_ptr<Layer>& Next_layer){}
    void Back_Propagation(const std::vector<double>& actual_values){}
};
#endif // CONVOLUTION_LAYER_H
