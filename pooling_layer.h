#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H
#include "layer.h"
#include "comm_funcs.h"

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

#endif // POOLING_LAYER_H
