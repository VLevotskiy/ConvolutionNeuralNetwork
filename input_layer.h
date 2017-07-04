#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H
#include "layer.h"

//Входной слой. Хранит данные поданые на вход. Не выполняет никаких вычислений.
class Input_Layer: public Layer{
public:
    Input_Layer(unsigned int layer_size_);
    void Calculate();
    void Back_Propagation(std::shared_ptr<Layer>& Next_layer){}
    void Back_Propagation(const std::vector<double>& actual_values){}
    void Fill_layer(std::vector< double>& data);
};

#endif // INPUT_LAYER_H
