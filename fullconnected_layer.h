#ifndef FULLCONNECTED_LAYER_H
#define FULLCONNECTED_LAYER_H
#include "layer.h"
#include "comm_funcs.h"

//Полносвязный слой. Каждый нейрон слоя соединен с каждым нейроном предыдущего слоя.
class FullConnected_Layer : public Layer{
public:
    FullConnected_Layer(unsigned int n, std::shared_ptr<Layer> prev, Activation_funcs Activation_function);
    void Calculate();
    void Back_Propagation(std::shared_ptr<Layer>& Next_layer);
    void Back_Propagation(const std::vector< double>& actual_values);
};



#endif // FULLCONNECTED_LAYER_H
