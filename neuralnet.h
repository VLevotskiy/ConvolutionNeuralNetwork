#ifndef NEURALNET_H
#define NEURALNET_H
#include <stdint.h>
#include <vector>
#include <string>
#include "layer.h"

class NeuralNet {
private:
    std::vector<Layer> layers;
    unsigned int num_of_layers;

    double training_speed;
    double error_coef;

    int8_t Parser(std::string& input,const std::string* possible_values_list, uint8_t num_of_possible);
public:
    NeuralNet(uint8_t num_of_layers_,std::string& types,std::vector<uint16_t>& layers_sizes);
    Layer* forward_propognition(std::vector<float>&);
    void back_propognition(std::vector<float>&);
    void training(std::string& path);

};

#endif // NEURON_H*/



#endif // NEURALNET_H
