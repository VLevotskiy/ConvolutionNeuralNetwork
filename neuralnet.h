#ifndef NEURALNET_H
#define NEURALNET_H
#include <stdint.h>
#include <vector>
#include <string>
#include <memory>
#include "common_functions.cpp"
#include "layer.h"

class NeuralNet {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    unsigned int num_of_layers;

    double training_speed;
    double error_coef;
public:
    NeuralNet(uint8_t num_of_layers_,std::string& types,std::vector<uint16_t>& layers_sizes);
    NeuralNet(std::string* names,uint16_t num_of_layers);
    Layer* forward_propognition(std::vector<float>&);
    void back_propognition(std::vector<float>&);
    void training(std::string& path);

};


#endif // NEURALNET_H
