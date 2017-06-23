#ifndef NEURALNET_H
#define NEURALNET_H
#include <stdint.h>
#include <vector>
#include <string>
#include <memory>
#include "comm_funcs.h"
#include "layer.h"

class NeuralNet {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    unsigned int num_of_layers;

    double training_speed;
    double error_coef;
public:
    NeuralNet(std::string* names, uint16_t num_of_layers, uint16_t input_layer_size);
    std::shared_ptr<Layer>* forward_propagaition(const std::vector<float>&);
    void back_propagaition(const std::vector<float>&);
    void calculate_error(const std::vector<float>&);
    void training(const std::string& path);

};


#endif // NEURALNET_H
