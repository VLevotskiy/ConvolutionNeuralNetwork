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
    double training_threshold;
public:
    NeuralNet(std::string* names, uint16_t num_of_layers, uint16_t input_layer_size);
    void forward_propagation(std::vector< double>&);
    bool back_propagation(const std::vector< double>& actual_values);
    double calculate_error(const std::vector< double>& actual_values);
    void training(const std::string& path);
    void update_weights();

};


#endif // NEURALNET_H
