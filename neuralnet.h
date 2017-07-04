#ifndef NEURALNET_H
#define NEURALNET_H
#include <stdint.h>
#include <vector>
#include <string>
#include <memory>
#include "comm_funcs.h"
#include "layer.h"
#include "fullconnected_layer.h"
#include "convolution_layer.h"
#include "pooling_layer.h"
#include "input_layer.h"

class NeuralNet {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    unsigned int num_of_layers;

    double training_speed;
    double inertia_coeff;
    double training_threshold;
    void create_net(std::string* layers_discription, uint16_t num_of_layers, double training_threshold_ = 0.0001, double training_speed_ = 0.1, double inertia_coeff_ = 0.05);
public:
    NeuralNet(std::string* layers_discription, uint16_t num_of_layers, double training_threshold_ = 0.0001, double training_speed_ = 1, double inertia_coeff_ = 0.05);
    NeuralNet(const std::string& nn_description_path, const std::string& nn_weights_path);
    void forward_propagation(std::vector< double>&);
    bool back_propagation(const std::vector< double>& actual_values);
    double calculate_error(const std::vector< double>& actual_values);
    std::shared_ptr<std::vector<double> > Get_last_layer();
    void update_weights();
    void save_net(const std::string& nn_description_path, const std::string& nn_weights_path);
    void load_weights(const std::string& path);


};


#endif // NEURALNET_H
