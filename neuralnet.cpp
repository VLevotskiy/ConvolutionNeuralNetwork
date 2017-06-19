#include "neuralnet.h"

NeuralNet::NeuralNet(uint8_t num_of_layers_,std::string& types,std::vector<uint16_t>& layers_sizes):num_of_layers(num_of_layers_){
    Layer* prev = nullptr;
    Layer_type LT = 0;
    int n;
    for (int i =0; i < num_of_layers_; i++){
        n = layers_sizes.at(i);
        int type = Parser(types[i],list_of_layers_types,NUM_OF_LAYERS_TYPES);
        switch(type){
        case 0:{
            LT = FullConnected;
            break;
        }
        case 1: {
            LT = Convolution;
            break;
        }
        case 2: {
            LT = Pooling;
            break;
        }
        }
        layers.emplace_back(n,prev,LT);
        prev = &layers.at(i);
    }
}
