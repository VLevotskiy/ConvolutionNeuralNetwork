#include "neuralnet.h"

// Layer_type = <Type> num_of_neurons = <n> activation_func = <func_name> ...
// Layer_type = Convolution num_of_neurons = <n> activation_func = <func_name> img_height = <n> img_width = <n> el_with = <n> el_height = <n> number_of_masks = <n>
// Layer_type = Pooling num_of_neurons = <n> activation_func = <func_name> el_with = <n> el_height = <n> number_of_masks = <n> step_size = <n>
NeuralNet::NeuralNet(std::string *str, uint16_t num_of_Layers,uint16_t input_layer_size) {
    layers.emplace_back( new Input_Layer(input_layer_size));
    const int NUM_OF_PARAMS = 9;
    const std::string parametrs_list[] {"Layer_type", "num_of_neurons", "activation_func", "el_width", "el_height", "number_of_masks", "step_size", "img_height", "img_width"};
    for (int j = 0; j < num_of_Layers; j++) {
        std::vector<std::string> words;
        Get_words(str[j],words,std::string(" "));
        for (auto i = words.begin(); i != words.end();++i){
            if ((*i)[0] == '='){
                words.erase(i);
            }
        }

        Layer_type LT;
        std::string Activ_func;
        uint16_t num_of_neurons = 0;
        uint8_t el_width = 0;
        uint8_t el_height = 0;
        uint8_t num_of_masks = 0;
        uint8_t step_size = 0;
        uint8_t img_height = 0;
        uint8_t img_width =0;
        for (int i = 0; i < words.size(); ++i){
            int8_t param = Parser(words.at(i),parametrs_list,NUM_OF_PARAMS);
            switch(param){
            case 0: {
                int8_t type = Parser(words[++i],list_of_layers_types,NUM_OF_LAYERS_TYPES);
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
                default: throw std::runtime_error("Unknown layer type " + words[i]); break;
                }
                break;
            }
            case 1: {
                try {
                    num_of_neurons = std::stoi(words[++i]);
                } catch(...) {
                    throw std::runtime_error("Wrong format of num_of_neurons " + words[i]);
                }
                break;
            }

            case 2: {
                Activ_func =words[++i];
                break;
            }

            case 3: {
                try {
                    el_width = std::stoi(words[++i]);
                } catch(...) {
                    throw std::string("Wrong format of el_width " + words[i]);
                }
                break;
            }
            case 4: {
                try{
                    el_height = std::stoi(words[++i]);
                } catch(...) {
                    throw std::string("Wrong format of el_height " + words[i]);
                }
                break;
            }
            case 5: {
                try{
                    num_of_masks = std::stoi(words[++i]);
                } catch(...) {
                    throw std::string("Wrong format of num_of_masks " + words[i]);
                }
                break;
            }
            case 6: {
                try{
                    step_size = std::stoi(words[++i]);
                } catch(...) {
                    throw std::string("Wrong format of step_size " + words[i]);
                }
                break;
            }
            case 7: {
                try{
                    img_height = std::stoi(words[++i]);
                } catch(...) {
                    throw std::string("Wrong format of img_height " + words[i]);
                }
                break;
            }
            case 8:{
                try{
                    img_width = std::stoi(words[++i]);
                } catch(...) {
                    throw std::string("Wrong format of img_width " + words[i]);
                }
                break;
            }
            default: throw std::string("Unknown parameter " + words[i]); break;
            }
        }
        auto prev = layers.at(j);

        switch((int)LT){
        case FullConnected:
            layers.emplace_back(new FullConnected_Layer(num_of_neurons,prev,Activ_func));
            break;
        case Convolution:
            layers.emplace_back(new Convolution_Layer(prev,Activ_func, img_width,img_height, el_width,el_height,num_of_masks));
            break;
        case Pooling:
            layers.emplace_back(new Pooling_Layer(prev,Activ_func, img_width,img_height, step_size,el_width,el_height,num_of_masks));
            break;
        }
    }
}

std::shared_ptr<Layer>* NeuralNet::forward_propagaition(const std::vector<float>& input_data) {
    std::static_pointer_cast<Input_Layer>(layers[0])->Fill_layer(input_data);
    for(uint8_t i = 1; i < layers.size();i++){
        layers[i]->Calculate();
    }
}

void NeuralNet::back_propagaition(const std::vector<float>&) {

}

