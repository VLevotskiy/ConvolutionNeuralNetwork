#include "neuralnet.h"
#include <iostream>
#include <fstream>
// Layer_type = <Type> num_of_neurons = <n> activation_func = <func_name> ...
// Layer_type = Convolution num_of_neurons = <n> activation_func = <func_name> img_height = <n> img_width = <n> el_with = <n> el_height = <n> number_of_masks = <n>
// Layer_type = Pooling num_of_neurons = <n> activation_func = <func_name> el_with = <n> el_height = <n> number_of_masks = <n> step_size = <n>
NeuralNet::NeuralNet(std::string *str, uint16_t num_of_Layers,uint16_t input_layer_size,double training_threshold_, double training_speed_,double inertia_coeff_) {
    const int NUM_OF_PARAMS = 9;
    const std::string parametrs_list[] {"Layer_type", "num_of_neurons", "activation_func", "el_width", "el_height", "number_of_masks", "step_size", "img_height", "img_width"};
    inertia_coeff = inertia_coeff_;
    training_speed = training_speed_;
    training_threshold = training_threshold_;
    num_of_layers = num_of_Layers+1;

    layers.emplace_back( std::make_shared<Input_Layer>(input_layer_size));//new Input_Layer(input_layer_size));

    for (int j = 0; j < num_of_Layers; j++) {
        std::vector<std::string> words;
        Get_words(str[j],words,std::string(" "));
        for (auto i = words.begin(); i != words.end();++i){
            if ((*i)[0] == '='){
                words.erase(i);
            }
        }

        Layer_type LT = Input;
        Activation_funcs Activ_func;
        uint16_t num_of_neurons = 0;
        uint8_t el_width = 0;
        uint8_t el_height = 0;
        uint8_t num_of_masks = 0;
        uint8_t step_size = 0;
        uint8_t img_height = 0;
        uint8_t img_width =0;
        for (size_t i = 0; i < words.size(); ++i){
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
                int act_fun = Parser(words[++i],list_of_activation_funcs,NUM_OF_ACTIVATION_FUNCS);
                switch(act_fun) {
                case 0: Activ_func = Sigmoid; break;
                case 1: Activ_func = ReLU; break;
                case 2: Activ_func = SoftMax; break;
                case 3: Activ_func = Linear; break;
                }

                break;
            }

            case 3: {
                try {
                    el_width = std::stoi(words[++i]);
                } catch(...) {
                    throw std::runtime_error("Wrong format of el_width " + words[i]);
                }
                break;
            }
            case 4: {
                try{
                    el_height = std::stoi(words[++i]);
                } catch(...) {
                    throw std::runtime_error("Wrong format of el_height " + words[i]);
                }
                break;
            }
            case 5: {
                try{
                    num_of_masks = std::stoi(words[++i]);
                } catch(...) {
                    throw std::runtime_error("Wrong format of num_of_masks " + words[i]);
                }
                break;
            }
            case 6: {
                try{
                    step_size = std::stoi(words[++i]);
                } catch(...) {
                    throw std::runtime_error("Wrong format of step_size " + words[i]);
                }
                break;
            }
            case 7: {
                try{
                    img_height = std::stoi(words[++i]);
                } catch(...) {
                    throw std::runtime_error("Wrong format of img_height " + words[i]);
                }
                break;
            }
            case 8:{
                try{
                    img_width = std::stoi(words[++i]);
                } catch(...) {
                    throw std::runtime_error("Wrong format of img_width " + words[i]);
                }
                break;
            }
            default: throw std::runtime_error("Unknown parameter " + words[i]); break;
            }
        }
        auto prev = layers.at(j);

        switch(LT){
        case FullConnected:
            //if (j != num_of_Layers -1){ num_of_neurons++;}  //добавляем нейрон связи для скрытых слоев
            layers.emplace_back(std::make_shared<FullConnected_Layer>(num_of_neurons, prev, Activ_func));//new FullConnected_Layer(num_of_neurons,prev,Activ_func));
            break;
        case Convolution:
            layers.emplace_back(std::make_shared<Convolution_Layer>(prev,Activ_func, img_width,img_height, el_width,el_height,num_of_masks));
            break;
        case Pooling:
            layers.emplace_back(std::make_shared<Pooling_Layer>(prev,Activ_func,img_width,img_height,step_size,el_width,el_height,num_of_masks));//new Pooling_Layer(prev,Activ_func, img_width,img_height, step_size,el_width,el_height,num_of_masks));
            break;
        case Input: break;
        default: throw std::runtime_error("Unknow layer type"); break;
        }
    }
}

void NeuralNet::forward_propagation(std::vector< double>& input_data) {
    std::static_pointer_cast<Input_Layer>(layers[0])->Fill_layer(input_data);
    for(uint8_t i = 1; i < layers.size();i++){
        layers[i]->Calculate();
    }
}

bool NeuralNet::back_propagation(const std::vector< double>& actual_values) {
    //Рассчет ошибки
    double Error = calculate_error(actual_values);
    std::cout << Error << std::endl;
    if (Error < training_threshold) {
        return true;
    }

    //Рассчет дельт послденго слоя
    layers[num_of_layers -1]->Back_Propagation(actual_values);

    //Рассчет дельт скртытых слоёв
    for (int i = num_of_layers-2; i >= 0; i--) {
        layers[i]->Back_Propagation(layers[i+1]);
    }

    //Изменение весов
    update_weights();

    return false;
}

double NeuralNet::calculate_error(const std::vector< double>& actual_values) {
    if (actual_values.size() != layers[num_of_layers-1]->Size()-1) throw std::runtime_error("Size of actual values < size of last layer");
    double sum = 0;
    std::vector<Neuron> last_layer_neurons = layers[num_of_layers-1]->Get_neurons();
    for (size_t i = 0; i <actual_values.size(); i++) {
        double temp = last_layer_neurons[i].Get_value() - actual_values[i];
        sum += temp * temp;
    }
    return 0.5 * sum;

}

void NeuralNet::update_weights(){
     for (int i = num_of_layers-1; i >= 0; --i) {
         layers[i]->Update_weights(training_speed, inertia_coeff);
     }
 }

void NeuralNet::Get_last_layer() {
    std::vector<Neuron> neuron = layers[num_of_layers-1]->Get_neurons();
    for (size_t i = 0; i <  layers[num_of_layers-1]->Size()-1; i++) {
        std::cout << i << "\t" << neuron[i].Get_value() << std::endl;
    }


}
