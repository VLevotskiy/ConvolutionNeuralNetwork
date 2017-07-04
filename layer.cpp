#include "layer.h"
#include "comm_funcs.h"
#include <fstream>

Layer::Layer(unsigned int n, std::shared_ptr<Layer> prev, Activation_funcs Activation_function, Layer_type type_){
    if (prev != 0)
        prev_layer = prev;
    else prev_layer = nullptr;
    layer_size = n;
    activation_func =Activation_function;
    //Activation_funcs act_fun = (Activation_funcs)Parser(Activation_function,list_of_activation_funcs,NUM_OF_ACTIVATION_FUNCS);
    switch(Activation_function) {
    case Sigmoid: D_activation_func = &DSIGMOID; break;
    case ReLU: D_activation_func = &DReLU; break;
    case SoftMax: D_activation_func = &DSoftMax; break;
    case Linear: D_activation_func = &DLinear; break;
    }

    type = type_;
    if (type != Pooling && type != Input){
        Create_neurons();
    }

}

void Layer::Create_neurons() {
    if (layer_size <=0) {throw std::runtime_error("Trying to create layer with size 0\n");}
    for(int i = 0; i  < layer_size; i++) {
        neurons.emplace_back(0);
    }
}

unsigned int Layer::Size() const {
    return layer_size;
}

std::shared_ptr<std::vector<Neuron>> Layer::Get_neurons() {
    return std::make_shared<std::vector<Neuron>>(neurons);
}

std::shared_ptr<Layer>  Layer::Get_Prev() const {
    return prev_layer;
}

Layer_type Layer::Get_type() const {
    return type;
}

void Layer::Update_weights(const double training_rate,const double inertia_coeff) {
    if (prev_layer == nullptr) return;
    auto prev_neurons = prev_layer->Get_neurons();

    for (size_t i = 0; i < Size()-1; i++) {
        auto neuron_connections = neurons[i].Get_connections();

        for (size_t j = 0; j < prev_layer->Size(); j++) {
            double dw = (1 - inertia_coeff) * training_rate * neurons[i].Get_delta() * prev_neurons->at(j).Get_value() + inertia_coeff * neuron_connections->at(j).Get_Last_dw();
            //neuron_connections[j].Set_Last_dw(dw);
            double new_weight = neuron_connections->at(j).Get_weight() + dw;
            //neuron_connections[j].Set_weight(new_weight);
            neurons[i].Update_weight(j, new_weight,dw);
        }
    }

}

void Layer::Activate_Layer() {
    int layer_size_ = layer_size;
    //if (type == FullConnected) {
    //    layer_size_ -= 1;
    //}
    switch (activation_func){
    case Sigmoid: {
#pragma omp parallel for
        for (int i = 0; i < layer_size_; i++){
            neurons[i].Set_value(SIGMOID_f(neurons[i].Get_value()));
        }
        break;
    }
    case ReLU: {
        #pragma omp parallel for
        for (int i = 0; i < layer_size_; i++){
            neurons[i].Set_value(ReLU_f(neurons[i].Get_value()));
        }
        break;
    }
    case SoftMax:{
        double sum = 0;
        std::vector<double> exp_values;
        #pragma omp parallel for
        for (int i = 0; i < layer_size_; i++){
            exp_values.push_back( exp(neurons[i].Get_value()) );
            sum +=exp_values[i];
        }

        if (sum==0) throw std::runtime_error("Softmax. Exponentioal sums = 0");
        #pragma omp parallel for
        for (int i = 0; i < layer_size_; i++){
            neurons[i].Set_value( exp_values[i]/sum);
        }
        break;
    }
    case Linear: return;
    }
}

