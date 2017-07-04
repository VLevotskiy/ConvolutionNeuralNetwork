#include "fullconnected_layer.h"

////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////Полносвязный слой//////////////////////////////////////////////////////
// соединяем  нейроны слоя связями с случайными весовыми коэффициентами с нейронами предыдущего слоя
FullConnected_Layer::FullConnected_Layer(unsigned int n, std::shared_ptr<Layer> prev, Activation_funcs Activation_function) : Layer(n,prev, Activation_function,FullConnected) {
    if (!prev){ throw std::runtime_error("FullConnected layer Null pointer for prev layer");}
    layer_size = layer_size+1;
    neurons.push_back(Neuron());
    neurons.at(layer_size-1).Set_value(( double)1.0);
    for (int i =0; i  < layer_size-1; i++){
         double* weights = new  double[prev_layer->Size()];
        gen_array(-0.2, 0.2, prev_layer->Size(), weights);
        for (int j = 0; j < prev_layer->Size(); j++){
            neurons[i].Add_Connection(j,weights[j]);
        }
        delete [] weights;
    }
}

void FullConnected_Layer::Calculate(){
    if (!prev_layer) return;
    auto prevNeurons = prev_layer->Get_neurons();
#pragma omp parallel for
    for (size_t i = 0; i < layer_size-1; i++) {
        double tmp = 0;
        auto wgths_i = neurons[i].Get_connections();
        if (wgths_i->size() != prevNeurons->size()) {throw std::runtime_error("FullConnection_Layer::Calculate. wgths"); }
        for (size_t j = 0; j < prev_layer->Size(); j++) {
            tmp += wgths_i->at(j).Get_weight() * prevNeurons->at(j).Get_value();
        }
        neurons[i].Set_value(tmp);
    }

    Activate_Layer();
}

//порядок слоёв в сети prev ->current->next
void FullConnected_Layer::Back_Propagation(std::shared_ptr<Layer>& Next_layer){
    auto next_neurons = Next_layer->Get_neurons();
    std::vector< double> sums(neurons.size());

    std::fill(sums.begin(),sums.end(),0);
#pragma omp parallel for
    for (size_t j = 0; j < neurons.size(); j++) {
        for (size_t k = 0; k < next_neurons->size()-1; ++k){
            auto current_next_con = next_neurons->at(k).Get_connections();
            sums[j] += current_next_con->at(j).Get_weight() * next_neurons->at(k).Get_delta();
        }
        neurons[j].Set_delta(sums[j] * D_activation_func(neurons[j].Get_value()));
    }
}

void FullConnected_Layer::Back_Propagation(const std::vector< double>& actual_values){
    if (actual_values.size() != layer_size-1) throw std::runtime_error("Size of actual values < size of last layer");
    for (size_t j = 0; j < layer_size; j++) {
        neurons[j].Set_delta( (actual_values[j] - neurons[j].Get_value()) * D_activation_func(neurons[j].Get_value()) );
    }
}

