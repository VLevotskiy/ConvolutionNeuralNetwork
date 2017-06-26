#include "layer.h"
#include "comm_funcs.h"
#include <fstream>

Layer::Layer(unsigned int n, std::shared_ptr<Layer> prev, std::string& Activation_function, Layer_type type_){
    if (prev != 0)
        prev_layer = prev;
    else prev_layer = nullptr;
    layer_size = n;
    int act_fun = Parser(Activation_function,list_of_activation_funcs,NUM_OF_ACTIVATION_FUNCS);
    switch(act_fun) {
    case 0: activation_func = &SIGMOID; break;
    case 1: activation_func = &ReLU; break;
    case 2: activation_func = &SoftMax; break;
    case 3: activation_func = &Linear; break;
    }
    type = type_;
    if (type != Pooling){
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

std::vector<Neuron>& Layer::Get_neurons() {
    return neurons;
}

std::shared_ptr<Layer>  Layer::Get_Prev() const {
    return prev_layer;
}

Layer_type Layer::Get_type() const {
    return type;
}

void Layer::Update_weights(const double training_rate) {
    std::vector<Neuron> prev_neurons = prev_layer->Get_neurons();

    for (size_t i = 0; i < Size(); i++) {
        std::vector<Connection> neuron_connections = neurons[i].Get_connections();

        for (size_t j = 0; j < prev_layer->Size(); j++) {
            double dw = training_rate * neurons[i].Get_delta() * prev_neurons[j].Get_value();
            neuron_connections[j].Set_Last_dw(dw);
            double new_weight = neuron_connections[j].Get_Weight() + dw;
            neuron_connections[j].Set_weight(new_weight);
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////Полносвязный слой//////////////////////////////////////////////////////
// соединяем  нейроны слоя связями с случайными весовыми коэффициентами с нейронами предыдущего слоя
FullConnected_Layer::FullConnected_Layer(unsigned int n, std::shared_ptr<Layer> prev,std::string& Activation_function) : Layer(n,prev, Activation_function,FullConnected) {
    if (!prev){ throw std::runtime_error("FullConnected layer Null pointer for prev layer");}
    layer_size = layer_size + 1;
    neurons.push_back(Neuron());
    neurons.at(layer_size-1).Set_value(( double)1.0);
    for (int i =0; i  < layer_size; i++){
         double* weights = new  double[prev_layer->Size()];
        gen_array(0.0001, 0.2, prev_layer->Size(), weights);
        for (int j = 0; j < prev_layer->Size(); j++){
            neurons[i].Add_Connection(j,weights[j]);
        }
        delete [] weights;
    }
}

void FullConnected_Layer::Calculate(){
    if (!prev_layer) return;
    std::vector<Neuron> prevNeurons = prev_layer->Get_neurons();
    for (size_t i = 0; i < layer_size-1; i++) {
         double tmp = 0;
        std::vector<Connection> wgths_i = neurons[i].Get_connections();
        if (wgths_i.size() != prevNeurons.size()) {throw std::runtime_error("FullConnection_Layer::Calculate. wgths"); }
        for (size_t j = 0; j < prev_layer->Size(); j++) {
            tmp += wgths_i.at(j).Get_Weight() * prevNeurons.at(j).Get_value();
        }
         double out = activation_func(tmp);
        neurons.at(i).Set_value(out);
    }
}

//порядок слоёв в сети prev ->current->next
void FullConnected_Layer::Back_Propagation(std::shared_ptr<Layer>& Next_layer){
    std::vector<Neuron> next_neurons = Next_layer->Get_neurons();
    std::vector< double> sums(neurons.size());

    std::fill(sums.begin(),sums.end(),0);
    for (size_t j = 0; j < neurons.size(); j++) {
        for (size_t k = 0; k < next_neurons.size(); ++k){
            std::vector<Connection> current_next_con = next_neurons[k].Get_connections();
            sums[j] += current_next_con[j].Get_Weight() * next_neurons[k].Get_delta();
        }
        neurons[j].Set_delta(sums[j] * DSIGMIOD(neurons[j].Get_value()));
    }
}

void FullConnected_Layer::Back_Propagation(std::vector< double>& actual_values){
    if (actual_values.size() != neurons.size()) throw std::runtime_error("Size of actual values < size of last layer");
    for (size_t j = 0; j < neurons.size(); j++) {
        neurons[j].Set_delta( (actual_values[j] - neurons[j].Get_value()) * DSIGMIOD(neurons[j].Get_value()) );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////Сверточный слой//////////////////////////////////////////////////////
Convolution_Layer::Convolution_Layer(std::shared_ptr<Layer> prev,std::string& Activation_function,\
                                     uint16_t input_height,uint16_t input_width,\
                                     uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks) : Layer(input_width * input_height * num_of_masks,prev, Activation_function,Convolution) {

     double** weights = new  double*[num_of_masks];
    mask_size = input_width * input_height;
    //layer_size = mask_size * num_of_masks;
    for (int i = 0; i < num_of_masks;i++){
        weights[i] = new  double[mask_size];
        gen_array(0.0001, 0.2, mask_size, &weights[i][0]);
    }

    int start_w_offset = (int)(el_width * 0.5);
    int start_h_offset = (int)(el_height * 0.5);

    for (int i =0; i <  input_height; i++){           //cтроки
        int temp_h = i - start_h_offset;
        for(int j = 0; j  < input_width; j++) {       //столбцы
            int temp_w = j - start_w_offset;
            for (int m = 0; m <  el_height; m++){   //строки маски
                for (int n = 0; n < el_width; n++){  //столбцы маски
                    int h_pos = temp_h + m;
                    int w_pos = temp_w + n;
                    if (h_pos < 0) continue;//h_pos = m;
                    if (w_pos < 0) continue;//w_pos = n;
                    if (h_pos >= input_height)continue;
                    if (w_pos >= input_width) continue;

                    //out[i * input_height + j] += input[h_pos * input_height + w_pos] * mask[m * el_height + n];
                    for (int l = 0; l < num_of_masks; l++){
                        neurons[i * input_height + j + l*mask_size].Add_Connection(h_pos * input_height + w_pos , weights[l][m * el_height + n]);
                    }
                }
            }
        }
    }

    for (int i =0; i < num_of_masks; i++){
        delete [] weights[i];
    }
    delete [] weights;
}

void Convolution_Layer::Calculate(){

}

////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////Объединяющий слой/////////////////////////////////////////////////////
Pooling_Layer::Pooling_Layer(std::shared_ptr<Layer> prev,std::string& Activation_function, \
                             uint16_t input_width, uint16_t input_height, uint8_t step_size,\
                             uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks) : Layer(0,prev, Activation_function, Pooling) {
    unsigned int new_w = input_width/el_width, new_h = input_height/el_height;
    if (input_width % el_width != 0) new_w +=1;
    if (input_height % el_height != 0) new_h +=1;
    mask_size =  new_w * new_h;
    layer_size = mask_size *num_of_masks;

    Create_neurons();

    for (int mask_num = 0; mask_num < num_of_masks; mask_num++) {
        for (int i =0, out_i =0; i < input_height; i+=step_size,out_i++) {
            for (int j = 0, out_j = 0; j < input_width; j+=step_size,out_j++){
                for (int m = 0; m < el_height;m++){
                    for (int n = 0; n < el_width;n++) {
                        const int  pos = (i+m)*input_width + j + n;
                        neurons[out_i * new_h+out_j + mask_size*mask_num].Add_Connection(pos,1);
                    }
                }
            }
        }
    }
}

void Pooling_Layer::Calculate() {

}

Input_Layer::Input_Layer(unsigned int layer_size_) :Layer() {
    type = Input;
    layer_size = layer_size_;
    Create_neurons();
}

void Input_Layer::Fill_layer(std::vector< double>& data) {
    if (layer_size <= 0) throw std::runtime_error("Input_layer::Fill_layer. Layer is empty");
    if (layer_size < data.size()) throw std::runtime_error("Input_layer::Fill_layer. Input_vector large then layer_size");

    for (int i = 0; i < data.size();i++) {
        neurons[i].Set_value(data[i]);
    }
}

void Input_Layer::Calculate(){
    return;
}


