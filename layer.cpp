#include "layer.h"
#include "comm_funcs.h"

Layer::Layer(unsigned int n, std::shared_ptr<Layer>& prev, std::string& Activation_function){
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

    for(int i = 0; i  < n; i++) {
        neurons.emplace_back(0);
    }
}

unsigned int Layer::Size() const {
    return layer_size;
}

std::vector<Neuron>* Layer::Get_neurons() {
    return &neurons;
}

std::shared_ptr<Layer>  Layer::Get_Prev() const {
    return prev_layer;
}

Layer_type Layer::Get_type() const {
    return type;
}

////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////Полносвязный слой//////////////////////////////////////////////////////
// соединяем  нейроны слоя связями с случайными весовыми коэффициентами с нейронами предыдущего слоя
FullConnected_Layer::FullConnected_Layer(unsigned int n, std::shared_ptr<Layer> prev,std::string& Activation_function) : Layer(n,prev, Activation_function) {
    if (!prev){ throw std::runtime_error("FullConnected layer Null pointer for prev layer");}
    layer_size = layer_size + 1;
    neurons.push_back(Neuron());
    neurons.at(layer_size-1).Set_value((float)1.0);
    for (int i =0; i  < layer_size; i++){
        float* weights = new float[prev_layer->Size()];
        gen_array(0.0001, 0.2, prev_layer->Size(), weights);
        for (int j = 0; j < prev_layer->Size(); j++){
            neurons[i].Add_Connection(j,weights[j]);
        }
        delete [] weights;
    }
}

void FullConnected_Layer::Calculate(){
    /*if (prev_layer == nullptr) return;
    for (size_t i = 0; i < layer_size-1; i++) {
        float tmp = 0;
        float* wgths_i = neurons[i].Weights();
        const Neuron* prevNeurons = prev_layer->GetNeurons();
        for (size_t j = 0; j < prev_layer->Size(); j++) {
            tmp += wgths_i[j] * prevNeurons[j].Out_value();
        }
        float out = activation_func(tmp);
        neurons[i].SetValue(out);
    }*/
}

////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////Сверточный слой//////////////////////////////////////////////////////
Convolution_Layer::Convolution_Layer(std::shared_ptr<Layer> prev,std::string& Activation_function,\
                                     uint16_t input_height,uint16_t input_width,\
                                     uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks) : Layer(0,prev, Activation_function) {
    float** weights = new float*[num_of_masks];
    const uint16_t mask_size = input_width * input_height;
    layer_size = mask_size * num_of_masks;
    for (int i = 0; i < num_of_masks;i++){
        weights[i] = new float[mask_size];
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
                             uint8_t el_width, uint8_t  el_height, uint8_t num_of_masks) : Layer(0,prev, Activation_function) {
    unsigned int new_w = input_width/el_width, new_h = input_height/el_height;
    if (input_width % el_width != 0) new_w +=1;
    if (input_height % el_height != 0) new_h +=1;

    layer_size = new_w * new_h *num_of_masks;

    for (int i =0, out_i =0; i < input_height; i+=step_size,out_i++) {
        for (int j = 0, out_j = 0; j < input_width; j+=step_size,out_j++){
            for (int m = 0; m < el_height;m++){
                for (int n = 0; n < el_width;n++) {
                    const int  pos = (i+m)*input_width + j + n;
                    neurons[out_i * new_h+out_j].Add_Connection(pos,1);
                }
            }
        }
    }
}

void Pooling_Layer::Calculate() {

}

Input_Layer::Input_Layer(unsigned int layer_size_) :Layer() {
    layer_size = layer_size_;
    for(int i = 0; i  < layer_size; i++) {
        neurons.emplace_back(0);
    }
    type = Input;
}
