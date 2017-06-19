#include "neuron.h"
#include "random_num.h"

Layer::Layer(int n, Layer* prev, Layer_type t, unsigned int step_size, unsigned int el_width, unsigned int el_height, unsigned char num_of_masks, size_t input_height, size_t input_width){
    prev_layer = prev;
    layer_size = n;

    for(int i = 0; i  < n; i++) {
        neurons.push_back(Neuron());
    }
    Set_Layer_Type(t,step_size, el_width, el_height, num_of_masks, input_height, input_width);
}

void Layer::Set_Layer_Type(Layer_type t, unsigned int step_size, unsigned int el_width, unsigned int el_height, unsigned char num_of_masks, size_t input_height, size_t input_width){
    if (input_height * input_width != prev_layer->Size()) {
        std::cerr << "Error! Wrong layer size!";
    }
    type = t;
    switch(type){
    case FullConnected: {
        for (int i =0; i  < layer_size; i++){
            float* weights = new float[prev_layer->Size()];
            gen_array_t(0.0001, 0.2, prev_layer->Size(), weights);
            for (int j = 0; j < prev_layer->Size(); j++){
                neurons[i].Add_Connection(j,weights[j]);
            }
            delete [] weights;
        }
        break;
    }
    case Convolution: {
        float** weights = new float*[num_of_masks];

        for (int i = 0; i < num_of_masks;i++){
            weights[i] = new float[el_width * el_height];
            gen_array_t(0.0001, 0.2, el_width * el_height, &weights[i][0]);
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
                            neurons[i * input_height + j + l*input_height*input_width].Add_Connection(h_pos * input_height + w_pos , weights[l][m * el_height + n]);
                        }
                    }
                }
            }
        }

        for (int i =0; i < num_of_masks; i++){
            delete [] weights[i];
        }
        delete [] weights;
        break;
    }
    case Pooling: {
        unsigned int new_w = input_width/el_width, new_h = input_width/el_height;
        if (input_width % el_width != 0) new_w +=1;
        if (input_width % el_height != 0) new_h +=1;
        for (int i =0, out_i =0; i < height; i+=step_size,out_i++) {
            for (int j = 0, out_j = 0; j < width; j+=step_size,out_j++){
                for (int m = 0; m < el_size;m++){
                    for (int n = 0; n < el_size;n++) {
                        const int  pos = (i+m)*width + j + n;
                        neurons[out_i * new_h+out_j].Add_Connection(pos,1);
                    }
                }
            }
        }
        break;
    }
    default: std::cerr << "Unknown type of layer!" << std::endl;break;
    }
}
