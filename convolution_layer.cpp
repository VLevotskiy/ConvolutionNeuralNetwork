#include "convolution_layer.h"

////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////Сверточный слой//////////////////////////////////////////////////////
Convolution_Layer::Convolution_Layer(std::shared_ptr<Layer> prev,Activation_funcs Activation_function,\
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
