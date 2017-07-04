#include "pooling_layer.h"

////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////Объединяющий слой/////////////////////////////////////////////////////
Pooling_Layer::Pooling_Layer(std::shared_ptr<Layer> prev, Activation_funcs Activation_function, \
                             uint16_t input_width, uint16_t input_height, uint8_t step_size, \
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
