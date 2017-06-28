#include <QCoreApplication>
#include "neuralnet.h"
#include <iostream>
#include <fstream>
#include <chrono>
std::string layers[]={"Layer_type = FullConnected num_of_neurons = 275 activation_func = Sigmoid",
                        "Layer_type = FullConnected num_of_neurons = 21 activation_func = SoftMax"};
                   //"Layer_type = Convolution activation_func = ReLU img_height = 29 img_width = 29 el_width = 3 el_height = 3 number_of_masks = 6",
                   //"Layer_type = Pooling activation_func = Linear img_height = 29 img_width = 29 el_width = 2 el_height = 2 number_of_masks = 6",
                   //"Layer_type = FullConnected num_of_neurons = 1 activation_func = Sigmoid"};

void training(const std::string& path, NeuralNet& nn);

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    NeuralNet new_net(layers, 2,273);

    double mat[4][3] = {{0.0, 0.0, 0.0},
                      {1.0, 0.0, 1.0},
                      {0.0, 1.0, 1.0},
                      {1.0, 1.0, 1.0}};

    training("C:/Users/coder/Documents/build-ConvolutionNeuralNetwork-Desktop_Qt_5_5_1_MinGW_32bit-Release/release/",new_net);

    return a.exec();
}

void training(const std::string& path, NeuralNet& nn){
    std::vector<double> actual_value;
    for (size_t i = 0; i < 21; i++){
        actual_value.push_back(0);
    }

    //for (int k =0; k < 10; k++){
        int i = 1;
        for (;i<200;i++){
            for (int j =0;j < 21; j++){

                std::string new_path =path + "13_21/";
                //std::string new_path =path + "7_13/";
                switch(j){
                case 0:{new_path.append("0/"); break;}
                case 1:{new_path.append("1/"); break;}
                case 2:{new_path.append("2/"); break;}
                case 3:{new_path.append("3/"); break;}
                case 4:{new_path.append("4/"); break;}
                case 5:{new_path.append("5/"); break;}
                case 6:{new_path.append("6/"); break;}
                case 7:{new_path.append("7/"); break;}
                case 8:{new_path.append("8/"); break;}
                case 9:{new_path.append("9/"); break;}
                case 10:{new_path.append("A/"); break;}
                case 11:{new_path.append("B/"); break;}
                case 12:{new_path.append("C/"); break;}
                case 13:{new_path.append("E/"); break;}
                case 14:{new_path.append("H/"); break;}
                case 15:{new_path.append("K/"); break;}
                case 16:{new_path.append("M/"); break;}
                case 17:{new_path.append("P/"); break;}
                case 18:{new_path.append("T/"); break;}
                case 19:{new_path.append("X/"); break;}
                case 20:{new_path.append("Y/"); break;}
                default:{continue;break;}
                }
                new_path += std::to_string(i)+".bmp";
                //std::cout << new_path << std::endl;
                std::vector<double> input_vector;
                std::ifstream input(new_path, std::ios::binary);
                if (!input.is_open()) throw std::runtime_error("Can't open file " + new_path);
                input.seekg(1078);

                char y;
                int symbls_cntr = 0;
                int img_width = 13;
                int padding = img_width +(img_width*3)%4;
                while (!input.eof())
                {
                    input.get(y);
                    symbls_cntr++;
                    if (symbls_cntr < img_width){
                        input_vector.push_back(y /255.);
                    }
                    if (symbls_cntr == padding-1) symbls_cntr = 0;
                }
                input.close();
                auto begin = std::chrono::high_resolution_clock::now();

                nn.forward_propagation(input_vector);

                auto end = std::chrono::high_resolution_clock::now();
                std::cout<< std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()<<"ms"<< std::endl;

                nn.Get_last_layer();

                actual_value[j] = 1;
                if (i < 180){
                    nn.back_propagation(actual_value);
                }
                else {
                    std::cout << "symbol " << j << std::endl;
                    nn.Get_last_layer();
                }
                actual_value[j] = 0;

            }
        }
    //}
}

void training2(){
    std::vector<double> actual_value;
    for (size_t i = 0; i < 21; i++){
        actual_value.push_back(0);
    }

    uint16_t random[501];
    bool randomed[501];

    //for (size_t k =0; k < epochs; k++){
        memset(&randomed,0,501*sizeof(bool));
        memset(&random,0,501*sizeof(uint16_t));
        for (int r = 0; r < 500; r++){
            bool next = false;

            while (!next){
                int newr = rand()%500;
                if (randomed[newr] == false){
                    random[r] = newr;
                    randomed[newr] = true;
                    next = true;
                }
            }
        }


        int i = 1;
        for (;i<500;i++){
            for (int j =0;j < 21; j++){

                std::string new_path =path + "13_21/";
                //std::string new_path =path + "7_13/";
                switch(j){
                case 0:{new_path.append("0/"); break;}
                case 1:{new_path.append("1/"); break;}
                case 2:{new_path.append("2/"); break;}
                case 3:{new_path.append("3/"); break;}
                case 4:{new_path.append("4/"); break;}
                case 5:{new_path.append("5/"); break;}
                case 6:{new_path.append("6/"); break;}
                case 7:{new_path.append("7/"); break;}
                case 8:{new_path.append("8/"); break;}
                case 9:{new_path.append("9/"); break;}
                case 10:{new_path.append("A/"); break;}
                case 11:{new_path.append("B/"); break;}
                case 12:{new_path.append("C/"); break;}
                case 13:{new_path.append("E/"); break;}
                case 14:{new_path.append("H/"); break;}
                case 15:{new_path.append("K/"); break;}
                case 16:{new_path.append("M/"); break;}
                case 17:{new_path.append("P/"); break;}
                case 18:{new_path.append("T/"); break;}
                case 19:{new_path.append("X/"); break;}
                case 20:{new_path.append("Y/"); break;}
                default:{continue;break;}
                }
                new_path += std::to_string(random[i])+".bmp";
                //std::cout << new_path << std::endl;
                std::vector<double> input_vector;
                std::ifstream input(new_path, std::ios::binary);
                if (!input.is_open()) throw std::runtime_error("Can't open file " + new_path);
                input.seekg(1078);

                char y;
                int symbls_cntr = 0;
                int img_width = 13;
                int padding = img_width +(img_width*3)%4;
                while (!input.eof())
                {
                    input.get(y);
                    symbls_cntr++;
                    if (symbls_cntr < img_width){
                        input_vector.push_back(y /255.);
                    }
                    if (symbls_cntr == padding-1) symbls_cntr = 0;
                }
                input.close();
                auto begin = std::chrono::high_resolution_clock::now();

                nn.forward_propagation(input_vector);

                auto end = std::chrono::high_resolution_clock::now();
                std::cout<< std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()<<"ms"<< std::endl;

                nn.Get_last_layer();

                actual_value[j] = 1;
               // if (i < 180){
                    nn.back_propagation(actual_value);
                /*}
                else {
                    std::cout << "symbol " << j << std::endl;
                    nn.Get_last_layer();
                }
                actual_value[j] = 0;*/

            }
        }

    /*int errors_cntr = 0;
    for (int i = 0;i < 50;i++){
        for(int j =0; j < 21; j++){
        //int l = rand()%250;
        int num = i;
        float inp[input_size];
        for (int m = 0; m < height;m++){
            for (size_t n = 0; n < width; n++){
                inp [n+m*width]= ((unsigned char)array[j][random[i]][(m*(width+3))+n+1078])/255.;
            }
        }

        out = xor_net->forward_propagation(inp,input_size);
        out_neuron = out->GetNeurons();
        float max = 0.0;
        unsigned int num_max = 0;
        for (int j =0; j < out->Size()-1;j++){
            if (out_neuron[j].Out_value() > max) {num_max = j; max =out_neuron[j].Out_value();}
        }
        if (j!=num_max) errors_cntr++;
            std::cout << "num " << j << " result: " << num_max << " = " << out_neuron[num_max].Out_value() << std::endl;
        }
    }*/

}
