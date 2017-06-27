#include <QCoreApplication>
#include "neuralnet.h"
#include <iostream>
std::string layers[]={"Layer_type = FullConnected num_of_neurons = 400 activation_func = Sigmoid",
                        "Layer_type = FullConnected num_of_neurons = 21 activation_func = Sigmoid"};
                   //"Layer_type = Convolution activation_func = ReLU img_height = 29 img_width = 29 el_width = 3 el_height = 3 number_of_masks = 6",
                   //"Layer_type = Pooling activation_func = Linear img_height = 29 img_width = 29 el_width = 2 el_height = 2 number_of_masks = 6",
                   //"Layer_type = FullConnected num_of_neurons = 1 activation_func = Sigmoid"};


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    NeuralNet new_net(layers, 2,273);

    double mat[4][3] = {{0.0, 0.0, 0.0},
                      {1.0, 0.0, 1.0},
                      {0.0, 1.0, 1.0},
                      {1.0, 1.0, 1.0}};

    new_net.training("C:/Users/coder/Documents/build-ConvolutionNeuralNetwork-Desktop_Qt_5_5_1_MinGW_32bit-Release/release/13_21/");
    /*std::vector<double> inp;
    std::vector<double> out;
    out.push_back(1);
    inp.push_back(0);
    inp.push_back(1);
    int cntr =0;
    for (int i =0; i < 10; i++){
        inp[0] = mat[cntr][0];
        inp[1] = mat[cntr][1];
        new_net.forward_propagation(inp);
        out[0] = mat[cntr][2];
        std::cout << "out = " << out[0] << std::endl;
        new_net.back_propagation(out);
        cntr++;
        if (cntr ==  4) cntr = 0;
    }
    std::cout << "CREATED!" << std::endl;*/

    return a.exec();
}
