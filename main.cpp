#include <QCoreApplication>
#include "neuralnet.h"
#include <iostream>
std::string layers[] = {"Layer_type = FullConnected num_of_neurons = 841 activation_func = Sigmoid",
                        "Layer_type = FullConnected num_of_neurons = 400 activation_func = Sigmoid",
                   //"Layer_type = Convolution activation_func = ReLU img_height = 29 img_width = 29 el_width = 3 el_height = 3 number_of_masks = 6",
                   //"Layer_type = Pooling activation_func = Linear img_height = 29 img_width = 29 el_width = 2 el_height = 2 number_of_masks = 6",
                   "Layer_type = FullConnected num_of_neurons = 22 activation_func = Sigmoid"};


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    NeuralNet new_net(layers, 3,841);

    std::vector<float> inp;
    for (int i =0; i < 841; i++){
        inp.push_back(1);
    }

    new_net.forward_propagaition(inp);

    std::cout << "CREATED!" << std::endl;

    return a.exec();
}
