#include <QCoreApplication>
#include "neuralnet.h"
#include <iostream>
std::string layers[] = {"Layer_type = FullConnected num_of_neurons = 841 activation_func = Linear",
                   "Layer_type = Convolution activation_func = ReLU img_height = 29 img_width = 29 el_with = 3 el_height = 3 number_of_masks = 6",
                   "Layer_type = Pooling activation_func = Linear img_height = 29 img_width = 29 el_with = 2 el_height = 2 number_of_masks = 6",
                   "Layer_type = FullConnected num_of_neurons = 841 activation_func = Sigmoid"};


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    NeuralNet new_net(layers, 4,841);
    std::cout << "CREATED!" << std::endl;

    return a.exec();
}
