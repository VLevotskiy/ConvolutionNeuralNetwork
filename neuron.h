#ifndef NEURON_H
#define NEURON_H
#include <stdlib.h>
#include <vector>
#include <string>
#include "convolutionnn.h"

enum Layer_type{FullConnected=1,Convolution,Pooling};
const int NUM_OF_LAYERS_TYPES = 3;
const std::string list_of_layers_types[3] = {"FullConnected", "Convolution", "Pooling"};


//класс связи. Хранит информацию о привязке нейрона к нейрону другого слоя, весовой коэффициент связи и последнее изменение связи
class Connection{
private:
    unsigned int neuron_num;
    float weight;
    float last_dw;
public:
    Connection():weight(0),neuron_num(0),last_dw(0){}
    Connection(const unsigned int n,const float w):last_dw(0){
        weight = w;
        neuron_num = n;
    }

    inline void Fill_Connection(unsigned int n, float w) {
        weight = w;
        neuron_num = n;
        last_dw  = 0;
    }
    inline void Set_neuron_num(const unsigned int n) { neuron_num = n; }
    inline void Set_weight(const float w) { weight = w; }
    inline void Set_Last_dw(const float dw) { last_dw = dw; }
    inline unsigned int Get_Neuron_num() const { return neuron_num; }
    inline float Get_Weight() const { return weight; }
    inline float Get_Last_dw() const { return last_dw; }
};

//Класс нейрон. Хранит информацию о значении нейрона и массив его связей с нейронами предыдущего слоя
class Neuron {
private:
    float out_value;
    std::vector<Connection> connections;
public:
    Neuron(){ connections.clear(); out_value = 0;}
    inline void Add_Connection( unsigned int n, float weight ){ connections.push_back(Connection(n,weight)); }
    inline float Get_value() const { return out_value;}
    inline std::vector<Connection>* Get_Connections() { return &connections;}
};

//Класс слоя. Хранит информацию о слое сети
class Layer {
private:
    std::vector<Neuron> neurons;
    unsigned int layer_size;
    Layer* prev_layer;
    Layer_type type;
public:
    Layer(int n, Layer* prev,Layer_type t, unsigned int step_size = 0, unsigned int el_width = 0, unsigned int el_height = 0, unsigned char num_of_masks = 0, size_t input_height = 0, size_t input_width = 0);
    void Set_Layer_Type(Layer_type t, unsigned int step_size = 0, unsigned int el_width = 0, unsigned int el_height = 0, unsigned char num_of_masks = 0, size_t input_height = 0, size_t input_width = 0);
    unsigned int Size() const { return layer_size;}
};

class NeuralNet {
private:
    std::vector<Layer> layers;
    unsigned int num_of_layers;
public:
    NeuralNet(uint8_t num_of_layers_,std::string& types,std::vector<uint16_t>& layers_sizes):num_of_layers(num_of_layers_){
        for (int i =0; i < num_of_layers_; i++){
            int type = Parser(types[i],)
        }
    }
};

#endif // NEURON_H*/

