#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include "connection.h"

class Neuron {
private:
    float out_value;
    std::vector<Connection> connections;
public:
    Neuron(){ connections.clear(); out_value = 0;}
    Neuron(float value) {connections.clear(); out_value = value;}
    void Add_Connection( unsigned int n, float weight ){ connections.push_back(Connection(n,weight)); }
    float Get_value() const { return out_value;}
    void Set_value(const float value) { out_value = value;}
    std::vector<Connection>* Get_Connections() { return &connections;}
};

#endif // NEURON_H
