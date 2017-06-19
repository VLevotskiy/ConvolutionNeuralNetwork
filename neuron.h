#ifndef NEURON_H
#define NEURON_H
#include <vector>

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

#endif // NEURON_H
