#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include "connection.h"

class Neuron {
private:
     double out_value;
     double delta;
    std::vector<Connection> connections;
public:
    Neuron(){ connections.clear(); out_value = 0;}
    Neuron( double value) {connections.clear(); out_value = value;}
    void Add_Connection( unsigned int n,  double weight ){ connections.push_back(Connection(n,weight)); }
    void Set_value(const  double value) { out_value = value;}
    void Set_delta(const  double value) { delta = value;}

     double Get_value() const { return out_value;}
     double Get_delta() const { return delta; }
    const std::vector<Connection>& Get_connections() { return connections;}

};

#endif // NEURON_H
