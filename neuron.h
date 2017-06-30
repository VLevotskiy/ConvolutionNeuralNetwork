#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <memory>
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
    //std::vector<Connection>& Get_connections() { return connections;}
    std::shared_ptr<std::vector<Connection> > Get_connections() { return std::make_shared<std::vector<Connection> >(connections);}
    void Update_weight(unsigned int num_of_neuron, double weight, double dw) {
        if (num_of_neuron >= connections.size())  throw std::runtime_error("Neuron::Update_weight. Neuron out of range");
        connections[num_of_neuron].Set_Last_dw(dw);
        connections[num_of_neuron].Set_weight(weight);
    }

};

#endif // NEURON_H
