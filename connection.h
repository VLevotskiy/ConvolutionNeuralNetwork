#ifndef CONNECTION_H
#define CONNECTION_H


//класс связи. Хранит информацию о привязке нейрона к нейрону другого слоя, весовой коэффициент связи и последнее изменение связи
class Connection{
private:
    unsigned int neuron_num;
     double weight;
     double last_dw;
public:
    Connection():weight(0),neuron_num(0),last_dw(0){}
    Connection(const unsigned int n,const  double w):last_dw(0){
        weight = w;
        neuron_num = n;
    }

    inline void Fill_Connection(unsigned int n,  double w) {
        weight = w;
        neuron_num = n;
        last_dw  = 0;
    }
    inline void Set_neuron_num(const unsigned int n) { neuron_num = n; }
    inline void Set_weight(const  double w) { weight = w; }
    inline void Set_Last_dw(const  double dw) { last_dw = dw; }
    inline unsigned int Get_Neuron_num() const { return neuron_num; }
    inline  double Get_Weight() const { return weight; }
    inline  double Get_Last_dw() const { return last_dw; }
};

#endif // CONNECTION_H
