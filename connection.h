#ifndef CONNECTION_H
#define CONNECTION_H


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

#endif // CONNECTION_H
