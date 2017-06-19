#pragma once
#ifndef RANDOM_NUM_H
#define RANDOM_NUM_H
#include <math.h>
#include <random>

float gaussrand_t(float MO, float sko)
{
    float sum = 0, x;
    for (int i = 0; i<25; i++)
        sum += 1.0*rand() / RAND_MAX;
    x = (sqrt(2.0)*(sko)*(sum - 12.5)) / 1.99661 + MO;

    return x;
}

void gen_array(const float N, const float M, const int Z, float *array)
{
    float average = (N + M) / 2.;
    float sigma = (average - N) / 3.;

    for (int i = 0; i<Z; i++) {
        float new_value = gaussrand_t(average, sigma);

         //есть вероятность (0.3%) что сгенерированное число выйдет за нужный нам диапазон
        while (new_value < N || new_value > M)
            new_value = gaussrand_t(average, sigma);

        array[i] = new_value;
    }
}
#endif // RANDOM_NUM_H
