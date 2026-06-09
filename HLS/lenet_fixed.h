#ifndef LENET_FIXED_H
#define LENET_FIXED_H

#include <ap_fixed.h>
#include <iostream>

// Storage type: 16 bits total, 6 integer bits, 10 fractional bits
// Range: -32 to +31.999, Precision: ~0.001
typedef ap_fixed<16,8> fixed_t;

// Accumulator type: wider to prevent overflow during MAC operations
// Range: -32768 to +32767, used inside layer computations only
typedef ap_fixed<32,16> acc_t;

// Input Dimensions
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_C 1

// Layer 1: Conv 5x5, 6 Filters
#define C1_H 28
#define C1_W 28
#define C1_CH 6
#define C1_K 5

// Layer 2: MaxPool 2x2
#define S2_H 14
#define S2_W 14
#define S2_CH 6
#define S2_K 2

// Layer 3: Conv 5x5, 16 Filters
#define C3_H 10
#define C3_W 10
#define C3_CH 16
#define C3_K 5

// Layer 4: MaxPool 2x2
#define S4_H 5
#define S4_W 5
#define S4_CH 16
#define S4_K 2

// Fully Connected Layers
#define FC1_UNITS 120
#define FC2_UNITS 84
#define OUTPUT_CLASSES 43

// Top function prototype
void lenet_predict_fixed(fixed_t input[1][32][32], int *predicted_class);

#endif