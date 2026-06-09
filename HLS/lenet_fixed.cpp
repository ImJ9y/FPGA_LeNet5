#include "lenet_fixed.h"
#include "parameters.h"

fixed_t relu_fixed(fixed_t x) {
    return (x > 0) ? x : fixed_t(0);
}

// --- Layer 1: Convolution (Input -> C1) ---
// Small: 1 input channel, 6 filters, 5x5 kernel = 25 MACs per output
void conv_layer1_fixed(fixed_t input[1][32][32], fixed_t output[6][28][28]) {
    #pragma HLS ARRAY_PARTITION variable=conv1_weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=conv1_bias complete dim=1
    for (int co = 0; co < 6; co++) {
        for (int h = 0; h < 28; h++) {
            for (int w = 0; w < 28; w++) {
                #pragma HLS PIPELINE II=1
                acc_t sum = conv1_bias[co];
                for (int kh = 0; kh < 5; kh++) {
                    for (int kw = 0; kw < 5; kw++) {
                        sum += input[0][h+kh][w+kw] * fixed_t(conv1_weights[co][0][kh][kw]);
                    }
                }
                output[co][h][w] = relu_fixed(fixed_t(sum));
            }
        }
    }
}

// --- Layer 2: Max Pooling (C1 -> S2) ---
void pool_layer1_fixed(fixed_t input[6][28][28], fixed_t output[6][14][14]) {
    for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 14; h++) {
            for (int w = 0; w < 14; w++) {
                #pragma HLS PIPELINE II=1
                fixed_t max_val = -128;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        fixed_t val = input[c][h*2+kh][w*2+kw];
                        if (val > max_val) max_val = val;
                    }
                }
                output[c][h][w] = max_val;
            }
        }
    }
}

// --- Layer 3: Convolution (S2 -> C3) ---
// Larger: 6 input channels, 16 filters, 5x5 = 150 MACs per output
// Pipeline at output pixel level only — no array partition on weights
void conv_layer2_fixed(fixed_t input[6][14][14], fixed_t output[16][10][10]) {
    for (int co = 0; co < 16; co++) {
        for (int h = 0; h < 10; h++) {
            for (int w = 0; w < 10; w++) {
                acc_t sum = conv2_bias[co];
                for (int ci = 0; ci < 6; ci++) {
                    for (int kh = 0; kh < 5; kh++) {
                        for (int kw = 0; kw < 5; kw++) {
                            #pragma HLS PIPELINE II=1
                            sum += input[ci][h+kh][w+kw] * fixed_t(conv2_weights[co][ci][kh][kw]);
                        }
                    }
                }
                output[co][h][w] = relu_fixed(fixed_t(sum));
            }
        }
    }
}

// --- Layer 4: Max Pooling (C3 -> S4) ---
void pool_layer2_fixed(fixed_t input[16][10][10], fixed_t output[16][5][5]) {
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 5; h++) {
            for (int w = 0; w < 5; w++) {
                #pragma HLS PIPELINE II=1
                fixed_t max_val = -128;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        fixed_t val = input[c][h*2+kh][w*2+kw];
                        if (val > max_val) max_val = val;
                    }
                }
                output[c][h][w] = max_val;
            }
        }
    }
}

// --- Layer 5: Fully Connected (S4 -> FC1) ---
void fc_layer1_fixed(fixed_t input[16][5][5], fixed_t output[120]) {
    for (int o = 0; o < 120; o++) {
        acc_t sum = fc1_bias[o];
        int flat_idx = 0;
        for (int c = 0; c < 16; c++) {
            for (int h = 0; h < 5; h++) {
                for (int w = 0; w < 5; w++) {
                    #pragma HLS PIPELINE II=1
                    sum += input[c][h][w] * fixed_t(fc1_weights[o][flat_idx]);
                    flat_idx++;
                }
            }
        }
        output[o] = relu_fixed(fixed_t(sum));
    }
}

// --- Layer 6: Fully Connected (FC1 -> FC2) ---
void fc_layer2_fixed(fixed_t input[120], fixed_t output[84]) {
    for (int o = 0; o < 84; o++) {
        acc_t sum = fc2_bias[o];
        for (int i = 0; i < 120; i++) {
            #pragma HLS PIPELINE II=1
            sum += input[i] * fixed_t(fc2_weights[o][i]);
        }
        output[o] = relu_fixed(fixed_t(sum));
    }
}

// --- Layer 7: Output (FC2 -> FC3) ---
void fc_layer3_fixed(fixed_t input[84], fixed_t output[43]) {
    for (int o = 0; o < 43; o++) {
        acc_t sum = fc3_bias[o];
        for (int i = 0; i < 84; i++) {
            #pragma HLS PIPELINE II=1
            sum += input[i] * fixed_t(fc3_weights[o][i]);
        }
        output[o] = fixed_t(sum);
    }
}

// --- TOP LEVEL FUNCTION ---
void lenet_predict_fixed(fixed_t input[1][32][32], int *predicted_class) {
    #pragma HLS INTERFACE m_axi     port=input depth=1024 offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=input      bundle=control
    #pragma HLS INTERFACE s_axilite port=predicted_class bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    fixed_t c1_out[6][28][28];
    fixed_t s2_out[6][14][14];
    fixed_t c3_out[16][10][10];
    fixed_t s4_out[16][5][5];
    fixed_t fc1_out[120];
    fixed_t fc2_out[84];
    fixed_t fc3_out[43];

    conv_layer1_fixed(input, c1_out);
    pool_layer1_fixed(c1_out, s2_out);
    conv_layer2_fixed(s2_out, c3_out);
    pool_layer2_fixed(c3_out, s4_out);
    fc_layer1_fixed(s4_out, fc1_out);
    fc_layer2_fixed(fc1_out, fc2_out);
    fc_layer3_fixed(fc2_out, fc3_out);

    int max_id = 0;
    fixed_t max_val = fc3_out[0];
    for (int i = 1; i < 43; i++) {
        if (fc3_out[i] > max_val) {
            max_val = fc3_out[i];
            max_id = i;
        }
    }
    *predicted_class = max_id;
}