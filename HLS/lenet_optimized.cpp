#include "lenet.h"
#include "parameters.h"

// --- Activation Function ---
float relu(float x) {
    return (x > 0) ? x : 0;
}

// --- Layer 1: Convolution (Input -> C1) ---
void conv_layer1(float input[1][32][32], float output[6][28][28]) {
    #pragma HLS ARRAY_PARTITION variable=conv1_weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=conv1_bias complete dim=1
    for (int co = 0; co < 6; co++) {
        for (int h = 0; h < 28; h++) {
            for (int w = 0; w < 28; w++) {
                #pragma HLS PIPELINE II=1
                float sum = conv1_bias[co];
                for (int ci = 0; ci < 1; ci++) {
                    for (int kh = 0; kh < 5; kh++) {
                        for (int kw = 0; kw < 5; kw++) {
                            sum += input[ci][h+kh][w+kw] * conv1_weights[co][ci][kh][kw];
                        }
                    }
                }
                output[co][h][w] = relu(sum);
            }
        }
    }
}

// --- Layer 2: Max Pooling (C1 -> S2) ---
void pool_layer1(float input[6][28][28], float output[6][14][14]) {
    for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 14; h++) {
            for (int w = 0; w < 14; w++) {
                #pragma HLS PIPELINE II=1
                float max_val = -1e9;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        float val = input[c][h*2 + kh][w*2 + kw];
                        if (val > max_val) max_val = val;
                    }
                }
                output[c][h][w] = max_val;
            }
        }
    }
}

// --- Layer 3: Convolution (S2 -> C3) ---
void conv_layer2(float input[6][14][14], float output[16][10][10]) {
    #pragma HLS ARRAY_PARTITION variable=conv2_weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=conv2_bias complete dim=1
    for (int co = 0; co < 16; co++) {
        for (int h = 0; h < 10; h++) {
            for (int w = 0; w < 10; w++) {
                #pragma HLS PIPELINE II=1
                float sum = conv2_bias[co];
                for (int ci = 0; ci < 6; ci++) {
                    for (int kh = 0; kh < 5; kh++) {
                        for (int kw = 0; kw < 5; kw++) {
                            sum += input[ci][h+kh][w+kw] * conv2_weights[co][ci][kh][kw];
                        }
                    }
                }
                output[co][h][w] = relu(sum);
            }
        }
    }
}

// --- Layer 4: Max Pooling (C3 -> S4) ---
void pool_layer2(float input[16][10][10], float output[16][5][5]) {
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 5; h++) {
            for (int w = 0; w < 5; w++) {
                #pragma HLS PIPELINE II=1
                float max_val = -1e9;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        float val = input[c][h*2 + kh][w*2 + kw];
                        if (val > max_val) max_val = val;
                    }
                }
                output[c][h][w] = max_val;
            }
        }
    }
}

// --- Layer 5: Fully Connected (S4 -> FC1) ---
void fc_layer1(float input[16][5][5], float output[120]) {
    #pragma HLS ARRAY_PARTITION variable=fc1_weights cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=fc1_bias complete dim=1
    for (int o = 0; o < 120; o++) {
        #pragma HLS PIPELINE II=1
        float sum = fc1_bias[o];
        int flat_idx = 0;
        for (int c = 0; c < 16; c++) {
            for (int h = 0; h < 5; h++) {
                for (int w = 0; w < 5; w++) {
                    sum += input[c][h][w] * fc1_weights[o][flat_idx];
                    flat_idx++;
                }
            }
        }
        output[o] = relu(sum);
    }
}

// --- Layer 6: Fully Connected (FC1 -> FC2) ---
void fc_layer2(float input[120], float output[84]) {
    #pragma HLS ARRAY_PARTITION variable=fc2_weights cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=fc2_bias complete dim=1
    for (int o = 0; o < 84; o++) {
        #pragma HLS PIPELINE II=1
        float sum = fc2_bias[o];
        for (int i = 0; i < 120; i++) {
            sum += input[i] * fc2_weights[o][i];
        }
        output[o] = relu(sum);
    }
}

// --- Layer 7: Output (FC2 -> FC3) ---
void fc_layer3(float input[84], float output[43]) {
    #pragma HLS ARRAY_PARTITION variable=fc3_weights cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=fc3_bias complete dim=1
    for (int o = 0; o < 43; o++) {
        #pragma HLS PIPELINE II=1
        float sum = fc3_bias[o];
        for (int i = 0; i < 84; i++) {
            sum += input[i] * fc3_weights[o][i];
        }
        output[o] = sum;
    }
}

// --- TOP LEVEL FUNCTION ---
void lenet_predict(float input[1][32][32], int *predicted_class) {

    // Internal Buffers
    float c1_out[6][28][28];
    float s2_out[6][14][14];
    float c3_out[16][10][10];
    float s4_out[16][5][5];
    float fc1_out[120];
    float fc2_out[84];
    float fc3_out[43];

    // Execution
    conv_layer1(input, c1_out);
    pool_layer1(c1_out, s2_out);
    conv_layer2(s2_out, c3_out);
    pool_layer2(c3_out, s4_out);
    fc_layer1(s4_out, fc1_out);
    fc_layer2(fc1_out, fc2_out);
    fc_layer3(fc2_out, fc3_out);

    // Argmax
    int max_id = 0;
    float max_val = fc3_out[0];
    for (int i = 1; i < 43; i++) {
        if (fc3_out[i] > max_val) {
            max_val = fc3_out[i];
            max_id = i;
        }
    }
    *predicted_class = max_id;
}