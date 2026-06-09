#include <iostream>
#include <cmath>
#include "lenet_fixed.h"
#include "parameters.h"
#include "image_data.h"

float relu_f(float x) { return x > 0 ? x : 0; }

void ref_conv1(float input[1][32][32], float output[6][28][28]) {
    for (int co = 0; co < 6; co++)
        for (int h = 0; h < 28; h++)
            for (int w = 0; w < 28; w++) {
                float sum = conv1_bias[co];
                for (int kh = 0; kh < 5; kh++)
                    for (int kw = 0; kw < 5; kw++)
                        sum += input[0][h+kh][w+kw] * conv1_weights[co][0][kh][kw];
                output[co][h][w] = relu_f(sum);
            }
}

int main() {

    // ---- Step 1: Input conversion check ----
    std::cout << "=== Step 1: Input conversion check ===" << std::endl;
    fixed_t input_fixed[1][32][32];
    float   input_float[1][32][32];
    float max_input_err = 0;
    for (int h = 0; h < 32; h++)
        for (int w = 0; w < 32; w++) {
            float orig = test_image[0][h][w];
            input_fixed[0][h][w] = fixed_t(orig);
            input_float[0][h][w] = orig;
            float err = fabs((float)input_fixed[0][h][w] - orig);
            if (err > max_input_err) max_input_err = err;
        }
    std::cout << "Max input conversion error: " << max_input_err << std::endl;

    // ---- Step 2: Conv1 output check ----
    std::cout << "\n=== Step 2: Conv1 output check ===" << std::endl;
    float ref_c1[6][28][28];
    ref_conv1(input_float, ref_c1);
    float max_c1_err = 0, max_c1_fixed_val = 0, max_c1_float_val = 0;
    for (int co = 0; co < 6; co++)
        for (int h = 0; h < 28; h++)
            for (int w = 0; w < 28; w++) {
                acc_t sum = conv1_bias[co];
                for (int kh = 0; kh < 5; kh++)
                    for (int kw = 0; kw < 5; kw++)
                        sum += input_fixed[0][h+kh][w+kw] * fixed_t(conv1_weights[co][0][kh][kw]);
                fixed_t out = (sum > 0) ? fixed_t(sum) : fixed_t(0);
                float err = fabs((float)out - ref_c1[co][h][w]);
                if (err > max_c1_err) {
                    max_c1_err = err;
                    max_c1_fixed_val = (float)out;
                    max_c1_float_val = ref_c1[co][h][w];
                }
            }
    std::cout << "Max conv1 error: " << max_c1_err << std::endl;
    std::cout << "  fixed=" << max_c1_fixed_val << " float=" << max_c1_float_val << std::endl;

    // ---- Step 3: FC1 weight range check ----
    std::cout << "\n=== Step 3: FC1 weight range check ===" << std::endl;
    float max_w = 0, min_w = 1e9;
    for (int o = 0; o < 120; o++)
        for (int i = 0; i < 400; i++) {
            float w = fc1_weights[o][i];
            if (fabs(w) > max_w) max_w = fabs(w);
            if (fabs(w) < min_w) min_w = fabs(w);
        }
    std::cout << "fc1_weights range: " << min_w << " to " << max_w << std::endl;

    // ---- Step 4: FC1 accumulator range check ----
    std::cout << "\n=== Step 4: FC1 accumulator range ===" << std::endl;
    float max_possible_sum = 0;
    for (int o = 0; o < 120; o++) {
        float sum = fabs(fc1_bias[o]);
        for (int i = 0; i < 400; i++)
            sum += fabs(fc1_weights[o][i]);
        if (sum > max_possible_sum) max_possible_sum = sum;
    }
    // fixed: ap_fixed<16,8> range is 127.99, not 31.99
    float fixed_t_max = 127.99;
    std::cout << "Max possible fc1 accumulator value: " << max_possible_sum << std::endl;
    std::cout << "ap_fixed<32,16> max range: 32767" << std::endl;
    std::cout << "ap_fixed<16,8>  max range: " << fixed_t_max << std::endl;
    if (max_possible_sum > fixed_t_max)
        std::cout << "WARNING: fc1 sum OVERFLOWS fixed_t storage!" << std::endl;
    else
        std::cout << "OK: fc1 sum fits in ap_fixed<16,8> storage." << std::endl;

    // ---- Step 5: Full prediction ----
    std::cout << "\n=== Step 5: Full prediction ===" << std::endl;
    int predicted_class = -1;
    lenet_predict_fixed(input_fixed, &predicted_class);
    std::cout << "Fixed predicted class: " << predicted_class << std::endl;
    std::cout << "Expected class       : 29" << std::endl;
    if (predicted_class == 29)
        std::cout << "PASS" << std::endl;
    else
        std::cout << "FAIL - investigating above steps" << std::endl;

    return 0;
}