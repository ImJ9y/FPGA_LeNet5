
#include <iostream>
#include <ctime>
#include "lenet.h"
#include "parameters.h"
#include "image_data.h"


void lenet_predict(float input[1][32][32], int *predicted_class);

int main(){
    
    int predicted_class = -1;
    
    //copy test iamge into a non-const buffer
    float input[1][32][32];
    for (int h = 0; h < 32; h++){
        for (int w = 0; w < 32; w++){
            input[0][h][w] = test_image[0][h][w];
        }
    }

    //retun inference and measure execution time
    clock_t start = clock();
    lenet_predict(input, &predicted_class);
    clock_t end = clock();

    double elapsed_ms = (double) (end - start) / CLOCKS_PER_SEC * 1000.0;

    //results
    std::cout << "========================" << std::endl;
    std::cout << "LeNet-5 Float (Baseline)" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Predicted Class: " << predicted_class << std::endl; 
    std::cout << "Execution Time: " << elapsed_ms << std::endl;
    std::cout << "========================" << std::endl;

    //PASS or FAIL

    int expected_class = 29;
    if (predicted_class == expected_class){
         std::cout << "PASS: Prediction matches expected class." << std::endl;
    }
    else {
        std::cout << "FAIL: Expected " << expected_class
                  << " but got " << predicted_class << std::endl;
    }

    return 0;
}