/*******************************************************************************
------------------------------------------------------------------------
Program  :  main.cpp
Purpose  :  Main
Author   :  Marimuthu, Prasanna
Date     :  23.01.2019
*******************************************************************************/

#include <iostream>
#include <string>
#include <stdio.h>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "imageUtility.cpp"
#include "cudaUtils.h"


void greyFilter(uchar4* const d_rgbaImage,
                unsigned char* const d_greyImage, const size_t numRows, const size_t numCols);

void sobelFilter(unsigned char* const d_greyImage, unsigned char* d_sobelImage, const size_t numRows, const size_t numCols);

using namespace std;

int main() {

    uchar4*        h_rgbaImage, *d_rgbaImage;
    unsigned char* h_greyImage, *h_sobelImage, *d_greyImage, *d_sobelImage;
    std::string fileName = "parking.jpg";
    std::string output_file = "greyOutput.png";
    std::string output_file_sobel = "sobelOutput.png";

    loadImage(&h_rgbaImage, fileName);
    createDeviceImageMemory(&h_rgbaImage, &h_greyImage, &h_sobelImage, &d_rgbaImage, &d_greyImage, &d_sobelImage);

    {
        auto startTime = std::chrono::high_resolution_clock::now();
        greyFilter(d_rgbaImage, d_greyImage, numRows(), numCols());
        auto endTime = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> totalTime_ms = (endTime - startTime);
        cout << totalTime_ms.count() << "\n";
    }

    size_t numPixels = numRows() * numCols();
    checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    saveImage(output_file, h_greyImage);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        sobelFilter(d_greyImage, d_sobelImage, numRows(), numCols());
        auto endTime = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> totalTime_ms = (endTime - startTime);
        cout << totalTime_ms.count() << "\n";
    }

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_sobelImage, d_sobelImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    saveImage(output_file_sobel, h_sobelImage);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    cleanup();
    return 0;
}


