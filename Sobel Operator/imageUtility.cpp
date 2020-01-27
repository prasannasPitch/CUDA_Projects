/*******************************************************************************
------------------------------------------------------------------------
Program  :  imageUtility.cpp
Purpose  :  utility file for image manupulation
Author   :  Marimuthu, Prasanna
Date     :  23.01.2019
*******************************************************************************/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include "cudaUtils.h"

namespace {

cv::Mat imageRGBA;
cv::Mat imageGrey;

size_t numRows() {
    return imageRGBA.rows;
}
    
size_t numCols() {
    return imageRGBA.cols;
}

uchar4*        d_rgbaImage__;
unsigned char* d_greyImage__;
unsigned char* d_sobelImage__;

void loadImage(uchar4** inputImage, const std::string& filename) {

    cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
    if(image.empty()) {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }

    cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);
    *inputImage = (uchar4*)imageRGBA.ptr<unsigned char>(0);
    imageGrey.create(imageRGBA.rows, imageRGBA.cols, CV_8UC1);

}

void saveImage(const std::string& output_file, unsigned char* data_ptr) {

    cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);
    cv::imwrite(output_file.c_str(), output);
}

void createDeviceImageMemory(uchar4** h_rgbaImage, unsigned char** h_greyImage, unsigned char** h_sobelImage,
                             uchar4** d_rgbaImage, unsigned char** d_greyImage, unsigned char** d_sobelImage) {

    *h_greyImage = imageGrey.ptr<unsigned char>(0);
    *h_sobelImage = imageGrey.ptr<unsigned char>(0);
    const size_t numPixels = numRows() * numCols();
    checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(d_sobelImage, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_sobelImage, 0, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));

    cudaMemcpy(*d_rgbaImage, *h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

    //set pointers to free memory
    d_rgbaImage__ = *d_rgbaImage;
    d_greyImage__ = *d_greyImage;
    d_sobelImage__ = *d_sobelImage;
}

void cleanup() {
    //cleanup
    cudaFree(d_rgbaImage__);
    cudaFree(d_greyImage__);
    cudaFree(d_sobelImage__);
}
}
