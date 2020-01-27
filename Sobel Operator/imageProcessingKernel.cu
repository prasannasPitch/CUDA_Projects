/*******************************************************************************
------------------------------------------------------------------------
Program  :  imageProcessingKernel.cu
Purpose  :  kernel for image processing
Author   :  Marimuthu, Prasanna
Date     :  23.01.2019
*******************************************************************************/

#include "cudaUtils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage, size_t numRows, size_t numCols) {

    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if(y < numCols && x < numRows) {
        int index = numRows * y + x;
        uchar4 color = rgbaImage[index];
        unsigned char greyness = (unsigned char)(0.299f * color.x + 0.587f * color.y + 0.114f * color.z);
        greyImage[index] = greyness;
    }
}

__global__
void sobel_gpu(unsigned char* const originalImage, unsigned char* filterOutputImage, size_t numRows, size_t numCols) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x > 0 && y > 0 && x < numRows - 1 && y < numCols - 1) {
        float gradX = (-1 * originalImage[(y - 1) * numRows + (x - 1)]) + (-2 * originalImage[y * numRows + (x - 1)]) + (-1 * originalImage[(y + 1) * numRows + (x - 1)]) +
                      (originalImage[(y - 1) * numRows + (x + 1)]) + (2 * originalImage[y * numRows + (x + 1)]) + (originalImage[(y + 1) * numRows + (x + 1)]);
        float gradY = (originalImage[(y - 1) * numRows + (x - 1)]) + (2 * originalImage[(y - 1) * numRows + x]) + (originalImage[(y - 1) * numRows + (x + 1)]) +
                      (-1 * originalImage[(y + 1) * numRows + (x - 1)]) + (-2 * originalImage[(y + 1) * numRows + x]) + (-1 * originalImage[(y + 1) * numRows + (x + 1)]);

        auto temp = sqrt((gradX * gradX) + (gradY * gradY));
        filterOutputImage[row * numRows + col] = temp < 128 ? 255 : 0;

    }
}

__global__
void sobel_gpu_convolve2D(unsigned char* const originalImage, unsigned char* filterOutputImage, size_t numRows, size_t numCols) {

    int maskHorrizontal[3 * 3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    int maskVertical[3 * 3] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    int maskDimension = 3;
    int maskOffset = mastkDimension / 2;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int startRow = row - maskOffset;
    int startCol = col - maskOffset;
    float temp1 = 0;
    float temp2 = 0;

    // Iterate over all the rows
    for(int i = 0; i < maskDimension; i++) {
        // Go over each column
        for(int j = 0; j < maskDimension; j++) {
            // Range check for rows
            if((startRow + i) >= 0 && (startRow + i) < numRows) {
                // Range check for columns
                if((startCol + j) >= 0 && (startCol + j) < numCols) {
                    // Accumulate result
                    temp1 += (originalImage[(startRow + i) * numRows + (startCol + j)] *
                              maskHorrizontal[i * maskDimension + j]);

                    temp2 += (originalImage[(startRow + i) * numRows + (startCol + j)] *
                              maskVertical[i * maskDimension + j]);
                }
            }
        }
    }

    auto temp = sqrt((temp1 * temp1) + (temp2 * temp2));
    filterOutputImage[row * numRows + col] = temp < 128 ? 255 : 0;
}

void greyFilter(uchar4* const d_rgbaImage, unsigned char* const d_greyImage, const size_t numRows, const size_t numCols) {

    int   blockWidth = 32;
    const dim3 blockSize(blockWidth, blockWidth, 1);
    int   blocksX = (int)numRows / blockWidth + 1;
    int   blocksY = (int)numCols / blockWidth + 1;
    const dim3 gridSize(blocksX, blocksY, 1);

    rgba_to_greyscale << <gridSize, blockSize >> > (d_rgbaImage, d_greyImage, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void sobelFilter(unsigned char* const d_greyImage, unsigned char* d_sobelImage, const size_t numRows, const size_t numCols) {

    int   blockWidth = 32;
    const dim3 blockSize(blockWidth, blockWidth, 1);
    int   blocksX = (int)numRows / blockWidth + 1;
    int   blocksY = (int)numCols / blockWidth + 1;
    const dim3 gridSize(blocksX, blocksY, 1);

    sobel_gpu1 << <gridSize, blockSize >> > (d_greyImage, d_sobelImage, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
