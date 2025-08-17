#include "kernel.hpp"
#include <iostream>
#include <curand.h>
#define TY 32
#define TX 32

#define INITIALIZATION_THRESHOLD 0.6f
curandGenerator_t gen;

__global__
void ising_iteration(){

}

__global__
void convert_to_bool(bool* d_state, uchar4* d_cols, int width, int height){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * width + x;
    if(x >= width || y >= height) return;
    d_state[idx] = (d_cols[idx].x > 128) ? true : false;

}
__global__
void convert_to_colour(bool* d_state, uchar4* d_cols, int width, int height){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * width + x;
    if(x >= width || y >= height) return;

    if(d_state[idx]){
        d_cols[idx].x = 255;
        d_cols[idx].y = 0;
        d_cols[idx].z = 0;
    } else {
        d_cols[idx].x = 0;
        d_cols[idx].y = 0;
        d_cols[idx].z = 255;
    }
}

void IsingKernelLauncher(uchar4 *d_out, float temperature, int width, int height, int iterations_per_draw){


    const dim3 blockSize(TX, TY);
    const dim3 gridSize((width + TX - 1) / TX, (height + TY - 1) / TY);
    

    
    ising_iteration<<<gridSize, blockSize>>>();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    return;
}



__global__ void initialization_kernel(uchar4 *d_out, float *d_rand, int width, int height, const float threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * width + x;
    if (x >= width || y >= height) return;
    if (d_rand[idx] < threshold) {
        d_out[idx].x = 255;
        d_out[idx].y = 0;
        d_out[idx].z = 0;
    } else {
        d_out[idx].x = 0;
        d_out[idx].y = 0;
        d_out[idx].z = 255;
    }
    d_out[idx].w = 255;
}

void InitializationKernelLauncher(uchar4 *d_out, int width, int height){
    const dim3 blockSize(TX, TY);
    const dim3 gridSize((width + TX - 1) / TX, (height + TY - 1) / TY);

    // Launch the kernel

    float* d_rand;
    cudaMalloc(&d_rand, width * height * sizeof(float));
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);


    curandGenerateUniform(gen, d_rand, width * height);

    initialization_kernel<<<gridSize, blockSize>>>(d_out, d_rand, width, height, INITIALIZATION_THRESHOLD);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    printf("State has been initialized\n\n");

    cudaDeviceSynchronize();
    cudaFree(d_rand);


    return;
}