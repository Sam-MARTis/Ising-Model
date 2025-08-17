#include "kernel.hpp"
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#define TY 32
#define TX 32

#define INITIALIZATION_THRESHOLD 0.9f
curandGenerator_t gen;

__global__
void ising_iteration(bool* d_state, float *d_rand, const float beta, const int width, const int height){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * width + x;

    if(x >= width || y >= height) return;

    // Calculate the index of the neighboring cells
    int left = (x == 0) ? (width - 1) : (x - 1);
    int right = (x == width - 1) ? 0 : (x + 1);
    int up = (y == 0) ? (height - 1) : (y - 1);
    int down = (y == height - 1) ? 0 : (y + 1);
    const bool current_state = d_state[idx];
    int8_t spin = current_state ? 1 : -1;
    int8_t leftSpin = d_state[up * width + left] ? 1 : -1;
    int8_t rightSpin = d_state[up * width + right] ? 1 : -1;
    int8_t upSpin = d_state[up * width + x] ? 1 : -1;
    int8_t downSpin = d_state[down * width + x] ? 1 : -1;

 
    int deltaE = 2 * spin * (upSpin + downSpin + leftSpin + rightSpin);


    if(deltaE < 0 || d_rand[idx] < expf(-beta * deltaE)) {
        d_state[idx] = !current_state; // Flip the state
    }

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
        d_cols[idx].y = 255;
        d_cols[idx].z = 255;
    } else {
        d_cols[idx].x = 0;
        d_cols[idx].y = 0;
        d_cols[idx].z = 0;
    }
    d_cols[idx].w = 255;
}

void IsingKernelLauncher(uchar4 *d_out, const float beta, int width, int height, int iterations_per_draw){

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((width + TX - 1) / TX, (height + TY - 1) / TY);
    float *d_rand;
    bool *d_state;
    cudaMalloc(&d_state, width * height * sizeof(bool));
    cudaMalloc(&d_rand, width * height * sizeof(float));
    convert_to_bool<<<gridSize, blockSize>>>(d_state, d_out, width, height);

    for (int i = 0; i < iterations_per_draw; i++) {
        curandGenerateUniform(gen, d_rand, width * height);
        ising_iteration<<<gridSize, blockSize>>>(d_state, d_rand, beta, width, height);
    }

    convert_to_colour<<<gridSize, blockSize>>>(d_state, d_out, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_rand);

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
        d_out[idx].y = 255;
        d_out[idx].z = 255;
    } else {
        d_out[idx].x = 0;
        d_out[idx].y = 0;
        d_out[idx].z = 0;
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