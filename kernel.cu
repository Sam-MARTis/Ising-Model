#include "kernel.hpp"
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#define TY 32
#define TX 32
#define TX2 64

#define INITIALIZATION_THRESHOLD 0.1f

#define DRAWS_PER_REPORT 30

// const static int cols[6] = {255, 0, 0, 100, 0, 255};
#define URED 255
#define UGREEN 255
#define UBLUE 255
#define DRED 0
#define DGREEN 0
#define DBLUE 0
curandGenerator_t gen;

__global__
void ising_iteration(bool* d_state, float *d_rand, const float beta, const int width, const int height, const int type){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * width + x;

    if(x >= width || y >= height) return;
    const int hammington_distance = x+y;
    if((hammington_distance+type) % 2) return;

    int left = (x == 0) ? (width - 1) : (x - 1);
    int right = (x == width - 1) ? 0 : (x + 1);
    int up = (y == 0) ? (height - 1) : (y - 1);
    int down = (y == height - 1) ? 0 : (y + 1);
    const bool current_state = d_state[idx];
    int spin = current_state ? 1 : -1;
    int leftSpin  = d_state[y * width + left] ? 1 : -1;
    int rightSpin = d_state[y * width + right] ? 1 : -1;
    int upSpin    = d_state[up * width + x] ? 1 : -1;
    int downSpin  = d_state[down * width + x] ? 1 : -1;

    
 
    float deltaE = 2 * spin * (upSpin + downSpin + leftSpin + rightSpin);

    const float val = expf(-beta * deltaE);

    __syncthreads();
    if(deltaE < 0 || d_rand[idx] < val) {
        d_state[idx] = !current_state;
    }
    __syncthreads();

}

__global__
void convert_to_bool(bool* d_state, uchar4* d_cols, int width, int height){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * width + x;
    if(x >= width || y >= height) return;
    d_state[idx] = (d_cols[idx].x == URED) ? true : false;

}
__global__
void convert_to_colour(bool* d_state, uchar4* d_cols, int width, int height){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * width + x;
    if(x >= width || y >= height) return;

    if(d_state[idx]){
        d_cols[idx].x = URED;
        d_cols[idx].y = UGREEN;
        d_cols[idx].z = UBLUE;
    } else {
        d_cols[idx].x = DRED;
        d_cols[idx].y = DGREEN;
        d_cols[idx].z = DBLUE;
    }
    d_cols[idx].w = 255;
}
__global__
void properties(bool* state, int size){
    __shared__ int sumData[TX2];
    __shared__ int sumsqData[TX2];
    // int sum = 0;
    // int sumsq = 0;
    // for(int i = 0; i < size; i++){
    //     const int& val = state[i] ? 1 : -1;
    //     sum += val;
    //     sumsq += val * val;
    // }
    const int id = threadIdx.x;
    for(int i = id; i < size; i += blockDim.x){
        const int val = state[i] ? 1 : -1;
        sumData[id] += val;
        sumsqData[id] += val * val;
    }
    __syncthreads();
    if(id == 0){
        int sum = 0;
        int sumsq = 0;
        for(int i = 0; i < blockDim.x; i++){
            sum += sumData[i];
            sumsq += sumsqData[i];
        }
        float mean = (float)sum / size;
        float variance = ((float)sumsq / size) - (mean * mean);
        printf("Mean: %f, Variance: %f\n", mean, variance);
    }
}
int counter = 0;
void IsingKernelLauncher(uchar4 *d_out, const float beta, int width, int height, int iterations_per_draw){
    counter+=1;
    const dim3 blockSize(TX, TY);
    const dim3 gridSize((width + TX - 1) / TX, (height + TY - 1) / TY);
    float *d_rand;
    bool *d_state;
    cudaMalloc(&d_state, width * height * sizeof(bool));
    cudaMalloc(&d_rand, width * height * sizeof(float));
    convert_to_bool<<<gridSize, blockSize>>>(d_state, d_out, width, height);

    for (int i = 0; i < iterations_per_draw; i++) {
        curandGenerateUniform(gen, d_rand, width * height);
        ising_iteration<<<gridSize, blockSize>>>(d_state, d_rand, beta, width, height, 1);
        // curandGenerateUniform(gen, d_rand, width * height);
        ising_iteration<<<gridSize, blockSize>>>(d_state, d_rand, beta, width, height, 2);
    }

    if(counter%DRAWS_PER_REPORT == 0){
        const dim3 blockSize2(TX2);
        const dim3 gridSize2(1);
        properties<<<gridSize2, blockSize2>>>(d_state, width * height);
    }
    convert_to_colour<<<gridSize, blockSize>>>(d_state, d_out, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_rand);
    cudaFree(d_state);
    
    cudaDeviceSynchronize();
    
    return;
}



__global__ void initialization_kernel(uchar4 *d_out, float *d_rand, int width, int height, const float threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * width + x;
    if (x >= width || y >= height) return;
    if (d_rand[idx] < threshold) {
        d_out[idx].x = URED;
        d_out[idx].y = UGREEN;
        d_out[idx].z = UBLUE;
    } else {
        d_out[idx].x = DRED;
        d_out[idx].y = DGREEN;
        d_out[idx].z = DBLUE;
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