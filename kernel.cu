#include "kernel.hpp"
#include <iostream>
#define TY 32
#define TX 32


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

void kernelLauncher(uchar4 *d_out, float temperature, int width, int height, int iterations_per_draw){


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