#include "SineWave.cuh"

__global__ void sine_wave_kernel(float4* pos, unsigned int mesh_width, unsigned int mesh_height, float time) {

    //Code
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)mesh_width;
    float v = y / (float)mesh_height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    float frequency = 4.0f;
    float w = sinf(u * frequency + time) * cosf(v * frequency + time) * 0.5f;

    pos[y * mesh_width + x] = make_float4(u, w, v, 1.0f);
}

void launch_cuda_kernel(float4* pos, unsigned int mesh_width, unsigned int mesh_height, float time) {

    //Code
    dim3 block = dim3(8, 8, 1);
    dim3 grid = dim3(mesh_width / block.x, mesh_height / block.y, 1);

    sine_wave_kernel << <grid, block >> > (pos, mesh_width, mesh_height, time);
}
