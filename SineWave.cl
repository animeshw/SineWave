__kernel void sine_wave_kernel(__global float4* pos, unsigned int mesh_width, unsigned int mesh_height, float time) {

    //Code
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    float u = x / (float)mesh_width;
    float v = y / (float)mesh_height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    float frequency = 4.0f;
    float w = sin(u * frequency + time) * cos(v * frequency + time) * 0.5f;

    pos[y * mesh_width + x] = (float4)(u, w, v, 1.0f);
}