#pragma once
#include <Windows.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_d3d11_interop.h>
#include <unordered_map>

class CUDACompute {
private:
	FILE* log_file;
	int device_id;
	int device_count;
	cudaGraphicsResource* graphics_resource;

public:
	CUDACompute();
	int get_device_count();
	std::unordered_map<int, std::string> get_devices();
	int get_device_id();
	void set_device_id(int id);
	void register_GL_graphics_resource(unsigned int buffer);
	void register_D3D11_graphics_resource(ID3D11Buffer* p_buffer);
	void map_resoucre();
	float4* get_mapped_pointer();
	void unmap_resoucre();
	void unregister_graphics_resource();
	~CUDACompute();
};