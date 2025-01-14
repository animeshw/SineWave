#pragma once
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include <CL/cl_d3d11.h>
#include <CL/cl_d3d11_ext.h>
#include <unordered_map>

class OpenCLCompute {
private:
	FILE* log_file;
	cl_uint platform_count;
	cl_uint device_count;
	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem graphics_resource;
	char* source_code;

public:
	OpenCLCompute();
	std::unordered_map<cl_uint, std::string> get_platforms();
	std::unordered_map<cl_uint, std::string> get_devices(cl_uint platform);
	void set_device_id(cl_uint device);
	char* loadOCLProgramSource(const char* filename, const char* preamble, size_t* size_final_length);
	void create_kernel();
	void create_D3D11_context(ID3D11Device* p_device);
	void create_GL_context();
	void register_GL_graphics_resource(unsigned int buffer);
	void register_D3D11_graphics_resource(ID3D11Buffer* p_buffer);
	void set_kernel_args(int width, int height, float speed);
	void map_GL_resoucre();
	void unmap_GL_resoucre();
	void map_D3D11_resoucre();
	void unmap_D3D11_resoucre();
	void launch_kernel(size_t* global_work_size);
	void unregister_graphics_resource();
	void finish();
	~OpenCLCompute();
};