#include "CUDACompute.hpp"

CUDACompute::CUDACompute() : log_file(nullptr), device_id(-1), device_count(0), graphics_resource(nullptr) {
	if (fopen_s(&log_file, "CUDA_Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("CUDA Log file cannot be opened!"), TEXT("ERROR!"), MB_OK | MB_ICONERROR | MB_TOPMOST);
		exit(EXIT_FAILURE);
	}

	cudaError_t ret_val = cudaGetDeviceCount(&device_count);
	if (ret_val != cudaSuccess) {
		fprintf(log_file, "cudaGetDeviceCount() failed : %s, %s, %d \n", cudaGetErrorString(ret_val), __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

int CUDACompute::get_device_count() { return device_count; }

std::unordered_map<int, std::string> CUDACompute::get_devices() {
	cudaError_t ret_val;
	cudaDeviceProp dev_prop;
	std::unordered_map<int, std::string> devices;
	for (int i = 0; i < device_count; ++i) {
		ret_val = cudaGetDeviceProperties(&dev_prop, i);
		if (ret_val != cudaSuccess) {
			fprintf(log_file, "cudaGetDeviceProperties() failed for deivce %d : %s, %s, %d\n", i, cudaGetErrorString(ret_val), __FILE__, __LINE__);
			fclose(log_file);
			exit(EXIT_FAILURE);
			break;
		}
		devices[i] = dev_prop.name;
	}
	return devices;
}

int CUDACompute::get_device_id() { return device_id; }

void CUDACompute::set_device_id(int id) {
	device_id = id;
	cudaSetDevice(device_id);
}

void CUDACompute::register_GL_graphics_resource(unsigned int buffer) {
	unregister_graphics_resource();
	cudaError_t ret_val = cudaGraphicsGLRegisterBuffer(&graphics_resource, buffer, cudaGraphicsMapFlagsWriteDiscard);
	if (ret_val != cudaSuccess) {
		fprintf(log_file, "cudaGraphicsGLRegisterBuffer() failed : %s %s %d\n", cudaGetErrorString(ret_val), __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

void CUDACompute::register_D3D11_graphics_resource(ID3D11Buffer* p_buffer) {
	unregister_graphics_resource();
	cudaError_t ret_val = cudaGraphicsD3D11RegisterResource(&graphics_resource, p_buffer, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsResourceSetMapFlags(graphics_resource, cudaGraphicsMapFlagsWriteDiscard);
	if (ret_val != cudaSuccess) {
		fprintf(log_file, "cudaGraphicsD3D11RegisterResource() failed : %s %s %d\n", cudaGetErrorString(ret_val), __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

void CUDACompute::map_resoucre() {
	cudaError_t ret_val = cudaGraphicsMapResources(1, &graphics_resource, 0);
	if (ret_val != cudaSuccess) {
		fprintf(log_file, "cudaGraphicsMapResources() failed : %s %s %d\n", cudaGetErrorString(ret_val), __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

float4* CUDACompute::get_mapped_pointer() {
	float4* pPos;
	size_t numBytes;
	cudaError_t ret_val = cudaGraphicsResourceGetMappedPointer((void**)&pPos, &numBytes, graphics_resource);
	if (ret_val != cudaSuccess) {
		fprintf(log_file, "cudaGraphicsResourceGetMappedPointer() failed : %s %s %d\n", cudaGetErrorString(ret_val), __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
	return pPos;
}

void CUDACompute::unmap_resoucre() {
	cudaError_t ret_val = cudaGraphicsUnmapResources(1, &graphics_resource, 0);
	if (ret_val != cudaSuccess) {
		fprintf(log_file, "cudaGraphicsUnmapResources() failed : %s %s %d\n", cudaGetErrorString(ret_val), __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

void CUDACompute::unregister_graphics_resource() {
	if (graphics_resource) {
		cudaGraphicsUnregisterResource(graphics_resource);
		graphics_resource = nullptr;
	}
}

CUDACompute::~CUDACompute() {
	unregister_graphics_resource();
	if (log_file) {
		fclose(log_file);
		log_file = nullptr;
	}
}