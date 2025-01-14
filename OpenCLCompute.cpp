#include "OpenCLCompute.hpp"

OpenCLCompute::OpenCLCompute() : log_file(nullptr), platform_count(0), device_count(0), platform_id(nullptr), device_id(nullptr),
				context(nullptr), command_queue(nullptr), program(nullptr), kernel(nullptr), graphics_resource(nullptr), source_code(nullptr){
	if (fopen_s(&log_file, "OpenCL_Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("OpenCL Log file cannot be opened!"), TEXT("ERROR!"), MB_OK | MB_ICONERROR | MB_TOPMOST);
		exit(EXIT_FAILURE);
	}

	cl_int ret_val = clGetPlatformIDs(0, NULL, &platform_count);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clGetPlatformIDs() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

std::unordered_map<cl_uint, std::string> OpenCLCompute::get_platforms() {
	cl_platform_id* platform_ids;
	std::unordered_map<cl_uint, std::string> platforms;

	platform_ids = (cl_platform_id*)malloc(platform_count * sizeof(cl_platform_id));
	cl_int ret_val = clGetPlatformIDs(platform_count, platform_ids, NULL);
	char info[512];

	for (cl_uint i = 0; i < platform_count; ++i) {
		ret_val = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(info), info, NULL);
		if (ret_val != CL_SUCCESS) {
			fprintf(log_file, "clGetPlatformInfo() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
			fclose(log_file);
			exit(EXIT_FAILURE);
		}
		platforms[i] = info;
	}
	free(platform_ids);
	return platforms;
}

std::unordered_map<cl_uint, std::string> OpenCLCompute::get_devices(cl_uint platform) {
	cl_platform_id* platform_ids;
	cl_device_id* device_ids;
	std::unordered_map<cl_uint, std::string> devices;

	platform_ids = (cl_platform_id*)malloc(platform_count * sizeof(cl_platform_id));
	cl_int ret_val = clGetPlatformIDs(platform_count, platform_ids, NULL);

	platform_id = platform_ids[platform];

	ret_val = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clGetDeviceIDs() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}

	device_ids = (cl_device_id*)malloc(device_count * sizeof(cl_device_id));
	ret_val = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, device_count, device_ids, NULL);
	char info[512];

	for (cl_uint i = 0; i < device_count; ++i) {
		ret_val = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(info), info, NULL);
		if (ret_val != CL_SUCCESS) {
			fprintf(log_file, "clGetDeviceInfo() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
			fclose(log_file);
			exit(EXIT_FAILURE);
		}
		devices[i] = info;
	}
	free(platform_ids);
	free(device_ids);
	return devices;
}

void OpenCLCompute::set_device_id(cl_uint device) {
	cl_device_id* device_ids = (cl_device_id*)malloc(device_count * sizeof(cl_device_id));

	cl_int ret_val = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, device_count, device_ids, NULL);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clGetDeviceIDs() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}

	device_id = device_ids[device];
	free(device_ids);
}

char* OpenCLCompute::loadOCLProgramSource(const char* filename, const char* preamble, size_t* size_final_length) {
	FILE* pFile = NULL;
	size_t sizeSourceLength;

	fopen_s(&pFile, filename, "rb");
	if (pFile == NULL)
		return NULL;

	size_t sizePreambleLength = (size_t)strlen(preamble);

	fseek(pFile, 0, SEEK_END);
	sizeSourceLength = ftell(pFile);
	fseek(pFile, 0, SEEK_SET);

	char* sourceString = (char*)malloc(sizeSourceLength + sizePreambleLength + 1);
	memcpy(sourceString, preamble, sizePreambleLength);
	if (fread((sourceString)+sizePreambleLength, sizeSourceLength, 1, pFile) != 1) {
		fclose(pFile);
		free(sourceString);
		return(0);
	}

	fclose(pFile);
	if (size_final_length != 0) {
		*size_final_length = sizeSourceLength + sizePreambleLength;
	}
	sourceString[sizeSourceLength + sizePreambleLength] = '\0';

	return(sourceString);
}

void OpenCLCompute::create_kernel() {
	cl_int ret_val;
	command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret_val);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clCreateCommandQueueWithProperties() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}

	size_t kernel_code_length;
	source_code = loadOCLProgramSource("SineWave.cl", "", &kernel_code_length);
	program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &kernel_code_length, &ret_val);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clCreateProgramWithSource() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}

	ret_val = clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clBuildProgram() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(log_file, "OpenCL Program Build Log : %s\n", buffer);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}

	kernel = clCreateKernel(program, "sine_wave_kernel", &ret_val);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clCreateKernel() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

void OpenCLCompute::create_D3D11_context(ID3D11Device* p_device) {
	cl_context_properties context_properties[] = { CL_CONTEXT_D3D11_DEVICE_NV, (cl_context_properties)p_device,
													CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,
													0 };
	cl_int ret_val;
	context = clCreateContext(context_properties, 1, &device_id, NULL, NULL, &ret_val);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clCreateContext() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
	if (context == nullptr) {
		cl_context_properties context_properties[] = { CL_CONTEXT_D3D11_DEVICE_KHR, (cl_context_properties)p_device,
														CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,
														CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
														0 };
		context = clCreateContext(context_properties, 1, &device_id, NULL, NULL, &ret_val);
		if (ret_val != CL_SUCCESS) {
			fprintf(log_file, "clCreateContext() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
			fclose(log_file);
			exit(EXIT_FAILURE);
		}
	}
	create_kernel();
}

void OpenCLCompute::create_GL_context() {
	cl_context_properties ocl_context_properties[] = { CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
														CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
														CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,
														0 };

	cl_int ret_val;
	context = clCreateContext(ocl_context_properties, 1, &device_id, NULL, NULL, &ret_val);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clCreateContext() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
	create_kernel();
}

void OpenCLCompute::register_GL_graphics_resource(unsigned int buffer) {
	unregister_graphics_resource();
	cl_int ret_val;
	graphics_resource = clCreateFromGLBuffer(context, CL_MEM_WRITE_ONLY, buffer, &ret_val);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clCreateContext() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

void OpenCLCompute::register_D3D11_graphics_resource(ID3D11Buffer* p_buffer) {
	unregister_graphics_resource();
	cl_int ret_val;
	clCreateFromD3D11BufferNV_fn ptrToFunction_clCreateFromD3D11BufferNV = NULL;
	ptrToFunction_clCreateFromD3D11BufferNV = (clCreateFromD3D11BufferNV_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clCreateFromD3D11BufferNV");
	if (ptrToFunction_clCreateFromD3D11BufferNV != NULL) {
		graphics_resource = ptrToFunction_clCreateFromD3D11BufferNV(context, CL_MEM_WRITE_ONLY, p_buffer, &ret_val);
		if (ret_val != CL_SUCCESS) {
			fprintf(log_file, "clCreateFromD3D11BufferNV() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
			fclose(log_file);
			exit(EXIT_FAILURE);
		}
	}
	else {
		clCreateFromD3D11BufferKHR_fn ptrToFunction_clCreateFromD3D11BufferKHR = NULL;
		ptrToFunction_clCreateFromD3D11BufferKHR = (clCreateFromD3D11BufferKHR_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clCreateFromD3D11BufferKHR");
		if (ptrToFunction_clCreateFromD3D11BufferKHR != NULL) {
			graphics_resource = ptrToFunction_clCreateFromD3D11BufferKHR(context, CL_MEM_WRITE_ONLY, p_buffer, &ret_val);
			if (ret_val != CL_SUCCESS) {
				fprintf(log_file, "clCreateFromD3D11BufferKHR() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
				fclose(log_file);
				exit(EXIT_FAILURE);
			}
		}
		else {
			fprintf(log_file, "clGetExtensionFunctionAddressForPlatform() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
			fclose(log_file);
			exit(EXIT_FAILURE);
		}
	}
}

void OpenCLCompute::set_kernel_args(int width, int height, float speed) {
	cl_int ret_val = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&graphics_resource);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clSetKernelArg() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}

	ret_val = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&width);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clSetKernelArg() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}

	ret_val = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&height);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clSetKernelArg() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}

	ret_val = clSetKernelArg(kernel, 3, sizeof(cl_float), (void*)&speed);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clSetKernelArg() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

void OpenCLCompute::map_GL_resoucre() {
	cl_int ret_val = clEnqueueAcquireGLObjects(command_queue, 1, &graphics_resource, 0, NULL, NULL);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clEnqueueAcquireGLObjects() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

void OpenCLCompute::unmap_GL_resoucre() {
	cl_int ret_val = clEnqueueReleaseGLObjects(command_queue, 1, &graphics_resource, 0, NULL, NULL);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clEnqueueReleaseGLObjects() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

void OpenCLCompute::map_D3D11_resoucre() {
	cl_int ret_val;
	clEnqueueAcquireD3D11ObjectsNV_fn ptrToFunction_clEnqueueAcquireD3D11ObjectsNV = NULL;
	ptrToFunction_clEnqueueAcquireD3D11ObjectsNV = (clEnqueueAcquireD3D11ObjectsNV_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clEnqueueAcquireD3D11ObjectsNV");
	if (ptrToFunction_clEnqueueAcquireD3D11ObjectsNV != NULL) {
		ret_val = ptrToFunction_clEnqueueAcquireD3D11ObjectsNV(command_queue, 1, &graphics_resource, 0, NULL, NULL);
		if (ret_val != CL_SUCCESS) {
			fprintf(log_file, "clEnqueueAcquireD3D11ObjectsNV() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
			fclose(log_file);
			exit(EXIT_FAILURE);
		}
	}
	else {
		clEnqueueAcquireD3D11ObjectsKHR_fn ptrToFunction_clEnqueueAcquireD3D11ObjectsKHR = NULL;
		ptrToFunction_clEnqueueAcquireD3D11ObjectsKHR = (clEnqueueAcquireD3D11ObjectsKHR_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clEnqueueAcquireD3D11ObjectsKHR");
		if (ptrToFunction_clEnqueueAcquireD3D11ObjectsKHR != NULL) {
			ret_val = ptrToFunction_clEnqueueAcquireD3D11ObjectsKHR(command_queue, 1, &graphics_resource, 0, NULL, NULL);
			if (ret_val != CL_SUCCESS) {
				fprintf(log_file, "clEnqueueAcquireD3D11ObjectsKHR() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
				fclose(log_file);
				exit(EXIT_FAILURE);
			}
		}
	}
}

void OpenCLCompute::unmap_D3D11_resoucre() {
	cl_int ret_val;
	clEnqueueReleaseD3D11ObjectsNV_fn ptrToFunction_clEnqueueReleaseD3D11ObjectsNV = NULL;
	ptrToFunction_clEnqueueReleaseD3D11ObjectsNV = (clEnqueueReleaseD3D11ObjectsNV_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clEnqueueReleaseD3D11ObjectsNV");
	if (ptrToFunction_clEnqueueReleaseD3D11ObjectsNV != NULL) {
		ret_val = ptrToFunction_clEnqueueReleaseD3D11ObjectsNV(command_queue, 1, &graphics_resource, 0, NULL, NULL);
		if (ret_val != CL_SUCCESS) {
			fprintf(log_file, "clEnqueueReleaseD3D11ObjectsNV() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
			fclose(log_file);
			exit(EXIT_FAILURE);
		}
	}
	else {
		clEnqueueReleaseD3D11ObjectsKHR_fn ptrToFunction_clEnqueueReleaseD3D11ObjectsKHR = NULL;
		ptrToFunction_clEnqueueReleaseD3D11ObjectsKHR = (clEnqueueReleaseD3D11ObjectsKHR_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clEnqueueReleaseD3D11ObjectsKHR");
		if (ptrToFunction_clEnqueueReleaseD3D11ObjectsKHR != NULL) {
			ret_val = ptrToFunction_clEnqueueReleaseD3D11ObjectsKHR(command_queue, 1, &graphics_resource, 0, NULL, NULL);
			if (ret_val != CL_SUCCESS) {
				fprintf(log_file, "clEnqueueReleaseD3D11ObjectsKHR() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
				fclose(log_file);
				exit(EXIT_FAILURE);
			}
		}
	}
}

void OpenCLCompute::launch_kernel(size_t* global_work_size) {
	cl_int ret_val = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
	if (ret_val != CL_SUCCESS) {
		fprintf(log_file, "clEnqueueNDRangeKernel() failed : %d, %s, %d \n", ret_val, __FILE__, __LINE__);
		fclose(log_file);
		exit(EXIT_FAILURE);
	}
}

void OpenCLCompute::unregister_graphics_resource() {
	if (graphics_resource) {
		clReleaseMemObject(graphics_resource);
		graphics_resource = nullptr;
	}
}

void OpenCLCompute::finish() { clFinish(command_queue); }

OpenCLCompute::~OpenCLCompute() {
	unregister_graphics_resource();

	if (source_code) {
		free(source_code);
		source_code = nullptr;
	}

	if (kernel) {
		clReleaseKernel(kernel);
		kernel = nullptr;
	}

	if (program) {
		clReleaseProgram(program);
		program = nullptr;
	}

	if (command_queue) {
		clReleaseCommandQueue(command_queue);
		command_queue = nullptr;
	}

	if (context) {
		clReleaseContext(context);
		context = nullptr;
	}

	if (log_file) {
		fclose(log_file);
		log_file = nullptr;
	}
}