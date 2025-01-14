#pragma once
#include "MyWindow.hpp"
#include "vmath.h"

class OpenGLWindow : public MyWindow {
private:
	HDC hdc;
	HGLRC hglrc;
	GLuint shader_program;
	GLuint model_view_proj_matrix_uniform;
	GLuint vertex_array_object;
	GLuint vertex_buffer_object_position;
	vmath::mat4 persp_proj_matrix;

	CUDACompute cuda_compute;
	OpenCLCompute opencl_compute;

	enum {
		ATTRIBUTE_POSITION = 0,
		ATTRIBUTE_COLOR,
		ATTRIBUTE_NORMAL,
		ATTRIBUTE_TEXCORD
	};

public:
	OpenGLWindow(HWND hwnd_parent);
	void initialize(int compute_api);
	void show();
	void resize(int, int);
	void render();
	void update();
	void uninitialize();
	CUDACompute& get_cuda_compute();
	OpenCLCompute& get_opencl_compute();

protected:
	static LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
};