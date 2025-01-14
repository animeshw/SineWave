#pragma once
#include "MyWindow.hpp"
#include <DirectXMath.h>
//#include "XNAMath/xnamath.h"

struct CBUFFER {
	DirectX::XMMATRIX world_view_proj_matrix;
	DirectX::XMVECTOR point_color;
};

class Direct3D11Window : public MyWindow {
private:
	IDXGISwapChain* swap_chain;
	ID3D11Device* device;
	ID3D11DeviceContext* device_context;
	ID3D11RenderTargetView* render_target_view;
	ID3D11VertexShader* vertex_shader;
	ID3D11PixelShader* pixel_shader;
	ID3D11Buffer* vertex_position_buffer;
	ID3D11InputLayout* input_layout;
	ID3D11Buffer* constant_buffer;
	ID3D11RasterizerState* rasterizer_state;
	ID3D11DepthStencilView* depth_stencil_view;

	DirectX::XMMATRIX persp_proj_matrix;
	float point_color[4];

	CUDACompute cuda_compute;
	OpenCLCompute opencl_compute;

public:
	Direct3D11Window(HWND hwnd_parent);
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
