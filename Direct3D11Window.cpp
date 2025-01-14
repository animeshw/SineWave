#include "Direct3D11Window.hpp"
#include "SineWave.cuh"

Direct3D11Window::Direct3D11Window(HWND hwnd_parent) : MyWindow(), swap_chain(NULL), device(NULL), device_context(NULL),
                        render_target_view(NULL), vertex_shader(NULL), pixel_shader(NULL), vertex_position_buffer(NULL),
                        input_layout(NULL), constant_buffer(NULL), rasterizer_state(NULL), depth_stencil_view(NULL) {
    persp_proj_matrix = DirectX::XMMatrixIdentity();

    point_color[0] = 0.0f;
    point_color[1] = 0.0f;
    point_color[2] = 0.0f;
    point_color[3] = 1.0f;

	if (fopen_s(&log_file, "DirectX11_Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Log file cannot be opened!"), TEXT("ERROR!"), MB_OK | MB_ICONERROR | MB_TOPMOST);
		DestroyWindow(hwnd);
	}

	int x_centre = (GetSystemMetrics(SM_CXSCREEN) / 2) - (RENDER_WIDTH / 2);
	int y_centre = (GetSystemMetrics(SM_CYSCREEN) / 2) - (RENDER_HEIGHT / 2);

	WNDCLASSEX wndclass;
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbWndExtra = 0;
	wndclass.cbClsExtra = 0;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = TEXT("DirectX");
	wndclass.lpszMenuName = nullptr;
	wndclass.hInstance = GetModuleHandle(NULL);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(NULL, IDC_ICON);
	wndclass.hIconSm = LoadIcon(NULL, IDC_ICON);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, TEXT("DirectX"), TEXT("Sine Wave Simulation - DirectX11"),
		WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_OVERLAPPEDWINDOW, x_centre, y_centre,
		RENDER_WIDTH, RENDER_HEIGHT, hwnd_parent, nullptr, GetModuleHandle(NULL), nullptr);

	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc;
	ZeroMemory((void*)&dxgiSwapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
	dxgiSwapChainDesc.BufferCount = 1;
	dxgiSwapChainDesc.BufferDesc.Width = RENDER_WIDTH;
	dxgiSwapChainDesc.BufferDesc.Height = RENDER_HEIGHT;
	dxgiSwapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Numerator = 120;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	dxgiSwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	dxgiSwapChainDesc.OutputWindow = hwnd;
	dxgiSwapChainDesc.SampleDesc.Count = 1;
	dxgiSwapChainDesc.SampleDesc.Quality = 0;
	dxgiSwapChainDesc.Windowed = TRUE;

	D3D_FEATURE_LEVEL d3dFeatureLevel_required = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeatureLevel_acquired = D3D_FEATURE_LEVEL_10_0;
	D3D_DRIVER_TYPE d3dDriverType;
	D3D_DRIVER_TYPE d3dDriverTypes[] = { D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP, D3D_DRIVER_TYPE_REFERENCE };
	UINT numberOfFeatureLevels = 1;
	UINT numberOfDriverTypes = sizeof(d3dDriverTypes) / sizeof(d3dDriverTypes[0]);
	UINT createDeviceFlags = 0;

	HRESULT hr;
	for (UINT driverTypeIndex = 0; driverTypeIndex < numberOfDriverTypes; driverTypeIndex++) {
		d3dDriverType = d3dDriverTypes[driverTypeIndex];
		hr = D3D11CreateDeviceAndSwapChain(NULL, d3dDriverType, NULL, createDeviceFlags, &d3dFeatureLevel_required,
						numberOfFeatureLevels, D3D11_SDK_VERSION, &dxgiSwapChainDesc, &swap_chain, &device,
						&d3dFeatureLevel_acquired, &device_context);

		if (SUCCEEDED(hr))
			break;
	}

	if (FAILED(hr)) {
		fprintf_s(log_file, "D3D11CreateDeviceAndSwapChain() failed\n");
        DestroyWindow(hwnd);
	}
}

void Direct3D11Window::initialize(int compute_api) {
    compute = compute_api;
    const char* vertexShaderSourceCode =
        "cbuffer ConstantBuffer{" \
        "   float4x4 worldViewProjectionMatrix;" \
        "   float4 out_color;" \
        "}" \
        "struct vertex_output{" \
        "   float4 position : SV_POSITION;" \
        "   float4 color : COLOR;" \
        "};" \
        "vertex_output main(float4 pos : POSITION){" \
        "   vertex_output output;" \
        "   output.position = mul(worldViewProjectionMatrix, pos);" \
        "   output.color = out_color;" \
        "   return(output);" \
        "}";

    ID3DBlob* pID3DBlob_VertexShaderCode = NULL;
    ID3DBlob* pID3DBlob_Error = NULL;

    HRESULT hr = D3DCompile(vertexShaderSourceCode,
        lstrlenA(vertexShaderSourceCode) + 1,
        "VS",
        NULL,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main",
        "vs_5_0",
        0,
        0,
        &pID3DBlob_VertexShaderCode,
        &pID3DBlob_Error);

    if (FAILED(hr)) {
        if (pID3DBlob_Error != NULL) {
            fprintf_s(log_file, "D3DCompile() failed for Vertex Shader : %s\n", (char*)pID3DBlob_Error->GetBufferPointer());
            pID3DBlob_Error->Release();
            pID3DBlob_Error = NULL;
            DestroyWindow(hwnd);
        }
        else {
            fprintf_s(log_file, "COM Error\n");
            DestroyWindow(hwnd);
        }
    }

    hr = device->CreateVertexShader(pID3DBlob_VertexShaderCode->GetBufferPointer(),
        pID3DBlob_VertexShaderCode->GetBufferSize(),
        NULL,
        &vertex_shader);
    if (FAILED(hr)) {
        fprintf_s(log_file, "ID3D11Device::CreateVertexShader() failed\n");
        DestroyWindow(hwnd);
    }

    device_context->VSSetShader(vertex_shader, 0, 0);

    const char* pixelShaderSourceCode =
        "struct vertex_output{" \
        "   float4 position : SV_POSITION;" \
        "   float4 color : COLOR;" \
        "};" \
        "float4 main(vertex_output input) : SV_TARGET{"\
        "   return(input.color);" \
        "}";

    ID3DBlob* pID3DBlob_PixelShaderCode = NULL;
    pID3DBlob_Error = NULL;

    hr = D3DCompile(pixelShaderSourceCode,
        lstrlenA(pixelShaderSourceCode) + 1,
        "PS",
        NULL,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main",
        "ps_5_0",
        0,
        0,
        &pID3DBlob_PixelShaderCode,
        &pID3DBlob_Error);

    if (FAILED(hr)) {
        if (pID3DBlob_Error != NULL) {
            fprintf_s(log_file, "D3DCompile() failed for Pixel Shader : %s\n", (char*)pID3DBlob_Error->GetBufferPointer());
            pID3DBlob_Error->Release();
            pID3DBlob_Error = NULL;
            DestroyWindow(hwnd);
        }
        else {
            fprintf_s(log_file, "COM Error\n");
            DestroyWindow(hwnd);
        }
    }

    hr = device->CreatePixelShader(pID3DBlob_PixelShaderCode->GetBufferPointer(),
        pID3DBlob_PixelShaderCode->GetBufferSize(),
        NULL,
        &pixel_shader);
    if (FAILED(hr)) {
        fprintf_s(log_file, "ID3D11Device::CreatePixelShader() failed\n");
        DestroyWindow(hwnd);
    }

    device_context->PSSetShader(pixel_shader, 0, 0);

    if (pID3DBlob_PixelShaderCode) {
        pID3DBlob_PixelShaderCode->Release();
        pID3DBlob_PixelShaderCode = NULL;
    }

    if (pID3DBlob_Error) {
        pID3DBlob_Error->Release();
        pID3DBlob_Error = NULL;
    }

    D3D11_INPUT_ELEMENT_DESC d3d11InputElementDesc[1];

    d3d11InputElementDesc[0].SemanticName = "POSITION";
    d3d11InputElementDesc[0].SemanticIndex = 0;
    d3d11InputElementDesc[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
    d3d11InputElementDesc[0].AlignedByteOffset = 0;
    d3d11InputElementDesc[0].InputSlot = 0;
    d3d11InputElementDesc[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
    d3d11InputElementDesc[0].InstanceDataStepRate = 0;

    hr = device->CreateInputLayout(d3d11InputElementDesc,
        _ARRAYSIZE(d3d11InputElementDesc),
        pID3DBlob_VertexShaderCode->GetBufferPointer(),
        pID3DBlob_VertexShaderCode->GetBufferSize(),
        &input_layout);

    if (FAILED(hr)) {
        fprintf_s(log_file, "ID3D11Device::CreateInputLayout() failed\n");
        if (pID3DBlob_VertexShaderCode) {
            pID3DBlob_VertexShaderCode->Release();
            pID3DBlob_VertexShaderCode = NULL;
        }
        DestroyWindow(hwnd);
    }
    else {
        if (pID3DBlob_VertexShaderCode) {
            pID3DBlob_VertexShaderCode->Release();
            pID3DBlob_VertexShaderCode = NULL;
        }
    }

    device_context->IASetInputLayout(input_layout);

    D3D11_BUFFER_DESC d3d11bufferDesc;
    ZeroMemory((void*)&d3d11bufferDesc, sizeof(D3D11_BUFFER_DESC));
    d3d11bufferDesc.ByteWidth = sizeof(float) * MESH_SIZE;
    d3d11bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    d3d11bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    d3d11bufferDesc.Usage = D3D11_USAGE_DYNAMIC;

    hr = device->CreateBuffer(&d3d11bufferDesc, NULL, &vertex_position_buffer);

    if (FAILED(hr)) {
        fprintf_s(log_file, "ID3D11Device::CreateBuffer() failed for vertex buffer - position\n");
        DestroyWindow(hwnd);
    }

    if (compute == CUDA) {
        cuda_compute.set_device_id(0);
        cuda_compute.register_D3D11_graphics_resource(vertex_position_buffer);
    }
    else if (compute == OPENCL) {
        opencl_compute.get_devices(NVIDIA_PLATFORM);
        opencl_compute.set_device_id(0);
        opencl_compute.create_D3D11_context(device);
        opencl_compute.register_D3D11_graphics_resource(vertex_position_buffer);
    }

    ZeroMemory((void*)&d3d11bufferDesc, sizeof(D3D11_BUFFER_DESC));
    d3d11bufferDesc.ByteWidth = sizeof(CBUFFER);
    d3d11bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    d3d11bufferDesc.CPUAccessFlags = 0;
    d3d11bufferDesc.Usage = D3D11_USAGE_DEFAULT;

    hr = device->CreateBuffer(&d3d11bufferDesc, NULL, &constant_buffer);

    if (FAILED(hr)) {
        fprintf_s(log_file, "ID3D11Device::CreateBuffer() failed for constant buffer\n");
        DestroyWindow(hwnd);
    }

    device_context->VSSetConstantBuffers(0, 1, &constant_buffer);

    D3D11_RASTERIZER_DESC d3d11RasterizerDesc;
    ZeroMemory((void*)&d3d11RasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));
    d3d11RasterizerDesc.AntialiasedLineEnable = FALSE;
    d3d11RasterizerDesc.CullMode = D3D11_CULL_NONE;
    d3d11RasterizerDesc.DepthBias = 0;
    d3d11RasterizerDesc.DepthBiasClamp = 0.0f;
    d3d11RasterizerDesc.DepthClipEnable = TRUE;
    d3d11RasterizerDesc.FillMode = D3D11_FILL_SOLID;
    d3d11RasterizerDesc.FrontCounterClockwise = FALSE;
    d3d11RasterizerDesc.MultisampleEnable = FALSE;
    d3d11RasterizerDesc.ScissorEnable = FALSE;
    d3d11RasterizerDesc.SlopeScaledDepthBias = 0.0f;

    hr = device->CreateRasterizerState(&d3d11RasterizerDesc, &rasterizer_state);
    if (FAILED(hr)) {
        fprintf_s(log_file, "ID3D11Device::CreateRasterizerState() failed\n");
        DestroyWindow(hwnd);
    }

    device_context->RSSetState(rasterizer_state);

    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)this);
}

void Direct3D11Window::show() {
    resize(RENDER_WIDTH, RENDER_HEIGHT);
    ShowWindow(hwnd, SW_SHOW);
    SetFocus(hwnd);
    SetForegroundWindow(hwnd);
}

void Direct3D11Window::resize(int width, int height) {
    if (height <= 0)
        height = 1;

    HRESULT hr = S_OK;

    if (render_target_view) {
        render_target_view->Release();
        render_target_view = NULL;
    }

    if (depth_stencil_view){
        depth_stencil_view->Release();
        depth_stencil_view = NULL;
    }

    swap_chain->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

    ID3D11Texture2D* pID3D11Texture2D_BackBuffer = NULL;
    swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pID3D11Texture2D_BackBuffer);

    hr = device->CreateRenderTargetView(pID3D11Texture2D_BackBuffer, NULL, &render_target_view);
    if (FAILED(hr)) {
        fprintf_s(log_file, "CreateRenderTargetView() failed\n");
        DestroyWindow(hwnd);
    }

    pID3D11Texture2D_BackBuffer->Release();
    pID3D11Texture2D_BackBuffer = NULL;

    D3D11_TEXTURE2D_DESC d3d11Texture2DDesc;
    ZeroMemory((void*)&d3d11Texture2DDesc, sizeof(D3D11_TEXTURE2D_DESC));

    d3d11Texture2DDesc.Width = (UINT)width;
    d3d11Texture2DDesc.Height = (UINT)height;
    d3d11Texture2DDesc.Format = DXGI_FORMAT_D32_FLOAT;
    d3d11Texture2DDesc.Usage = D3D11_USAGE_DEFAULT;
    d3d11Texture2DDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    d3d11Texture2DDesc.SampleDesc.Count = 1;
    d3d11Texture2DDesc.SampleDesc.Quality = 0;
    d3d11Texture2DDesc.ArraySize = 1;
    d3d11Texture2DDesc.MipLevels = 1;
    d3d11Texture2DDesc.CPUAccessFlags = 0;
    d3d11Texture2DDesc.MiscFlags = 0;

    ID3D11Texture2D* pID3D11Texture2D_DepthBuffer = NULL;

    hr = device->CreateTexture2D(&d3d11Texture2DDesc, NULL, &pID3D11Texture2D_DepthBuffer);
    if (FAILED(hr)) {
        fprintf_s(log_file, "ID3D11Device::CreateTexture2D() failed\n");
        DestroyWindow(hwnd);
    }

    D3D11_DEPTH_STENCIL_VIEW_DESC d3d11DepthStencilViewDesc;
    ZeroMemory((void*)&d3d11DepthStencilViewDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));

    d3d11DepthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
    d3d11DepthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;

    hr = device->CreateDepthStencilView(pID3D11Texture2D_DepthBuffer, &d3d11DepthStencilViewDesc, &depth_stencil_view);
    if (FAILED(hr)) {
        fprintf_s(log_file, "ID3D11Device::CreateDepthStencilView() failed\n");
        DestroyWindow(hwnd);
    }

    device_context->OMSetRenderTargets(1, &render_target_view, depth_stencil_view);

    D3D11_VIEWPORT d3dViewPort;
    ZeroMemory((void*)&d3dViewPort, sizeof(D3D11_VIEWPORT));
    d3dViewPort.TopLeftX = 0;
    d3dViewPort.TopLeftY = 0;
    d3dViewPort.Width = (float)width;
    d3dViewPort.Height = (float)height;
    d3dViewPort.MinDepth = 0.0f;
    d3dViewPort.MaxDepth = 1.0f;
    device_context->RSSetViewports(1, &d3dViewPort);

    persp_proj_matrix = DirectX::XMMatrixPerspectiveFovLH(DirectX::XMConvertToRadians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
}

void Direct3D11Window::render() {
    device_context->ClearRenderTargetView(render_target_view, point_color);
    device_context->ClearDepthStencilView(depth_stencil_view, D3D11_CLEAR_DEPTH, 1.0f, 0);

    UINT stride = sizeof(float) * 4;
    UINT offset = 0;
    if (compute == CUDA) {
        cuda_compute.map_resoucre();
        float4* pPos = cuda_compute.get_mapped_pointer();
        launch_cuda_kernel(pPos, MESH_WIDTH, MESH_HEIGHT, speed);
        cuda_compute.unmap_resoucre();
    }
    else if (compute == OPENCL) {
        opencl_compute.set_kernel_args(MESH_WIDTH, MESH_HEIGHT, speed);
        opencl_compute.map_D3D11_resoucre();

        size_t globalWorkSize[2];
        globalWorkSize[0] = MESH_WIDTH;
        globalWorkSize[1] = MESH_HEIGHT;

        opencl_compute.launch_kernel(globalWorkSize);

        opencl_compute.unmap_D3D11_resoucre();
        opencl_compute.finish();
    }
    device_context->IASetVertexBuffers(0, 1, &vertex_position_buffer, &stride, &offset);

    device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

    DirectX::XMMATRIX translationMatrix = DirectX::XMMatrixTranslation(0.0f, 0.0f, 0.1f);
    DirectX::XMMATRIX worldMatrix = translationMatrix;
    DirectX::XMMATRIX viewMatrix = DirectX::XMMatrixIdentity();
    DirectX::XMMATRIX wvpMatrix = worldMatrix * viewMatrix * persp_proj_matrix;

    CBUFFER constantBuffer;
    ZeroMemory((void*)&constantBuffer, sizeof(CBUFFER));
    constantBuffer.world_view_proj_matrix = wvpMatrix;
    constantBuffer.point_color = DirectX::XMVECTOR{ 0.6f, 0.3f, 0.6f, 1.0f };

    device_context->UpdateSubresource(constant_buffer, 0, NULL, &constantBuffer, 0, 0);

    device_context->Draw(MESH_WIDTH * MESH_HEIGHT, 0);

    swap_chain->Present(1, 0);
}

void Direct3D11Window::update() {
    speed += 0.01f;
}

CUDACompute& Direct3D11Window::get_cuda_compute() {
    return cuda_compute;
}

OpenCLCompute& Direct3D11Window::get_opencl_compute() {
    return opencl_compute;
}

void Direct3D11Window::uninitialize() {
    if (is_fullscreen) {
        dw_style = GetWindowLong(hwnd, GWL_STYLE);
        SetWindowLong(hwnd, GWL_STYLE, dw_style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(hwnd, &wp_prev);
        SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        ShowCursor(TRUE);
    }

    if (depth_stencil_view) {
        depth_stencil_view->Release();
        depth_stencil_view = NULL;
    }

    if (rasterizer_state) {
        rasterizer_state->Release();
        rasterizer_state = NULL;
    }

    if (constant_buffer) {
        constant_buffer->Release();
        constant_buffer = NULL;
    }

    if (input_layout) {
        input_layout->Release();
        input_layout = NULL;
    }

    if (vertex_position_buffer) {
        vertex_position_buffer->Release();
        vertex_position_buffer = NULL;
    }

    if (pixel_shader) {
        pixel_shader->Release();
        pixel_shader = NULL;
    }

    if (vertex_shader) {
        vertex_shader->Release();
        vertex_shader = NULL;
    }

    if (render_target_view) {
        render_target_view->Release();
        render_target_view = NULL;
    }

	if (swap_chain) {
		swap_chain->Release();
		swap_chain = NULL;
	}

	if (device_context) {
		device_context->Release();
		device_context = NULL;
	}

	if (device) {
		device->Release();
		device = NULL;
	}

	if (log_file) {
		fclose(log_file);
		log_file = NULL;
	}
}

LRESULT Direct3D11Window::WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {
	Direct3D11Window* pDX11Wnd = (Direct3D11Window*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
	switch (iMsg) {
    case WM_SETFOCUS:
        pDX11Wnd->is_active = true;
        break;

    case WM_KILLFOCUS:
        pDX11Wnd->is_active = false;
        break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_DESTROY:
		if (pDX11Wnd)
			pDX11Wnd->uninitialize();
		break;

    case WM_SIZE:
        pDX11Wnd->resize(LOWORD(lParam), HIWORD(lParam));
        break;

    case WM_KEYDOWN:
        switch (wParam) {

        case VK_ESCAPE:
            DestroyWindow(hwnd);
            break;

        case 0x46:
        case 0x66:
            pDX11Wnd->toggle_fullscreen();
            break;

        default:
            break;
        }
        break;

	default:
		break;
	}

	return DefWindowProc(hwnd, iMsg, wParam, lParam);
}