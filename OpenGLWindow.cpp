#include "OpenGLWindow.hpp"
#include "SineWave.cuh"

OpenGLWindow::OpenGLWindow(HWND hwnd_parent) : MyWindow(), hdc(nullptr), hglrc(nullptr), shader_program(0), 
                model_view_proj_matrix_uniform(0), vertex_array_object(0), vertex_buffer_object_position(0){
    persp_proj_matrix = vmath::mat4::identity();

    if (fopen_s(&log_file, "OpenGL_Log.txt", "w") != 0) {
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
	wndclass.lpszClassName = TEXT("OpenGL");
	wndclass.lpszMenuName = nullptr;
	wndclass.hInstance = GetModuleHandle(NULL);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(NULL, IDC_ICON);
	wndclass.hIconSm = LoadIcon(NULL, IDC_ICON);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, TEXT("OpenGL"), TEXT("Sine Wave Simulation - OpenGL"),
		WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_OVERLAPPEDWINDOW, x_centre, y_centre,
		RENDER_WIDTH, RENDER_HEIGHT, hwnd_parent, nullptr, GetModuleHandle(NULL), nullptr);

    PIXELFORMATDESCRIPTOR pfd;
    ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cRedBits = 8;
    pfd.cGreenBits = 8;
    pfd.cBlueBits = 8;
    pfd.cAlphaBits = 8;
    pfd.cDepthBits = 32;

    hdc = GetDC(hwnd);

    int iPixelFormatIndex = ChoosePixelFormat(hdc, &pfd);
    if (iPixelFormatIndex == 0) {
        fprintf(log_file, "ChoosePixelFormat() failed.\n");
        DestroyWindow(hwnd);
    }

    if (SetPixelFormat(hdc, iPixelFormatIndex, &pfd) == FALSE) {
        fprintf(log_file, "SetPixelFormat() failed.\n");
        DestroyWindow(hwnd);
    }

    hglrc = wglCreateContext(hdc);
    if (hglrc == NULL) {
        fprintf(log_file, "wglCreateContext() failed.\n");
        DestroyWindow(hwnd);
    }

    if (wglMakeCurrent(hdc, hglrc) == FALSE) {
        fprintf(log_file, "wglMakeCurrent() failed.\n");
        DestroyWindow(hwnd);
    }

    GLenum glewError = glewInit();
    if (glewError != GLEW_OK) {
        fprintf(log_file, "glewInit() failed.\n");
        DestroyWindow(hwnd);
    }
    wglSwapIntervalEXT(1);
}

void OpenGLWindow::initialize(int compute_api) {
    compute = compute_api;
    GLuint vso = glCreateShader(GL_VERTEX_SHADER);
    const GLchar* vertexShaderSourceCode = "#version 450 core" \
        "\n" \
        "in vec4 vPosition;" \
        "in vec4 vColor;" \
        "uniform mat4 u_mvpMatrix;" \
        "out vec4 out_vColor;" \
        "void main(void)" \
        "{" \
        "gl_Position = u_mvpMatrix * vPosition;" \
        "out_vColor = vColor;" \
        "}";
    glShaderSource(vso, 1, (const char**)&vertexShaderSourceCode, NULL);

    glCompileShader(vso);

    GLint   infoLength = 0;
    GLint   shaderCompiledStatus = 0;
    GLchar* szInfoLog = NULL;

    glGetShaderiv(vso, GL_COMPILE_STATUS, &shaderCompiledStatus);
    if (shaderCompiledStatus == GL_FALSE) {
        glGetShaderiv(vso, GL_INFO_LOG_LENGTH, &infoLength);
        if (infoLength > 0) {
            szInfoLog = (GLchar*)malloc(sizeof(GLchar) * infoLength);
            if (szInfoLog != NULL) {
                GLsizei written;
                glGetShaderInfoLog(vso, infoLength, &written, szInfoLog);
                fprintf(log_file, "Vertex Shader Compilation Log : %s\n", szInfoLog);
                free(szInfoLog);
                DestroyWindow(hwnd);
            }
        }
    }

    GLuint fso = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar* fragmentShaderSourceCode = "#version 450 core" \
        "\n" \
        "in vec4 out_vColor;"
        "out vec4 FragColor;" \
        "void main(void)" \
        "{" \
        "FragColor = out_vColor;" \
        "}";
    glShaderSource(fso, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);

    glCompileShader(fso);
    glGetShaderiv(fso, GL_COMPILE_STATUS, &shaderCompiledStatus);
    if (shaderCompiledStatus == GL_FALSE) {
        glGetShaderiv(fso, GL_INFO_LOG_LENGTH, &infoLength);
        if (infoLength > 0) {
            szInfoLog = (GLchar*)malloc(sizeof(GLchar) * infoLength);
            if (szInfoLog != NULL) {
                GLsizei written;
                glGetShaderInfoLog(fso, infoLength, &written, szInfoLog);
                fprintf(log_file, "Fragment Shader Compilation Log : %s\n", szInfoLog);
                free(szInfoLog);
                DestroyWindow(hwnd);
            }
        }
    }

    shader_program = glCreateProgram();

    glAttachShader(shader_program, vso);
    glAttachShader(shader_program, fso);

    glBindAttribLocation(shader_program, ATTRIBUTE_POSITION, "vPosition");
    glBindAttribLocation(shader_program, ATTRIBUTE_COLOR, "vColor");

    glLinkProgram(shader_program);
    GLint shaderProgramLinkStatus = 0;
    glGetProgramiv(shader_program, GL_LINK_STATUS, &shaderProgramLinkStatus);
    if (shaderProgramLinkStatus == GL_FALSE) {
        glGetProgramiv(shader_program, GL_INFO_LOG_LENGTH, &infoLength);
        if (infoLength > 0) {
            szInfoLog = (GLchar*)malloc(sizeof(GLchar) * infoLength);
            if (szInfoLog != NULL) {
                GLsizei written;
                glGetProgramInfoLog(shader_program, infoLength, &written, szInfoLog);
                fprintf(log_file, "Shader Program Link Log : %s\n", szInfoLog);
                free(szInfoLog);
                DestroyWindow(hwnd);
            }
        }
    }

    model_view_proj_matrix_uniform = glGetUniformLocation(shader_program, "u_mvpMatrix");

    if (compute == CUDA) {
        cuda_compute.set_device_id(0);
    }
    else if (compute == OPENCL) {
        opencl_compute.get_devices(NVIDIA_PLATFORM);
        opencl_compute.set_device_id(0);
        opencl_compute.create_GL_context();
    }

    glGenVertexArrays(1, &vertex_array_object);
    glBindVertexArray(vertex_array_object);

    glGenBuffers(1, &vertex_buffer_object_position);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object_position);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * MESH_SIZE, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (compute == CUDA) {
        cuda_compute.register_GL_graphics_resource(vertex_buffer_object_position);
    }
    else if (compute == OPENCL) {
        opencl_compute.register_GL_graphics_resource(vertex_buffer_object_position);
    }

    glBindVertexArray(0);

    glShadeModel(GL_SMOOTH);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glEnable(GL_CULL_FACE);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)this);
}

void OpenGLWindow::show() {
    resize(RENDER_WIDTH, RENDER_HEIGHT);

    ShowWindow(hwnd, SW_SHOW);
    SetFocus(hwnd);
    SetForegroundWindow(hwnd);
}

void OpenGLWindow::resize(int width, int height) {
    if (height == 0)
        height = 1;
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    persp_proj_matrix = vmath::perspective(45.0f, (float)width / (float)height, 0.1f, 100.0f);
}

void OpenGLWindow::render() {
    vmath::mat4 modelViewProjectionMatrix = vmath::mat4::identity();
    vmath::mat4 modelViewMatrix = vmath::mat4::identity();
    vmath::mat4 translateMatrix = vmath::mat4::identity();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader_program);

    translateMatrix = vmath::translate(0.0f, 0.0f, -0.1f);
    modelViewMatrix = translateMatrix;

    modelViewProjectionMatrix = persp_proj_matrix * modelViewMatrix;

    glVertexAttrib4f(ATTRIBUTE_COLOR, 0.6f, 0.3f, 0.6f, -0.8f);
    glUniformMatrix4fv(model_view_proj_matrix_uniform, 1, GL_FALSE, modelViewProjectionMatrix);

    glBindVertexArray(vertex_array_object);

    if (compute == CUDA) {
        cuda_compute.map_resoucre();

        float4* pPos = cuda_compute.get_mapped_pointer();
        launch_cuda_kernel(pPos, MESH_WIDTH, MESH_HEIGHT, speed);

        cuda_compute.unmap_resoucre();
    }
    else if(compute == OPENCL){
        opencl_compute.set_kernel_args(MESH_WIDTH, MESH_HEIGHT, speed);
        opencl_compute.map_GL_resoucre();
        
        size_t globalWorkSize[2];
        globalWorkSize[0] = MESH_WIDTH;
        globalWorkSize[1] = MESH_HEIGHT;

        opencl_compute.launch_kernel(globalWorkSize);

        opencl_compute.unmap_GL_resoucre();
        opencl_compute.finish();
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object_position);
    glVertexAttribPointer(ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(ATTRIBUTE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDrawArrays(GL_POINTS, 0, MESH_WIDTH * MESH_HEIGHT);

    glBindVertexArray(0);

    glUseProgram(0);

    SwapBuffers(hdc);
}

void OpenGLWindow::update() {
    speed += 0.01f;
}

CUDACompute& OpenGLWindow::get_cuda_compute() {
    return cuda_compute;
}

OpenCLCompute& OpenGLWindow::get_opencl_compute() {
    return opencl_compute;
}

void OpenGLWindow::uninitialize() {
    GLsizei shader_count;
    GLuint* pShaders;

    if (is_fullscreen) {
        dw_style = GetWindowLong(hwnd, GWL_STYLE);
        SetWindowLong(hwnd, GWL_STYLE, dw_style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(hwnd, &wp_prev);
        SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        ShowCursor(TRUE);
    }

    if (vertex_array_object) {
        glDeleteVertexArrays(1, &vertex_array_object);
        vertex_array_object = 0;
    }

    if (vertex_buffer_object_position) {
        glDeleteBuffers(1, &vertex_buffer_object_position);
        vertex_buffer_object_position = 0;
    }

    if (shader_program) {
        glUseProgram(shader_program);
        glGetProgramiv(shader_program, GL_ATTACHED_SHADERS, &shader_count);

        pShaders = (GLuint*)malloc(sizeof(GLuint) * shader_count);
        glGetAttachedShaders(shader_program, shader_count, &shader_count, pShaders);

        for (GLsizei i = 0; i < shader_count; i++) {
            glDetachShader(shader_program, pShaders[i]);
            glDeleteShader(pShaders[i]);
            pShaders[i] = 0;
        }

        free(pShaders);
        pShaders = NULL;

        glDeleteShader(shader_program);
        shader_program = 0;

        glUseProgram(0);
    }

    if (wglGetCurrentContext() == hglrc)
        wglMakeCurrent(NULL, NULL);

    if (hglrc) {
        wglDeleteContext(hglrc);
        hglrc = NULL;
    }

    if (hdc) {
        ReleaseDC(hwnd, hdc);
        hdc = NULL;
    }

    if (log_file) {
        fclose(log_file);
        log_file = NULL;
    }
}

LRESULT OpenGLWindow::WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {
	OpenGLWindow* pOGLWnd = (OpenGLWindow*)GetWindowLongPtr(hwnd, GWLP_USERDATA);

	switch (iMsg) {
    case WM_SETFOCUS:
        if (pOGLWnd)
            pOGLWnd->is_active = true;
        break;

    case WM_KILLFOCUS:
        if (pOGLWnd)
            pOGLWnd->is_active = false;
        break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_DESTROY:
		if (pOGLWnd)
			pOGLWnd->uninitialize();
		break;

    case WM_SIZE:
        if (pOGLWnd)
            pOGLWnd->resize(LOWORD(lParam), HIWORD(lParam));
        break;

    case WM_KEYDOWN:
        switch (wParam) {

        case VK_ESCAPE:
            DestroyWindow(hwnd);
            break;

        case 0x46:
        case 0x66:
            if (pOGLWnd)
                pOGLWnd->toggle_fullscreen();
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