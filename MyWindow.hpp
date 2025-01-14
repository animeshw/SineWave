#pragma once
#include <Windows.h>
#include <gl/glew.h>
#include <gl/wglew.h>
#include <gl/GL.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>

#include "CUDACompute.hpp"
#include "OpenCLCompute.hpp"

const int RENDER_WIDTH = 1280;
const int RENDER_HEIGHT = 720;
const int MESH_WIDTH = 1024 * 1;
const int MESH_HEIGHT = MESH_WIDTH;
const int MESH_SIZE = MESH_WIDTH * MESH_HEIGHT * 4;
const int CUDA = 1;
const int OPENCL = 2;
const int NVIDIA_PLATFORM = 0;
const int INTEL_PLATFORM = 1;

class MyWindow {
protected:
	HWND hwnd;
	FILE* log_file;
	bool is_fullscreen;
    bool is_active;
    DWORD dw_style;
    WINDOWPLACEMENT wp_prev;
    int compute;
    float points_pos[MESH_WIDTH][MESH_HEIGHT][4];
    float speed;

protected:
	MyWindow() : hwnd(NULL), log_file(NULL), is_fullscreen(false), is_active(false), dw_style(0), compute(-1), speed(1.0f) {
        wp_prev = { sizeof(WINDOWPLACEMENT) };
        for (int i = 0; i < MESH_WIDTH; i++) {
            for (int j = 0; j < MESH_HEIGHT; j++) {
                for (int k = 0; k < 4; k++) {
                    points_pos[i][j][k] = 0.0f;
                }
            }
        }
    }

public:
    virtual void initialize(int compute_api) = 0;
    virtual void show() = 0;
	virtual void resize(int, int) = 0;
	virtual void render() = 0;
	virtual void update() = 0;
	virtual void uninitialize() = 0;
    virtual CUDACompute& get_cuda_compute() = 0;
    virtual OpenCLCompute& get_opencl_compute() = 0;

    bool is_window_active() { return is_active; }

    void toggle_fullscreen(void) {
        MONITORINFO mi = { sizeof(MONITORINFO) };

        if (is_fullscreen == false) {
            dw_style = GetWindowLong(hwnd, GWL_STYLE);
            if (dw_style & WS_OVERLAPPEDWINDOW) {
                if (GetWindowPlacement(hwnd, &wp_prev) && GetMonitorInfo(MonitorFromWindow(hwnd, MONITORINFOF_PRIMARY), &mi)) {
                    SetWindowLong(hwnd, GWL_STYLE, dw_style & ~WS_OVERLAPPEDWINDOW);
                    SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                        mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_FRAMECHANGED | SWP_NOZORDER);
                    ShowCursor(FALSE);
                    is_fullscreen = true;
                }
            }
        }
        else {
            SetWindowLong(hwnd, GWL_STYLE, dw_style | WS_OVERLAPPEDWINDOW);
            SetWindowPlacement(hwnd, &wp_prev);
            SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);
            ShowCursor(TRUE);
            is_fullscreen = false;
        }
    }

	virtual ~MyWindow() {
		DestroyWindow(hwnd);
	}
};
