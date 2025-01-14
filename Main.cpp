#include "MyWindow.hpp"
#include "OpenGLWindow.hpp"
#include "Direct3D11Window.hpp"
#include <windowsx.h>
#include <CommCtrl.h>
#include <math.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "OpenCL.lib")
#pragma comment(lib, "cudart.lib")

#define ID_COMBOBOX_RENDER 1
#define ID_COMBOBOX_COMPUTE 2
#define ID_BUTTON_RUN 3

const int WIDTH = 600;
const int HEIGHT = 400;


extern "C"
{
	__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

MyWindow* pWindow = NULL;

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {
	static HWND hwndRenderComboBox;
	static HWND hwndComputeComboBox;
	static HWND hwndRunButton;
	static int RenderIndex;
	static int ComputeIndex;
	HBRUSH hBrush;

	switch (iMsg) {
		case WM_ERASEBKGND:
			return 0;

		case WM_CREATE:
			hwndRenderComboBox = CreateWindow(WC_COMBOBOX, TEXT("RENDER API"),
				CBS_DROPDOWN | CBS_HASSTRINGS | WS_CHILD | WS_OVERLAPPED | WS_VISIBLE,
				300, 100, 200, 200, hwnd, (HMENU)ID_COMBOBOX_RENDER, ((LPCREATESTRUCT)lParam)->hInstance, NULL);
			SendMessageA(hwndRenderComboBox, (UINT)CB_ADDSTRING, (WPARAM)0, (LPARAM)"Direct3D11");
			SendMessageA(hwndRenderComboBox, (UINT)CB_ADDSTRING, (WPARAM)0, (LPARAM)"OpenGL");

			hwndComputeComboBox = CreateWindow(WC_COMBOBOX, TEXT("COMPUTE API"),
				CBS_DROPDOWN | CBS_HASSTRINGS | WS_CHILD | WS_OVERLAPPED | WS_VISIBLE,
				300, 150, 200, 200, hwnd, (HMENU)ID_COMBOBOX_COMPUTE, ((LPCREATESTRUCT)lParam)->hInstance, NULL);
			SendMessageA(hwndComputeComboBox, (UINT)CB_ADDSTRING, (WPARAM)0, (LPARAM)"OpenCL");
			SendMessageA(hwndComputeComboBox, (UINT)CB_ADDSTRING, (WPARAM)0, (LPARAM)"CUDA");

			CreateWindow( WC_STATIC, TEXT("Render API : "), WS_VISIBLE | WS_CHILD, 150, 105, 80, 20,
				hwnd, NULL, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

			CreateWindow(WC_STATIC, TEXT("Compute API : "), WS_VISIBLE | WS_CHILD, 150, 155, 100, 20,
				hwnd, NULL, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

			hwndRunButton = CreateWindow(WC_BUTTON, TEXT("RUN"), WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
				250, 250, 100, 25, hwnd, (HMENU)ID_BUTTON_RUN, ((LPCREATESTRUCT)lParam)->hInstance, NULL);

			ComboBox_Enable(hwndComputeComboBox, false);
			break;

		case WM_CTLCOLORSTATIC:
			hBrush = (HBRUSH)GetStockObject(WHITE_BRUSH);
			SetBkMode((HDC)wParam, TRANSPARENT);
			return (LRESULT)hBrush;

		case WM_COMMAND:
			if (LOWORD(wParam) == ID_COMBOBOX_RENDER) {
				if (HIWORD(wParam) == CBN_SELCHANGE) {
					int ItemIndex = (int)SendMessage((HWND)lParam, (UINT)CB_GETCURSEL, (WPARAM)0, (LPARAM)0);
					if (ItemIndex == 0) {
						RenderIndex = 0;
					}
					else if (ItemIndex == 1) {
						RenderIndex = 1;
					}
				}
				if (HIWORD(wParam) == CBN_SELENDOK) {
					ComboBox_Enable(hwndComputeComboBox, true);
					SendMessageA(hwndComputeComboBox, (UINT)CB_RESETCONTENT, (WPARAM)0, (LPARAM)0);
					SendMessageA(hwndComputeComboBox, (UINT)CB_ADDSTRING, (WPARAM)0, (LPARAM)"OpenCL");
					SendMessageA(hwndComputeComboBox, (UINT)CB_ADDSTRING, (WPARAM)0, (LPARAM)"CUDA");
				}
			}
			else if (LOWORD(wParam) == ID_COMBOBOX_COMPUTE) {
				if (HIWORD(wParam) == CBN_SELCHANGE) {
					int ItemIndex = (int)SendMessage((HWND)lParam, (UINT)CB_GETCURSEL, (WPARAM)0, (LPARAM)0);
					if (ItemIndex == 0) {
						ComputeIndex = OPENCL;
					}
					else if (ItemIndex == 1) {
						ComputeIndex = CUDA;
					}
				}
			}

			if (LOWORD(wParam) == ID_BUTTON_RUN && HIWORD(wParam) == BN_CLICKED) {
				if (SendMessage(hwndRenderComboBox, CB_GETCURSEL, 0, 0) == CB_ERR) {
					MessageBox(NULL, TEXT("Render API Not Selected"), TEXT("ERROR"), MB_OK | MB_TOPMOST | MB_ICONERROR);
					break;
				}
				else if (SendMessage(hwndComputeComboBox, CB_GETCURSEL, 0, 0) == CB_ERR) {
					MessageBox(NULL, TEXT("Compute API Not Selected"), TEXT("ERROR"), MB_OK | MB_TOPMOST | MB_ICONERROR);
					break;
				}
				
				if (RenderIndex == 0) {
					if (pWindow) {
						delete pWindow;
						pWindow = nullptr;
					}
					pWindow = new Direct3D11Window(hwnd);
					pWindow->initialize(ComputeIndex);
					pWindow->show();
				}
				else if (RenderIndex == 1) {
					if (pWindow) {
						delete pWindow;
						pWindow = nullptr;
					}
					pWindow = new OpenGLWindow(hwnd);
					pWindow->initialize(ComputeIndex);
					pWindow->show();
				}
			}
			break;

		case WM_KEYDOWN:
			switch (wParam) {
				case VK_ESCAPE:
					DestroyWindow(hwnd);
					break;

				default:
					break;
			}
			break;
		
		case WM_CLOSE:
			DestroyWindow(hwnd);
			break;

		case WM_DESTROY:
			PostQuitMessage(0);
			break;

		default:
			break;
	}
	return DefWindowProc(hwnd, iMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {	
	int x_centre = (GetSystemMetrics(SM_CXSCREEN) / 2) - (WIDTH / 2);
	int y_centre = (GetSystemMetrics(SM_CYSCREEN) / 2) - (HEIGHT / 2);

	WNDCLASSEX wndclass;
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbWndExtra = 0;
	wndclass.cbClsExtra = 0;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = TEXT("SineWave");
	wndclass.lpszMenuName = nullptr;
	wndclass.hInstance = GetModuleHandle(NULL);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(NULL, IDC_ICON);
	wndclass.hIconSm = LoadIcon(NULL, IDC_ICON);

	RegisterClassEx(&wndclass);

	HWND hwnd = CreateWindowEx(WS_EX_APPWINDOW, TEXT("SineWave"), TEXT("Sine Wave Simulation"), 
		WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE | WS_OVERLAPPED | WS_MINIMIZEBOX | WS_SYSMENU, x_centre, y_centre,
		WIDTH, HEIGHT, nullptr, nullptr, GetModuleHandle(NULL), nullptr);

	ShowWindow(hwnd, SW_NORMAL);
	SetFocus(hwnd);
	SetForegroundWindow(hwnd);

	MSG msg;
	while (true) {
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT)
				break;
			else {
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else if (pWindow && pWindow->is_window_active()) {
			pWindow->render();
			pWindow->update();
		}

	}
	
	if (pWindow) {
		delete pWindow;
		pWindow = nullptr;
	}
	return (int)msg.wParam;
}