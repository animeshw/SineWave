# Sine Wave Renderer with Direct3D 11 and OpenGL

This project creates a graphical application that renders a sine wave using Direct3D 11 and OpenGL, integrating compute APIs like CUDA or OpenCL for efficient calculations. The application features native Win32 windowing for a seamless user experience and supports keyboard event handling for real-time interaction.

- The application utilizes native Win32 to create the rendering window, handling essential window events such as resizing and closing. Users can interact with the sine wave parameters through keyboard input, enhancing the experience.

- Direct3D 11 is used to set up a complete rendering pipeline, including vertex buffers, shaders, and rasterization processes. Similarly, an OpenGL context is established, complete with shaders and buffers for rendering the sine wave, allowing for a side-by-side comparison of both graphics APIs.

- To compute the sine wave efficiently, the project implements compute shaders with CUDA or OpenCL, allowing for parallel calculations of sine wave data. This compute capability enhances performance by offloading calculations to the GPU, and the results are transferred back to the CPU for rendering.

- The sine wave is defined by the mathematical function \( y = A \sin(Bx + C) + D \), with parameters like amplitude (A), frequency (B), phase shift (C), and vertical offset (D) being adjustable in real time. Users can manipulate these parameters using the arrow keys on the keyboard.

- The rendering pipeline generates vertex data based on the computed sine wave values. Vertex and fragment shaders are set up for rendering, and double buffering is implemented to ensure smooth rendering and prevent flickering during updates.

- Performance optimization is a key aspect of the project, comparing rendering speeds and visual fidelity between Direct3D 11 and OpenGL. Profiling tools are utilized to analyze the performance impact of compute shaders versus traditional CPU calculations, providing insights into modern graphics programming techniques.

- Technologies used in this project include C++ as the primary programming language, with Direct3D 11 and OpenGL serving as the graphics APIs. Compute APIs like CUDA or OpenCL are integrated for efficient calculations, and the Win32 API is employed for window management. Math libraries such as GLM for OpenGL or DirectXMath for Direct3D may be utilized to assist in mathematical computations.

This project offers a comprehensive exploration of graphics programming, blending fundamental mathematical concepts with advanced rendering technologies, enhancing understanding of rendering pipelines, user interactivity, and the benefits of compute APIs in real-time graphics applications.
