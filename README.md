## Work In Progress
This project is currently being worked on. I have set up the fluid sim in 2D, ran on the CPU, but I plan to move the code to compute shaders, and expand into 3D.
![Fluid Sim ran on CPU](website/2DFluidSimOpenGLCPU.mp4)  

## References
This project is based on concepts from **Jos Stam's "Real-Time Fluid Dynamics for Games"**  
[Stable Fluids](http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/GDC03.pdf)

## Requisites
- **OpenGL** for rendering  
- **Glad** for OpenGL function loading  
- **GLFW** for context/window management  
- **GLM** for math utilities  
- Build system configured via **CMake**  
  - See included `CMakeLists.txt` for setup details  

## Development Notes
More details about the development process and experiments can be found on my portfolio:  
[https://o1wilson.github.io/#projects](https://o1wilson.github.io/#projects)
