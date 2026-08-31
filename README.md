## Real-Time 2D Fluid Simulation
A real-time CPU-based fluid simulation built in C++ and OpenGL using Jos Stam's Stable Fluids method. The project simulates smoke-like fluid behavior by solving velocity and density fields on a grid and rendering the results in real time.

## Overview
This project explores the mathematics and engineering behind real-time fluid simulation. Rather than animating fluid with predefined effects, the simulation calculates how fluid moves and interacts over time based on physical principles.

The current implementation runs entirely on the CPU using a 2D grid and renders the resulting density field through OpenGL. Future development includes moving the solver to GPU compute shaders and expanding the simulation into 3D space.

## Features
* Real-time 2D fluid simulation
* CPU-based Stable Fluids implementation
* OpenGL 4.6 rendering pipeline
* Velocity and density field simulation
* Bilinear and bicubic interpolation modes
* Interactive fluid injection and force application

## How Fluid Simulation Works
Fluid motion can be represented as a grid where each cell stores:

* Velocity: the speed and direction the fluid is moving
* Density: the visible "smoke" or dye carried by the fluid

Each frame, the simulation performs a series of steps:

### 1. Add Forces
External forces are introduced into the fluid, such as user-generated movement or injected velocity.

### 2. Diffusion
Diffusion causes the fluid to spread into neighboring cells, producing motion over time.

### 3. Projection
Real fluids do not spontaneously gain or lose volume. The projection step enforces this behavior by correcting the velocity field and maintaining a divergence-free flow.

### 4. Advection
The simulation then transports density and velocity throughout the grid, moving material according to the current flow direction.

Together, these steps create stable and visually realistic fluid motion while remaining efficient enough for real-time applications.

## Technical Details
This project is based on concepts from **Jos Stam's "Real-Time Fluid Dynamics for Games"**  
Paper: [Stable Fluids](http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/GDC03.pdf)

Key implementation components include:
* Gauss-Seidel iterative solver for diffusion
* Semi-Lagrangian advection
* Pressure projection for incompressibility
* Grid-based Eulerian simulation
* Boundary condition handling
* Bilinear and bicubic sampling for transport operations

## Controls
| Key |	Action |
|---|------|
| W |	Inject fluid from the top |
| S |	Inject fluid from the bottom |
| A | Create opposing corner flows |
| D |	Create alternate corner flows |
| 1 |	Bilinear interpolation |
| 2 |	Bicubic interpolation |
| C |	Clear simulation |
| ESC |	Exit |

## Requisites
- **OpenGL 4.6** for rendering  
- **Glad** for OpenGL function loading  
- **GLFW** for context/window management  
- **GLM** for math utilities
- **Project linking has already been configured. Create the directories C:\OpenGL\lib and C:\OpenGL\includes, then place the required library and header files into their respective folders. Additionally replace the user folder with your user in the solution properties.**

## Future Improvements
* GPU compute shader implementation
* 3D fluid simulation
* Improved visualization and color mapping
* Performance benchmarking and optimization
* Interactive mouse-based force injection

## What I Learned

Through this project I gained experience with:

- Numerical simulation techniques
- Solving partial differential equations in real time
- OpenGL rendering pipelines
- Memory-efficient grid-based data structures
- Interpolation methods and their visual impact
- Balancing simulation accuracy with performance

This project strengthened my understanding of both graphics programming and the mathematical foundations behind physically-based simulation.

## Development Notes
Pictures and details about the development process and experiments can be found on my portfolio:  
[https://o1wilson.github.io/#projects](https://o1wilson.github.io/#projects)
