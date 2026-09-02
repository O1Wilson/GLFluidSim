#pragma once

#include <GLFW/glfw3.h>

#include "FluidSolver.h"

// Translates raw GLFW key state into FluidSolver commands and window control.
// Polling-based (checked once per frame), matching the original processInput/fluidStart.
class InputHandler {
public:
    explicit InputHandler(FluidSolver& solver);

    // Handles window-level input: escape to close, 1/2 to switch interpolation
    // mode, C to reset the simulation.
    void processWindowInput(GLFWwindow* window);

    // Handles the A/D/W/S "jet" sources that inject density and velocity into the fluid.
    void applyFluidSources(GLFWwindow* window);

private:
    FluidSolver& solver;
};
