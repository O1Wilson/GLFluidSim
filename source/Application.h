#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <shader.h>

#include "FluidSolver.h"
#include "Renderer.h"
#include "InputHandler.h"

// Owns the GLFW window and the main loop; wires FluidSolver, Renderer and
// InputHandler together each frame, exactly like the original main().
class Application {
public:
    Application(unsigned int width, unsigned int height, const char* title, int gridSize);
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    // Creates the window/GL context, loads GLAD, compiles the shader, and
    // initializes the renderer. Returns false on failure.
    bool init();

    // Runs the main loop until the window is closed.
    void run();

private:
    unsigned int width;
    unsigned int height;
    const char* title;

    GLFWwindow* window = nullptr;
    Shader* screenShader = nullptr;

    FluidSolver solver;
    Renderer renderer;
    InputHandler input;

    float diff = 0.00001f;
    float visc = 0.00001f;

    static void framebufferSizeCallback(GLFWwindow* window, int w, int h);
};
