#include "Application.h"

#include <algorithm>
#include <iostream>

Application::Application(unsigned int width, unsigned int height, const char* title, int gridSize)
    : width(width), height(height), title(title),
      solver(gridSize), renderer(gridSize), input(solver) {}

Application::~Application() {
    delete screenShader;
    if (window) {
        glfwTerminate();
    }
}

bool Application::init() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cout << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to init GLAD\n";
        return false;
    }
    glDisable(GL_DEPTH_TEST);

    screenShader = new Shader({
        {"shaders/framebuffer.vs", GL_VERTEX_SHADER},
        {"shaders/framebuffer.frag", GL_FRAGMENT_SHADER}
    });
    screenShader->use();
    screenShader->setInt("screenTexture", 0);

    renderer.init();

    return true;
}

void Application::run() {
    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        input.processWindowInput(window);

        double now = glfwGetTime();
        float dt = float(std::min(0.1, now - lastTime));
        lastTime = now;

        input.applyFluidSources(window);

        solver.velStep(visc, dt);
        solver.densStep(diff, dt);

        renderer.uploadDensity(solver.getDensity().data());

        glClearColor(0.05f, 0.06f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        renderer.render(*screenShader);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void Application::framebufferSizeCallback(GLFWwindow* /*window*/, int w, int h) {
    glViewport(0, 0, w, h);
}
