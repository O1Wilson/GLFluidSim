#include "InputHandler.h"

InputHandler::InputHandler(FluidSolver& solver) : solver(solver) {}

void InputHandler::processWindowInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) solver.setInterpMode(FluidSolver::BILINEAR);
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) solver.setInterpMode(FluidSolver::BICUBIC);

    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) solver.reset();
}

void InputHandler::applyFluidSources(GLFWwindow* window) {
    const int N = solver.getN();

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        int x1 = 15;
        int y1 = N - 15;
        solver.addDensity(x1, y1, 150.0f);
        solver.addVelocity(x1, y1, 400.0f, -400.0f);

        int x2 = N - 15;
        int y2 = 15;
        solver.addDensity(x2, y2, 150.0f);
        solver.addVelocity(x2, y2, -400.0f, 400.0f);
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        int x1 = N - 15;
        int y1 = N - 15;
        solver.addDensity(x1, y1, 150.0f);
        solver.addVelocity(x1, y1, -400.0f, -400.0f);

        int x2 = 15;
        int y2 = 15;
        solver.addDensity(x2, y2, 150.0f);
        solver.addVelocity(x2, y2, 400.0f, 400.0f);
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        int x = N / 2;
        int y = N - 15;
        solver.addDensity(x, y, 150.0f);
        solver.addVelocity(x, y, 0.0f, -400.0f);
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        int x = N / 2;
        int y = 15;
        solver.addDensity(x, y, 150.0f);
        solver.addVelocity(x, y, 0.0f, 400.0f);
    }
}
