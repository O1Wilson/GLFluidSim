#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <shader.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

static const unsigned int SCR_WIDTH = 800;
static const unsigned int SCR_HEIGHT = 600;

static const int N = 128;
static const int SIZE = (N + 2) * (N + 2);

static std::vector<float> u(SIZE), v(SIZE);
static std::vector<float> u_prev(SIZE), v_prev(SIZE);
static std::vector<float> dens(SIZE), dens_prev(SIZE);

static float diff = 0.0f;
static float visc = 0.0001f;
static float dt = 0.016f;

static GLuint quadVAO = 0, quadVBO = 0;
static GLuint densityTex = 0;
static GLuint advectTex;

static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
static void processInput(GLFWwindow* window);
static void set_bnd(int N, int b, float* x);
static void add_source(int N, float* x, const float* s, float dt);
static void lin_solve(int N, int b, float* x, const float* x0, float a, float c);
static void diffuse(int N, int b, float* x, const float* x0, float diff, float dt);
static void advect(int N, int b, float* d, const float* d0, const float* u, const float* v, float dt);
static void project(int N, float* u, float* v, float* p, float* div);
static void vel_step(int N, float* u, float* v, float* u0, float* v0, float visc, float dt);
static void dens_step(int N, float* x, float* x0, float* u, float* v, float diff, float dt);
static void uploadDensityTexture(GLuint tex, const float* density);
static void fluidStart();
static void setupQuad(GLuint& vao, GLuint& vbo);

inline int IX(int i, int j) { return i + (N + 2) * j; }

enum InterpMode { BILINEAR = 0, BICUBIC = 1 };
static InterpMode interpMode = BILINEAR;

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "2D Fluid Sim", nullptr, nullptr);
    if (!window) { std::cout << "Failed to create GLFW window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cout << "Failed to init GLAD\n"; return -1; }
    glDisable(GL_DEPTH_TEST);

    Shader screenShader({
        {"shaders/vertex/framebuffer.vs", GL_VERTEX_SHADER},
        {"shaders/fragment/framebuffer.frag", GL_FRAGMENT_SHADER}
    });

    Shader advectShader({
        {"shaders/compute/advect.comp", GL_COMPUTE_SHADER}
    });

    screenShader.use();
    screenShader.setInt("screenTexture", 0);

    setupQuad(quadVAO, quadVBO);

    glGenTextures(1, &densityTex);
    glBindTexture(GL_TEXTURE_2D, densityTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, N, N, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &advectTex);
    glBindTexture(GL_TEXTURE_2D, advectTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    advectShader.use();
    glBindImageTexture(0, advectTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glDispatchCompute(N / 16, N / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        double now = glfwGetTime();
        dt = float(std::min(0.1, now - lastTime));
        lastTime = now;

        fluidStart();

        vel_step(N, u.data(), v.data(), u_prev.data(), v_prev.data(), visc, dt);
        dens_step(N, dens.data(), dens_prev.data(), u.data(), v.data(), diff, dt);

        uploadDensityTexture(densityTex, dens.data());

        glClearColor(0.05f, 0.06f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        screenShader.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, advectTex);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteTextures(1, &densityTex);

    glfwTerminate();
    return 0;
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

static void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) interpMode = BILINEAR;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) interpMode = BICUBIC;

    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        std::fill(u.begin(), u.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        std::fill(u_prev.begin(), u_prev.end(), 0.0f);
        std::fill(v_prev.begin(), v_prev.end(), 0.0f);
        std::fill(dens.begin(), dens.end(), 0.0f);
        std::fill(dens_prev.begin(), dens_prev.end(), 0.0f);
    }
}

static void set_bnd(int N, int b, float* x) {
    for (int i = 1; i <= N; i++) {
        x[IX(0, i)] = (b == 1) ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N + 1, i)] = (b == 1) ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)] = (b == 2) ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = (b == 2) ? -x[IX(i, N)] : x[IX(i, N)];
    }
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N + 1)] = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)] = 0.5f * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

static void add_source(int N, float* x, const float* s, float dt) {
    int size = (N + 2) * (N + 2);
    for (int i = 0;i < size;i++) x[i] += dt * s[i];
}

static void lin_solve(int N, int b, float* x, const float* x0, float a, float c) {
    for (int k = 0; k < 20; k++) {
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                    x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
            }
        }
        set_bnd(N, b, x);
    }
}

static void diffuse(int N, int b, float* x, const float* x0, float diff, float dt) {
    float a = dt * diff * N * N;
    lin_solve(N, b, x, x0, a, 1.0f + 4.0f * a);
}

inline float cubicInterpolate(float p0, float p1, float p2, float p3, float t) {
    float a = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float b = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    float c = -0.5f * p0 + 0.5f * p2;
    float d = p1;
    return ((a * t + b) * t + c) * t + d;
}

float bicubicSample(const float* d0, float x, float y, int N) {
    int ix = (int)floor(x);
    int iy = (int)floor(y);

    float tx = x - ix;
    float ty = y - iy;

    float arr[4];
    for (int m = -1; m <= 2; m++) {
        float row[4];
        for (int n = -1; n <= 2; n++) {
            int xi = std::min(N + 1, std::max(0, ix + n));
            int yi = std::min(N + 1, std::max(0, iy + m));
            row[n + 1] = d0[IX(xi, yi)];
        }
        arr[m + 1] = cubicInterpolate(row[0], row[1], row[2], row[3], tx);
    }
    return cubicInterpolate(arr[0], arr[1], arr[2], arr[3], ty);
}

static void advect(int N, int b, float* d, const float* d0, const float* u, const float* v, float dt) {
    float dt0 = dt * N;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            float x = i - dt0 * u[IX(i, j)];
            float y = j - dt0 * v[IX(i, j)];

            if (x < 0.5f) x = 0.5f;
            if (x > N + 0.5f) x = N + 0.5f;
            int i0 = (int)x;
            int i1 = i0 + 1;

            if (y < 0.5f) y = 0.5f;
            if (y > N + 0.5f) y = N + 0.5f;
            int j0 = (int)y;
            int j1 = j0 + 1;

            if (interpMode == BILINEAR) {
                float s1 = x - i0; float s0 = 1.0f - s1;
                float t1 = y - j0; float t0 = 1.0f - t1;

                d[IX(i, j)] =
                    s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                    s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
            }
            else {
                d[IX(i, j)] = bicubicSample(d0, x, y, N);
            }
        }
    }
    set_bnd(N, b, d);
}

static void project(int N, float* u, float* v, float* p, float* div) {
    float h = 1.0f / N;

    for (int i = 1;i <= N;i++) {
        for (int j = 1;j <= N;j++) {
            div[IX(i, j)] = -0.5f * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                v[IX(i, j + 1)] - v[IX(i, j - 1)]);
            p[IX(i, j)] = 0.0f;
        }
    }
    set_bnd(N, 0, div);
    set_bnd(N, 0, p);

    lin_solve(N, 0, p, div, 1.0f, 4.0f);

    for (int i = 1;i <= N;i++) {
        for (int j = 1;j <= N;j++) {
            u[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
            v[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
        }
    }
    set_bnd(N, 1, u);
    set_bnd(N, 2, v);
}

static void vel_step(int N, float* u, float* v, float* u0, float* v0, float visc, float dt) {
    add_source(N, u, u0, dt);
    add_source(N, v, v0, dt);

    std::swap(u0, u); diffuse(N, 1, u, u0, visc, dt);
    std::swap(v0, v); diffuse(N, 2, v, v0, visc, dt);

    project(N, u, v, u0, v0);

    std::swap(u0, u); std::swap(v0, v);
    advect(N, 1, u, u0, u0, v0, dt);
    advect(N, 2, v, v0, u0, v0, dt);

    project(N, u, v, u0, v0);

    std::fill(u0, u0 + SIZE, 0.0f);
    std::fill(v0, v0 + SIZE, 0.0f);
}

static void dens_step(int N, float* x, float* x0, float* u, float* v, float diff, float dt) {
    add_source(N, x, x0, dt);
    std::swap(x0, x); diffuse(N, 0, x, x0, diff, dt);
    std::swap(x0, x); advect(N, 0, x, x0, u, v, dt);

    std::fill(x0, x0 + SIZE, 0.0f);
}

static void uploadDensityTexture(GLuint tex, const float* density) {
    std::vector<unsigned char> buffer(N * N);
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            float d = std::min(1.0f, std::max(0.0f, density[IX(i, j)]));
            buffer[(j - 1) * N + (i - 1)] = static_cast<unsigned char>(d * 255.0f);
        }
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, densityTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, N, N, 0, GL_RED, GL_UNSIGNED_BYTE, buffer.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

static void fluidStart() {
    int i1 = N / 3;
    int j1 = N;

    int i2 = 2 * N / 3;
    int j2 = 1;

    dens_prev[IX(i1, j1)] = 200.0f;
    v_prev[IX(i1, j1)] = -500.0f;

    dens_prev[IX(i2, j2)] = 200.0f;
    v_prev[IX(i2, j2)] = 500.0f;
}

static void setupQuad(GLuint& vao, GLuint& vbo) {
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);
}