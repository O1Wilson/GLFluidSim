#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <shader.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

/* -------{Simulation Constants and Globals}-------- */

static const unsigned int SCR_WIDTH = 800;
static const unsigned int SCR_HEIGHT = 600;

static const int N = 128;
static const int SIZE = (N + 2) * (N + 2);

const GLuint gx = (N + 15) / 16;
const GLuint gy = (N + 15) / 16;

static std::vector<float> u(SIZE), v(SIZE);
static std::vector<float> u_prev(SIZE), v_prev(SIZE);
static std::vector<float> dens(SIZE), dens_prev(SIZE);

static float diff = 0.0001f;
static float visc = 0.0001f;
static float dt = 0.016f;

static Shader* fluidShaderPtr = nullptr;

enum InterpMode { BILINEAR = 0, BICUBIC = 1 };
static InterpMode interpMode = BILINEAR;

static GLuint quadVAO = 0, quadVBO = 0;
static GLuint advectTex, densitySimTex, velTex;

static void uploadVelocityTexture(GLuint tex, const float* u, const float* v);
static void uploadDensitySimTexture(GLuint tex, const float* density);

inline int IX(int i, int j) { return i + (N + 2) * j; }

GLuint createTexture2D(int width, int height, GLint internalFormat, GLenum format, GLenum type, GLint minFilter = GL_NEAREST, 
    GLint magFilter = GL_NEAREST, GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE, const void* data = nullptr) 
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);

    return tex;
}

/* -------{GLFW Input + Window Callbacks}-------- */

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

static void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) interpMode = BILINEAR;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) interpMode = BICUBIC;

    //update to gpu
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        std::fill(u.begin(), u.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        std::fill(u_prev.begin(), u_prev.end(), 0.0f);
        std::fill(v_prev.begin(), v_prev.end(), 0.0f);
        std::fill(dens.begin(), dens.end(), 0.0f);
        std::fill(dens_prev.begin(), dens_prev.end(), 0.0f);
    }
}

/* -------{Math / Fluid Simulation Functions}-------- */

/* ------------------------------------------------------------------------
   {Variable Dictionary}
   Based on Jos Stam's "Stable Fluids" (1999)

   u      : x-component of velocity field (horizontal velocity)
   v      : y-component of velocity field (vertical velocity)
   u0     : previous step's x-component velocity (temporary buffer)
   v0     : previous step's y-component velocity (temporary buffer)

   dens   : density field (dye concentration)
   dens0  : previous step’s density field (temporary buffer)

   p      : pressure field
   div    : divergence field

   N      : grid resolution (size is N x N cells)
   SIZE   : total number of cells including boundary
   IX(i,j): macro function mapping 2D grid coordinates to 1D index

   dt     : timestep (step size)
   diff   : diffusion rate
   visc   : viscosity
   b      : flag for boundary type  (0 = scalar, 1 = u-component, 2 = v-component)

   ------------------------------------------------------------------------ */

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

void fluidCompute(Shader& shader, GLuint dst, GLuint src, GLuint vel, float dt, int interpMode, int N) {
    shader.use();
    shader.setFloat("dt0", dt * N);
    shader.setInt("interpMode", interpMode);

    glBindImageTexture(0, dst, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(1, src, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, vel, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);

    glDispatchCompute(gx, gy, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

static void downloadField(GLuint tex, float* dst) {
    std::vector<float> buffer(N * N);
    glBindTexture(GL_TEXTURE_2D, tex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, buffer.data());

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            dst[IX(i, j)] = buffer[(j - 1) * N + (i - 1)];
        }
    }
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

    uploadVelocityTexture(velTex, u0, v0);
    uploadDensitySimTexture(densitySimTex, u);
    fluidCompute(*fluidShaderPtr, advectTex, densitySimTex, velTex, dt, interpMode, N);
    downloadField(advectTex, u);

    uploadVelocityTexture(velTex, u0, v0);
    uploadDensitySimTexture(densitySimTex, v);
    fluidCompute(*fluidShaderPtr, advectTex, densitySimTex, velTex, dt, interpMode, N);
    downloadField(advectTex, v);

    project(N, u, v, u0, v0);

    std::fill(u0, u0 + SIZE, 0.0f);
    std::fill(v0, v0 + SIZE, 0.0f);
}

static void dens_step(int N, float* x, float* x0, float* u, float* v, float diff, float dt) {
    add_source(N, x, x0, dt);
    std::swap(x0, x);
    diffuse(N, 0, x, x0, diff, dt);
    std::swap(x0, x);

    uploadDensitySimTexture(densitySimTex, x0);
    uploadVelocityTexture(velTex, u, v);
    fluidCompute(*fluidShaderPtr, advectTex, densitySimTex, velTex, dt, interpMode, N);
    downloadField(advectTex, x);

    std::fill(x0, x0 + SIZE, 0.0f);
}

/* -------{Temporary GPU Upload Utilities}-------- */

static void uploadVelocityTexture(GLuint tex, const float* u, const float* v) {
    std::vector<glm::vec2> buffer(N * N);
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            buffer[(j - 1) * N + (i - 1)] = glm::vec2(u[IX(i, j)], v[IX(i, j)]);
        }
    }
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, N, N, 0, GL_RG, GL_FLOAT, buffer.data());
}

static void uploadDensitySimTexture(GLuint tex, const float* density) {
    std::vector<float> buffer(N * N);
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            buffer[(j - 1) * N + (i - 1)] = density[IX(i, j)];
        }
    }
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, N, N, 0, GL_RED, GL_FLOAT, buffer.data());
}

/* -------{Simulation Setup / Fluid Spawn}-------- */

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

/* -------{Rendering Setup}-------- */

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

void renderFrame(Shader& shader, GLuint tex, GLuint vao) {
    glClearColor(0.05f, 0.06f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    shader.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

/* -------{Program Initializion and Render Loop}-------- */

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

    Shader screenShader({
        {"shaders/vertex/framebuffer.vs", GL_VERTEX_SHADER},
        {"shaders/fragment/framebuffer.frag", GL_FRAGMENT_SHADER}
    });

    Shader fluidShader({
        {"shaders/compute/advect.comp", GL_COMPUTE_SHADER}
    });
    fluidShaderPtr = &fluidShader;

    screenShader.use();
    screenShader.setInt("screenTexture", 0);

    setupQuad(quadVAO, quadVBO);

    advectTex = createTexture2D(N, N, GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST, GL_NEAREST);
    densitySimTex = createTexture2D(N, N, GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST, GL_NEAREST);
    velTex = createTexture2D(N, N, GL_RG32F, GL_RG, GL_FLOAT, GL_NEAREST, GL_NEAREST);

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        double now = glfwGetTime();
        dt = float(std::min(0.1, now - lastTime));
        lastTime = now;

        fluidStart();

        vel_step(N, u.data(), v.data(), u_prev.data(), v_prev.data(), visc, dt);
        dens_step(N, dens.data(), dens_prev.data(), u.data(), v.data(), diff, dt);

        uploadVelocityTexture(velTex, u.data(), v.data());
        uploadDensitySimTexture(densitySimTex, dens.data());

        fluidCompute(fluidShader, advectTex, densitySimTex, velTex, dt, interpMode, N);
        renderFrame(screenShader, advectTex, quadVAO);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteTextures(1, &advectTex);
    glDeleteTextures(1, &densitySimTex);
    glDeleteTextures(1, &velTex);

    glfwTerminate();
    return 0;
}