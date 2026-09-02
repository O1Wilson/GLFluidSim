#include "Renderer.h"

#include <algorithm>

namespace {
    // Same ghost-cell indexing scheme as FluidSolver::IX, duplicated here
    // since the renderer only needs read-only access to a flat float array,
    // not the solver's internals.
    inline int IX(int i, int j, int N) { return i + (N + 2) * j; }
}

Renderer::Renderer(int gridSize) : N(gridSize), textureBuffer(N * N) {}

Renderer::~Renderer() {
    if (quadVAO) glDeleteVertexArrays(1, &quadVAO);
    if (quadVBO) glDeleteBuffers(1, &quadVBO);
    if (densityTex) glDeleteTextures(1, &densityTex);
}

void Renderer::init() {
    setupQuad();
    setupTexture();
}

void Renderer::setupQuad() {
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);
}

void Renderer::setupTexture() {
    glGenTextures(1, &densityTex);
    glBindTexture(GL_TEXTURE_2D, densityTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, N, N, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void Renderer::uploadDensity(const float* density) {
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            float d = std::min(1.0f, std::max(0.0f, density[IX(i, j, N)]));
            textureBuffer[(j - 1) * N + (i - 1)] = static_cast<unsigned char>(d * 255.0f);
        }
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, densityTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, N, N, 0, GL_RED, GL_UNSIGNED_BYTE, textureBuffer.data());
}

void Renderer::render(Shader& shader) {
    shader.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, densityTex);
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}
