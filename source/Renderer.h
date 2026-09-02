#pragma once

#include <glad/glad.h>
#include <shader.h>

#include <vector>

// Owns all GPU-side resources needed to display the density field:
// the fullscreen quad and the density texture, plus the upload/draw calls.
class Renderer {
public:
    explicit Renderer(int gridSize);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Creates the quad VAO/VBO and the density texture. Requires a current GL context.
    void init();

    // Converts the solver's float density field (with ghost cells) into an
    // 8-bit texture and uploads it to the GPU.
    void uploadDensity(const float* density);

    // Draws the fullscreen quad textured with the density texture, using the given shader.
    void render(Shader& shader);

private:
    int N;
    GLuint quadVAO = 0;
    GLuint quadVBO = 0;
    GLuint densityTex = 0;
    std::vector<unsigned char> textureBuffer;

    void setupQuad();
    void setupTexture();
};
