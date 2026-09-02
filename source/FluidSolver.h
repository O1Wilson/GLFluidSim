#pragma once

#include <vector>

// Encapsulates the Jos Stam-style stable fluid simulation:
// grid storage, diffusion, advection, projection, and density transport.
class FluidSolver {
public:
    enum InterpMode { BILINEAR = 0, BICUBIC = 1 };

    explicit FluidSolver(int N);

    // Advances velocity field by dt using the given viscosity.
    void velStep(float visc, float dt);

    // Advances density field by dt using the given diffusion rate.
    void densStep(float diff, float dt);

    // Zeros all fields (velocity, previous velocity, density, previous density).
    void reset();

    // Adds to the pending density/velocity sources at grid cell (i, j).
    // These are consumed (and cleared) on the next velStep/densStep call.
    void addDensity(int i, int j, float amount);
    void addVelocity(int i, int j, float du, float dv);

    void setInterpMode(InterpMode mode) { interpMode = mode; }
    InterpMode getInterpMode() const { return interpMode; }

    int getN() const { return N; }
    const std::vector<float>& getDensity() const { return dens; }

private:
    int N;
    int SIZE;

    std::vector<float> u, v;
    std::vector<float> u_prev, v_prev;
    std::vector<float> dens, dens_prev;

    InterpMode interpMode = BILINEAR;

    inline int IX(int i, int j) const { return i + (N + 2) * j; }

    void set_bnd(int b, float* x);
    void add_source(float* x, const float* s, float dt);
    void lin_solve(int b, float* x, const float* x0, float a, float c);
    void diffuse(int b, float* x, const float* x0, float diff, float dt);
    void advect(int b, float* d, const float* d0, const float* u, const float* v, float dt);
    void project(float* u, float* v, float* p, float* div);

    float bicubicSample(const float* d0, float x, float y);
    static float cubicInterpolate(float p0, float p1, float p2, float p3, float t);
};
