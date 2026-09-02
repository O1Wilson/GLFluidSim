#include "FluidSolver.h"

#include <algorithm>
#include <cmath>

FluidSolver::FluidSolver(int N)
    : N(N), SIZE((N + 2) * (N + 2)),
      u(SIZE, 0.0f), v(SIZE, 0.0f),
      u_prev(SIZE, 0.0f), v_prev(SIZE, 0.0f),
      dens(SIZE, 0.0f), dens_prev(SIZE, 0.0f) {}

void FluidSolver::reset() {
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(u_prev.begin(), u_prev.end(), 0.0f);
    std::fill(v_prev.begin(), v_prev.end(), 0.0f);
    std::fill(dens.begin(), dens.end(), 0.0f);
    std::fill(dens_prev.begin(), dens_prev.end(), 0.0f);
}

void FluidSolver::addDensity(int i, int j, float amount) {
    dens_prev[IX(i, j)] += amount;
}

void FluidSolver::addVelocity(int i, int j, float du, float dv) {
    u_prev[IX(i, j)] += du;
    v_prev[IX(i, j)] += dv;
}

void FluidSolver::set_bnd(int b, float* x) {
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

void FluidSolver::add_source(float* x, const float* s, float dt) {
    for (int i = 0; i < SIZE; i++) x[i] += dt * s[i];
}

void FluidSolver::lin_solve(int b, float* x, const float* x0, float a, float c) {
    for (int k = 0; k < 20; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                    x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
            }
        }
        set_bnd(b, x);
    }
}

void FluidSolver::diffuse(int b, float* x, const float* x0, float diff, float dt) {
    float a = dt * diff * N * N;
    lin_solve(b, x, x0, a, 1.0f + 4.0f * a);
}

float FluidSolver::cubicInterpolate(float p0, float p1, float p2, float p3, float t) {
    float a = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float b = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    float c = -0.5f * p0 + 0.5f * p2;
    float d = p1;
    return ((a * t + b) * t + c) * t + d;
}

float FluidSolver::bicubicSample(const float* d0, float x, float y) {
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

void FluidSolver::advect(int b, float* d, const float* d0, const float* u, const float* v, float dt) {
    float dt0 = dt * N;
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
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
                d[IX(i, j)] = bicubicSample(d0, x, y);
            }
        }
    }
    set_bnd(b, d);
}

void FluidSolver::project(float* u, float* v, float* p, float* div) {
    float h = 1.0f / N;

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            div[IX(i, j)] = -0.5f * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                v[IX(i, j + 1)] - v[IX(i, j - 1)]);
            p[IX(i, j)] = 0.0f;
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);

    lin_solve(0, p, div, 1.0f, 4.0f);

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            u[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
            v[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
        }
    }
    set_bnd(1, u);
    set_bnd(2, v);
}

// NOTE: the pointer-swap choreography below is preserved exactly as it was
// in the original free function. u0/v0 are used as scratch space during the
// projection steps and end up zeroed at the end (they represent the
// "previous"/source buffers, which are consumed each step).
void FluidSolver::velStep(float visc, float dt) {
    float* uPtr = u.data();
    float* vPtr = v.data();
    float* u0 = u_prev.data();
    float* v0 = v_prev.data();

    add_source(uPtr, u0, dt);
    add_source(vPtr, v0, dt);

    std::swap(u0, uPtr); diffuse(1, uPtr, u0, visc, dt);
    std::swap(v0, vPtr); diffuse(2, vPtr, v0, visc, dt);

    project(uPtr, vPtr, u0, v0);

    std::swap(u0, uPtr); std::swap(v0, vPtr);
    advect(1, uPtr, u0, u0, v0, dt);
    advect(2, vPtr, v0, u0, v0, dt);

    project(uPtr, vPtr, u0, v0);

    std::fill(u0, u0 + SIZE, 0.0f);
    std::fill(v0, v0 + SIZE, 0.0f);
}

void FluidSolver::densStep(float diff, float dt) {
    float* x = dens.data();
    float* x0 = dens_prev.data();
    float* uPtr = u.data();
    float* vPtr = v.data();

    add_source(x, x0, dt);
    std::swap(x0, x); diffuse(0, x, x0, diff, dt);
    std::swap(x0, x); advect(0, x, x0, uPtr, vPtr, dt);

    std::fill(x0, x0 + SIZE, 0.0f);
}
