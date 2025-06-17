#pragma once

inline __device__ float clamp(float x, float min, float max) {
    return fmaxf(min, fminf(x, max));
}
