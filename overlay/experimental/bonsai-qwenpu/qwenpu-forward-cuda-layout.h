#pragma once

struct qwenpu_cuda_coefficient {
    unsigned int index;
    float value;
};

struct qwenpu_cuda_layer {
    const qwenpu_cuda_coefficient * coefficients;
    const unsigned int * offsets;
    float * state;
    float * lagged;
    float * q_values;
    float * k_values;
    float * projected;
    unsigned int dims, width, heads, values, qhot, ref_begin, ref_end;
    unsigned int first, second, v, decay, beta, gate, out, keys, beta_const;
    unsigned int projected_count;
};
