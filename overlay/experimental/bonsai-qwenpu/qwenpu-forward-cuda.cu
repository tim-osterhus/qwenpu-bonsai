#include "qwenpu-forward-cuda-layout.h"

__device__ float positive(float value) {
    return value > 0.0f ? value : 0.0f;
}

__device__ float round_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffff) > 0x7f800000) {
        return __uint_as_float((bits & 0xffff0000) | 0x00400000);
    }
    return __uint_as_float((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
}

__device__ float dot(const qwenpu_cuda_layer & layer, unsigned int row, const float * src) {
    float sum = 0.0f;
    for (auto i = layer.offsets[row]; i < layer.offsets[row + 1]; ++i) {
        const auto coefficient = layer.coefficients[i];
        sum += src[coefficient.index] * coefficient.value;
    }
    return sum;
}

__device__ float dot_lagged(const qwenpu_cuda_layer & layer, unsigned int row,
                            const float * src, unsigned int lag_slot) {
    float sum = 0.0f;
    for (auto i = layer.offsets[row]; i < layer.offsets[row + 1]; ++i) {
        const auto coefficient = layer.coefficients[i];
        const auto lag = coefficient.index / layer.dims;
        const auto column = coefficient.index - lag * layer.dims;
        // lag_slot is the oldest stored residual. The current residual is still in src.
        const auto slot = (lag_slot + 3 - lag) % 3;
        const float value = lag ? layer.lagged[slot * layer.dims + column] : src[column];
        sum += value * coefficient.value;
    }
    return sum;
}

extern "C" __global__ void qwenpu_ffn(qwenpu_cuda_layer layer, const float * src) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < layer.projected_count) {
        const float activation = positive(round_bf16(dot(layer, layer.second + row, src)));
        layer.projected[row] = activation * dot(layer, layer.first + row, src);
    }
}

extern "C" __global__ void qwenpu_query_key(qwenpu_cuda_layer layer, const float * src,
                                           unsigned int lag_slot) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < layer.heads * layer.width) {
        layer.q_values[row] = positive(dot_lagged(layer, layer.first + row, src, lag_slot));
        layer.k_values[row] = positive(dot_lagged(layer, layer.second + row, src, lag_slot));
    }
}

extern "C" __global__ void qwenpu_recurrent(qwenpu_cuda_layer layer, const float * src,
                                          unsigned int lag_slot) {
    const unsigned int head = blockIdx.x;
    const unsigned int col = threadIdx.x;
    const auto key_head = layer.offsets[layer.keys + head];
    const auto * query = layer.q_values + key_head * layer.width;
    auto * matrix = layer.state + head * layer.width * layer.width;
    const auto output = head * layer.width + col;

    __shared__ float factor, norm;
    __shared__ bool normalize;
    __shared__ unsigned int active, write_row;
    __shared__ float read[128];
    if (col == 0) {
        factor = 1.0f;
        if (layer.offsets[layer.decay + head] != layer.offsets[layer.decay + head + 1]) {
            const float decay = dot(layer, layer.decay + head, src);
            factor = decay == 0.0f ? 1.0f : decay == 110.0f ? 0.5f : 0.0f;
        }
        const auto * key = layer.k_values + key_head * layer.width;
        unsigned int maximum = 0;
        active = 0;
        for (unsigned int row = 0; row < layer.width; ++row) {
            if (key[row] > key[maximum]) {
                maximum = row;
            }
            active += query[row] > 0.0f;
        }
        write_row = layer.width;
        if (key[maximum] > 0.0f &&
            dot(layer, layer.beta + head, src) + layer.offsets[layer.beta_const + head] == 1.0f) {
            write_row = maximum;
        }
    }
    __syncthreads();

    if (factor != 1.0f) {
        for (unsigned int row = 0; row < layer.width; ++row) {
            matrix[row * layer.width + col] *= factor;
        }
    }
    if (write_row < layer.width) {
        matrix[write_row * layer.width + col] = positive(dot_lagged(layer, layer.v + output, src, lag_slot));
    }
    if (active > layer.qhot) {
        layer.projected[output] = __uint_as_float(0x7fc00000);
        return;
    }

    float value = 0.0f;
    for (unsigned int row = 0; row < layer.width; ++row) {
        if (query[row] > 0.0f) {
            value += matrix[row * layer.width + col];
        }
    }
    read[col] = value;
    __syncthreads();
    if (col == 0) {
        float sum = 0.0f;
        for (auto ref = layer.ref_begin; ref < layer.ref_end; ++ref) {
            sum += read[ref] * read[ref];
        }
        normalize = sum > 0.0f;
        norm = normalize ? sqrtf(sum / layer.width) : 1.0f;
    }
    __syncthreads();
    layer.projected[output] = normalize ? value / norm * positive(dot(layer, layer.gate + output, src)) : 0.0f;
}

extern "C" __global__ void qwenpu_gather_keep(const unsigned int * indices, unsigned int count,
                                            const float * src, float * dst) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < count) {
        dst[row] = src[indices[row]];
    }
}

extern "C" __global__ void qwenpu_residual(qwenpu_cuda_layer layer, const float * src,
                                         float * dst, unsigned int lag_slot) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < layer.dims) {
        const float value = src[row];
        dst[row] = value + dot(layer, layer.out + row, layer.projected);
        if (layer.values) {
            layer.lagged[lag_slot * layer.dims + row] = value;
        }
    }
}
