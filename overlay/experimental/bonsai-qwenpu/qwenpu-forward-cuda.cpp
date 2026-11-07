#include "qwenpu-forward-cuda.h"
#include "qwenpu-forward-cuda-layout.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace qwenpu_forward_cuda {
namespace {

using cuda_handle = void *;
using cuda_pointer = uintptr_t;

struct driver_library {
    HMODULE handle = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);

    driver_library() {
        if (!handle) {
            throw std::runtime_error("QwenPU could not load nvcuda.dll");
        }
    }

    ~driver_library() { FreeLibrary(handle); }
};

// Keep the launcher independent of CUDA toolkit headers and import libraries.
struct driver_api {
    driver_library library;

    template<class Function>
    Function load(const char * name) const {
        const auto address = GetProcAddress(library.handle, name);
        if (!address) {
            throw std::runtime_error(std::string("QwenPU CUDA driver is missing ") + name);
        }
        return reinterpret_cast<Function>(address);
    }

#define CUDA_DRIVER(name, ...) \
    using name##_fn = int (WINAPI *)(__VA_ARGS__); \
    name##_fn name = load<name##_fn>(#name)

    CUDA_DRIVER(cuInit, unsigned int);
    CUDA_DRIVER(cuDeviceGet, int *, int);
    CUDA_DRIVER(cuDevicePrimaryCtxRetain, cuda_handle *, int);
    CUDA_DRIVER(cuDevicePrimaryCtxRelease_v2, int);
    CUDA_DRIVER(cuCtxPushCurrent_v2, cuda_handle);
    CUDA_DRIVER(cuCtxPopCurrent_v2, cuda_handle *);
    CUDA_DRIVER(cuModuleLoadData, cuda_handle *, const void *);
    CUDA_DRIVER(cuModuleGetFunction, cuda_handle *, cuda_handle, const char *);
    CUDA_DRIVER(cuModuleUnload, cuda_handle);
    CUDA_DRIVER(cuMemAlloc_v2, cuda_pointer *, size_t);
    CUDA_DRIVER(cuMemFree_v2, cuda_pointer);
    CUDA_DRIVER(cuMemcpyHtoD_v2, cuda_pointer, const void *, size_t);
    CUDA_DRIVER(cuMemsetD32_v2, cuda_pointer, unsigned int, size_t);
    CUDA_DRIVER(cuStreamCreate, cuda_handle *, unsigned int);
    CUDA_DRIVER(cuStreamDestroy_v2, cuda_handle);
    CUDA_DRIVER(cuStreamSynchronize, cuda_handle);
    CUDA_DRIVER(cuLaunchKernel, cuda_handle, unsigned int, unsigned int, unsigned int,
                unsigned int, unsigned int, unsigned int, unsigned int,
                cuda_handle, void **, void **);
    CUDA_DRIVER(cuGetErrorName, int, const char **);

#undef CUDA_DRIVER

    void check(int status, const char * action) const {
        if (status) {
            const char * name = nullptr;
            cuGetErrorName(status, &name);
            throw std::runtime_error(std::string("QwenPU CUDA ") + action + " failed (" +
                                     (name ? name : std::to_string(status)) + ")");
        }
    }
};

const driver_api & cuda() {
    static const driver_api api;
    return api;
}

struct context_scope {
    explicit context_scope(cuda_handle context) {
        cuda().check(cuda().cuCtxPushCurrent_v2(context), "push context");
    }

    ~context_scope() {
        cuda_handle context;
        cuda().cuCtxPopCurrent_v2(&context);
    }
};

struct device_buffer {
    cuda_pointer address = 0;

    device_buffer() = default;
    device_buffer(device_buffer && other) noexcept : address(std::exchange(other.address, 0)) {}
    device_buffer & operator=(device_buffer && other) noexcept {
        clear();
        address = std::exchange(other.address, 0);
        return *this;
    }
    ~device_buffer() { clear(); }

    void clear() noexcept {
        if (address) {
            cuda().cuMemFree_v2(std::exchange(address, 0));
        }
    }

    void allocate(size_t bytes) {
        if (bytes) {
            cuda().check(cuda().cuMemAlloc_v2(&address, bytes), "allocate device buffer");
        }
    }

    template<class T>
    void upload(const std::vector<T> & data) {
        const auto bytes = data.size() * sizeof(T);
        allocate(bytes);
        if (bytes) {
            cuda().check(cuda().cuMemcpyHtoD_v2(address, data.data(), bytes), "upload weights");
        }
    }
};

struct loaded_layer {
    device_buffer coefficients, offsets, scratch;
    qwenpu_cuda_layer config{};
    unsigned int lag_slot = 0;
};

static_assert(sizeof(qwenpu_cuda_layer) == 128, "QwenPU CUDA kernel argument layout requires a 64-bit host");
static_assert(sizeof(unsigned int) == sizeof(uint32_t));

unsigned int offset(size_t size) {
    if (size > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("QwenPU CUDA coefficient table is too large");
    }
    return static_cast<unsigned int>(size);
}

unsigned int append_matrix(std::vector<qwenpu_cuda_coefficient> & coefficients,
                           std::vector<unsigned int> & offsets, const rows & matrix) {
    const auto start = offset(offsets.size());
    for (const auto & row : matrix) {
        offsets.push_back(offset(coefficients.size()));
        for (const auto & item : row) {
            coefficients.push_back({item.index, item.value});
        }
    }
    offsets.push_back(offset(coefficients.size()));
    return start;
}

constexpr unsigned int threads = 128;
constexpr unsigned int stream_non_blocking = 1;

} // namespace

struct device::impl {
    int ordinal = 0;
    cuda_handle context = nullptr;
    cuda_handle module = nullptr;
    cuda_handle stream = nullptr;
    cuda_handle ffn, query_key, recurrent, residual, gather;
    std::vector<loaded_layer> layers;
    device_buffer keep;
    unsigned int keep_count = 0;

    impl(int device_ordinal, const char * cubin_path) {
        std::ifstream file(std::filesystem::u8path(cubin_path), std::ios::binary);
        file.exceptions(std::ios::badbit);
        const std::vector<char> cubin((std::istreambuf_iterator<char>(file)), {});
        if (cubin.empty()) {
            throw std::runtime_error(std::string("QwenPU could not read CUDA kernels from ") + cubin_path);
        }

        const auto & api = cuda();
        api.check(api.cuInit(0), "initialize driver");
        api.check(api.cuDeviceGet(&ordinal, device_ordinal), "select device");
        api.check(api.cuDevicePrimaryCtxRetain(&context, ordinal), "retain primary context");
        try {
            context_scope current(context);
            api.check(api.cuModuleLoadData(&module, cubin.data()), "load SM61 kernels");
            api.check(api.cuStreamCreate(&stream, stream_non_blocking), "create stream");
            ffn = function("qwenpu_ffn");
            query_key = function("qwenpu_query_key");
            recurrent = function("qwenpu_recurrent");
            residual = function("qwenpu_residual");
            gather = function("qwenpu_gather_keep");
        } catch (...) {
            close();
            throw;
        }
    }

    ~impl() { close(); }

    void close() noexcept {
        const auto & api = cuda();
        if (api.cuCtxPushCurrent_v2(context) == 0) {
            if (stream) {
                api.cuStreamSynchronize(stream);
                api.cuStreamDestroy_v2(stream);
            }
            layers.clear();
            keep.clear();
            if (module) {
                api.cuModuleUnload(module);
            }
            cuda_handle previous;
            api.cuCtxPopCurrent_v2(&previous);
        }
        api.cuDevicePrimaryCtxRelease_v2(ordinal);
    }

    cuda_handle function(const char * name) const {
        cuda_handle result;
        cuda().check(cuda().cuModuleGetFunction(&result, module, name), name);
        return result;
    }

    void launch(cuda_handle kernel, unsigned int blocks, void ** arguments) const {
        cuda().check(cuda().cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0,
                                          stream, arguments, nullptr), "launch kernel");
    }
};

device::device(int ordinal, const char * cubin_path) : gpu(std::make_unique<impl>(ordinal, cubin_path)) {}
device::~device() = default;

size_t device::add_layer(const layer_weights & weights) {
    const bool recurrent = weights.values != 0;
    const auto head_count = weights.heads * weights.width;
    const auto value_count = weights.values * weights.width;
    if (!weights.dims || weights.dims > 32768 || weights.out.size() != weights.dims ||
        weights.first.size() != weights.second.size() ||
        (recurrent && (weights.width != threads || !weights.heads || weights.heads > 64 ||
                       weights.values > 64 || weights.qhot > weights.width ||
                       weights.ref_begin >= weights.ref_end || weights.ref_end > weights.width ||
                       weights.first.size() != head_count || weights.v.size() != value_count ||
                       weights.gate.size() != value_count || weights.decay.size() != weights.values ||
                       weights.beta.size() != weights.values || weights.keys.size() != weights.values ||
                       weights.beta_const.size() != weights.values ||
                       std::any_of(weights.keys.begin(), weights.keys.end(),
                                   [&](uint32_t key) { return key >= weights.heads; })))) {
        throw std::runtime_error("invalid QwenPU CUDA layer shape");
    }

    context_scope current(gpu->context);
    loaded_layer layer;
    auto & config = layer.config;
    config.dims = weights.dims;
    config.width = weights.width;
    config.heads = weights.heads;
    config.values = weights.values;
    config.qhot = weights.qhot;
    config.ref_begin = weights.ref_begin;
    config.ref_end = weights.ref_end;
    config.projected_count = recurrent ? value_count : offset(weights.first.size());

    std::vector<qwenpu_cuda_coefficient> coefficients;
    std::vector<unsigned int> offsets;
    config.first = append_matrix(coefficients, offsets, weights.first);
    config.second = append_matrix(coefficients, offsets, weights.second);
    config.v = append_matrix(coefficients, offsets, weights.v);
    config.decay = append_matrix(coefficients, offsets, weights.decay);
    config.beta = append_matrix(coefficients, offsets, weights.beta);
    config.gate = append_matrix(coefficients, offsets, weights.gate);
    config.out = append_matrix(coefficients, offsets, weights.out);
    config.keys = offset(offsets.size());
    offsets.insert(offsets.end(), weights.keys.begin(), weights.keys.end());
    config.beta_const = offset(offsets.size());
    offsets.insert(offsets.end(), weights.beta_const.begin(), weights.beta_const.end());

    layer.coefficients.upload(coefficients);
    layer.offsets.upload(offsets);
    config.coefficients = reinterpret_cast<const qwenpu_cuda_coefficient *>(layer.coefficients.address);
    config.offsets = reinterpret_cast<const unsigned int *>(layer.offsets.address);

    const auto state_count = value_count * weights.width;
    const auto lag_count = recurrent ? 3 * weights.dims : 0;
    const size_t scratch_count = state_count + lag_count + 2 * head_count + config.projected_count;
    layer.scratch.allocate(scratch_count * sizeof(float));
    if (scratch_count) {
        cuda().check(cuda().cuMemsetD32_v2(layer.scratch.address, 0, scratch_count), "initialize scratch");
        auto scratch = reinterpret_cast<float *>(layer.scratch.address);
        config.state = scratch;
        config.lagged = scratch + state_count;
        config.q_values = config.lagged + lag_count;
        config.k_values = config.q_values + head_count;
        config.projected = config.k_values + head_count;
    }
    cuda().check(cuda().cuStreamSynchronize(nullptr), "synchronize layer upload");
    gpu->layers.push_back(std::move(layer));
    return gpu->layers.size() - 1;
}

void device::reset_layer(size_t index, const float * state, size_t count, bool reset_history) {
    auto & layer = gpu->layers.at(index);
    const auto & config = layer.config;
    if (count != config.values * config.width * config.width) {
        throw std::runtime_error("invalid QwenPU CUDA snapshot size");
    }
    if (!count) {
        return;
    }
    context_scope current(gpu->context);
    cuda().check(cuda().cuStreamSynchronize(gpu->stream), "synchronize before state reset");
    if (state) {
        cuda().check(cuda().cuMemcpyHtoD_v2(layer.scratch.address, state, count * sizeof(float)), "upload state");
    } else {
        cuda().check(cuda().cuMemsetD32_v2(layer.scratch.address, 0, count), "clear state");
    }
    if (reset_history) {
        cuda().check(cuda().cuMemsetD32_v2(reinterpret_cast<cuda_pointer>(config.lagged), 0,
                                         3 * config.dims), "clear history");
        layer.lag_slot = 0;
    }
    cuda().check(cuda().cuStreamSynchronize(nullptr), "synchronize state reset");
}

void device::set_keep(const std::vector<uint32_t> & indices) {
    if (gpu->layers.empty() || gpu->keep_count || indices.empty() ||
        *std::max_element(indices.begin(), indices.end()) >= gpu->layers.back().config.dims) {
        throw std::runtime_error("invalid QwenPU CUDA output columns");
    }
    const auto count = offset(indices.size());
    context_scope current(gpu->context);
    device_buffer keep;
    keep.upload(indices);
    cuda().check(cuda().cuStreamSynchronize(nullptr), "synchronize output column upload");
    gpu->keep = std::move(keep);
    gpu->keep_count = count;
}

void device::forward(size_t index, void * dst, const void * src) {
    auto & layer = gpu->layers.at(index);
    auto & config = layer.config;
    context_scope current(gpu->context);
    if (config.values) {
        void * args[] = {&config, &src, &layer.lag_slot};
        gpu->launch(gpu->query_key, config.heads, args);
        gpu->launch(gpu->recurrent, config.values, args);
    } else if (config.projected_count) {
        void * args[] = {&config, &src};
        gpu->launch(gpu->ffn, (config.projected_count + threads - 1) / threads, args);
    }
    void * args[] = {&config, &src, &dst, &layer.lag_slot};
    gpu->launch(gpu->residual, (config.dims + threads - 1) / threads, args);
    if (config.values) {
        layer.lag_slot = (layer.lag_slot + 1) % 3;
    }
}

void device::gather_keep(void * dst, const void * src) {
    if (!gpu->keep_count) {
        throw std::runtime_error("QwenPU CUDA output columns are not loaded");
    }
    context_scope current(gpu->context);
    void * args[] = {&gpu->keep.address, &gpu->keep_count, &src, &dst};
    gpu->launch(gpu->gather, (gpu->keep_count + threads - 1) / threads, args);
}

void device::synchronize() {
    context_scope current(gpu->context);
    cuda().check(cuda().cuStreamSynchronize(gpu->stream), "synchronize stream");
}

} // namespace qwenpu_forward_cuda
