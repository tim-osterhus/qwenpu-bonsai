#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace qwenpu_forward_cuda {

struct coefficient {
    uint32_t index;
    float value;
};
using row = std::vector<coefficient>;
using rows = std::vector<row>;

struct layer_weights {
    uint32_t dims = 0, width = 0, heads = 0, values = 0, qhot = 0, ref_begin = 0, ref_end = 0;
    rows first, second, v, decay, beta, gate, out;
    std::vector<uint32_t> keys, beta_const;
};

class device {
public:
    device(int ordinal, const char * cubin_path);
    ~device();

    size_t add_layer(const layer_weights & weights);
    // Synchronize work submitted for this layer before replacing its state.
    void reset_layer(size_t layer, const float * state, size_t count, bool reset_history = true);
    void set_keep(const std::vector<uint32_t> & indices);

    // src and dst are distinct contiguous F32 CUDA buffers on this device.
    // Dispatches run on the executor's nonblocking stream until synchronize().
    void forward(size_t layer, void * dst, const void * src);
    void gather_keep(void * dst, const void * src);
    void synchronize();

private:
    struct impl;
    std::unique_ptr<impl> gpu;
};

} // namespace qwenpu_forward_cuda
