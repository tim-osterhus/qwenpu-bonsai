#include "llama-qwenpu-forward.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "qwenpu-forward-cuda.h"

#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"
#include "gguf.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using namespace qwenpu_forward_cuda;

struct payload_slice {
    uint64_t offset;
    uint64_t size;
};

struct reader {
    std::ifstream file;
    uint64_t remaining;

    reader(const char * path, payload_slice slice, const char * format)
        : file(std::filesystem::u8path(path), std::ios::binary), remaining(slice.size) {
        const auto size = std::filesystem::file_size(std::filesystem::u8path(path));
        if (slice.offset > size || slice.size > size - slice.offset) {
            throw std::runtime_error("QwenPU payload exceeds file bounds");
        }
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.seekg(static_cast<std::streamoff>(slice.offset));
        char magic[4];
        read(magic, sizeof(magic));
        if (std::memcmp(magic, format, sizeof(magic)) != 0) {
            throw std::runtime_error("invalid QwenPU payload format");
        }
    }

    void read(void * data, size_t size) {
        if (size > remaining) {
            throw std::runtime_error("QwenPU payload ended before its declared data");
        }
        file.read(static_cast<char *>(data), size);
        remaining -= size;
    }

    uint32_t number() {
        uint32_t n;
        read(&n, sizeof(n));
        return n;
    }

    rows matrix(uint32_t columns) {
        const auto n = number();
        if (n > 1000000 || n > remaining / sizeof(uint32_t)) {
            throw std::runtime_error("invalid QwenPU matrix size");
        }
        rows result(n);
        for (auto & row : result) {
            const auto count = number();
            if (count > columns || count > remaining / (sizeof(uint32_t) + sizeof(float))) {
                throw std::runtime_error("invalid QwenPU sparse row");
            }
            row.reserve(count);
            for (uint32_t j = 0; j < count; ++j) {
                coefficient item{number(), 0};
                read(&item.value, sizeof(item.value));
                if (item.index >= columns || !std::isfinite(item.value) ||
                        (!row.empty() && item.index <= row.back().index)) {
                    throw std::runtime_error("invalid QwenPU sparse coefficient");
                }
                row.push_back(item);
            }
        }
        return result;
    }

    void finish() const {
        if (remaining != 0) {
            throw std::runtime_error("unexpected trailing QwenPU payload data");
        }
    }
};

payload_slice find_payload(const gguf_context * metadata, const char * name) {
    const auto id = gguf_find_tensor(metadata, name);
    if (id < 0 || gguf_get_tensor_type(metadata, id) != GGML_TYPE_I8 || gguf_get_tensor_size(metadata, id) == 0) {
        throw std::runtime_error(std::string("missing or invalid QwenPU payload: ") + name);
    }
    return {gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, id), gguf_get_tensor_size(metadata, id)};
}

std::string cubin_path() {
    if (const char * path = std::getenv("QWENPU_FORWARD_CUBIN")) {
        return path;
    }
    HMODULE dll = nullptr;
    wchar_t path[4096];
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&cubin_path), &dll)) {
        throw std::runtime_error("could not locate QwenPU runtime DLL");
    }
    const auto length = GetModuleFileNameW(dll, path, sizeof(path) / sizeof(*path));
    if (length == 0 || length == sizeof(path) / sizeof(*path)) {
        throw std::runtime_error("could not locate QwenPU runtime DLL path");
    }
    return (std::filesystem::path(path).parent_path() / "qwenpu-forward-cuda.cubin").u8string();
}

layer_weights read_layer(reader & r, uint32_t dims, uint32_t width, uint32_t ref_begin, uint32_t ref_end) {
    layer_weights op;
    op.dims = dims;
    op.width = width;
    op.ref_begin = ref_begin;
    op.ref_end = ref_end;
    const auto kind = r.number();
    if (kind == 0) {
        op.first = r.matrix(dims);
        op.second = r.matrix(dims);
        op.out = r.matrix(static_cast<uint32_t>(op.first.size()));
        if (op.first.empty() || op.first.size() != op.second.size() || op.out.size() != dims) {
            throw std::runtime_error("invalid QwenPU FFN shape");
        }
        return op;
    }
    op.heads = r.number();
    op.values = r.number();
    op.qhot = r.number();
    if (kind != 1 || op.heads == 0 || op.heads > 64 || op.values == 0 || op.values > 64 || op.qhot > width) {
        throw std::runtime_error("invalid QwenPU recurrent layer shape");
    }
    for (uint32_t i = 0; i < op.values; ++i) {
        op.keys.push_back(r.number());
        if (op.keys.back() >= op.heads) {
            throw std::runtime_error("invalid QwenPU recurrent head");
        }
    }
    for (uint32_t i = 0; i < op.values; ++i) {
        op.beta_const.push_back(r.number());
        if (op.beta_const.back() > 1) {
            throw std::runtime_error("invalid QwenPU beta");
        }
    }
    op.first = r.matrix(4 * dims);
    op.second = r.matrix(4 * dims);
    op.v = r.matrix(4 * dims);
    op.decay = r.matrix(dims);
    op.beta = r.matrix(dims);
    op.gate = r.matrix(dims);
    op.out = r.matrix(op.values * width);
    if (op.first.size() != op.heads * width || op.second.size() != op.heads * width ||
        op.v.size() != op.values * width || op.decay.size() != op.values || op.beta.size() != op.values ||
        op.gate.size() != op.values * width || op.out.size() != dims) {
        throw std::runtime_error("invalid QwenPU recurrent projection shape");
    }
    return op;
}

struct family {
    uint32_t base;
    uint32_t count;
    bool control;
    bool lookup;
    rows embedding;

    bool contains(uint32_t id) const {
        return id >= base && id - base < (control ? 1 : count);
    }
};

void gpu_only(ggml_tensor *, int, int, void *) {
    GGML_ABORT("QwenPU custom node assigned to the CPU");
}

} // namespace

struct llama_qwenpu_forward::impl {
    struct layer_info {
        size_t index;
        uint32_t state_size;
    };

    std::string path;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata;
    ggml_backend_t cuda_backend;
    ggml_guid guid = { 0x1e, 0x22, 0x9e, 0xc1, 0x1c, 0x25, 0x4a, 0xa9, 0xb7, 0xe4, 0x8b, 0x77, 0xc2, 0xe0, 0x51, 0x42 };
    ggml_backend_device packed_device = {};
    ggml_backend packed_backend = {};
    device gpu;
    std::vector<layer_info> layers;
    llama_qwenpu_forward_info stats = {};
    uint32_t language_width;
    uint32_t fam_bits;
    uint32_t pay_bits;
    uint32_t q_family;
    std::vector<uint32_t> printable;
    std::vector<family> families;
    rows text_embedding;
    std::vector<uint8_t> text_bytes;
    std::unordered_map<llama_token, row> special;
    std::vector<float> embedding;
    std::vector<float> kept;
    bool failed = false;

    static int ordinal(ggml_backend_t backend) {
        const char * name = ggml_backend_dev_name(ggml_backend_get_device(backend));
        if (std::strncmp(name, "CUDA", 4) != 0 || name[4] == '\0') {
            throw std::runtime_error("QwenPU forward requires a native NVIDIA CUDA backend");
        }
        return std::stoi(name + 4);
    }

    impl(ggml_backend_t cuda, const char * file, uint32_t vocab, uint32_t language_width)
        : path(file), metadata(gguf_init_from_file(file, {true, nullptr}), gguf_free),
          cuda_backend(cuda), gpu(ordinal(cuda), cubin_path().c_str()), language_width(language_width) {
        if (!metadata) {
            throw std::runtime_error("could not read the QwenPU checkpoint directory");
        }
        reader r(file, find_payload(metadata.get(), qwenpu_forward_graph_tensor), "QPF1");
        const auto dims = r.number();
        const auto width = r.number();
        const auto ref_begin = r.number();
        const auto ref_end = r.number();
        fam_bits = r.number();
        pay_bits = r.number();
        // The QPC2 header carries GO, HALT and the BYTE range for its standalone driver.
        for (int i = 0; i < 4; ++i) {
            r.number();
        }
        const auto count = r.number();
        if (dims == 0 || dims > 32768 || width != 128 || ref_begin >= ref_end || ref_end > width ||
            count == 0 || count > 64 || fam_bits == 0 || fam_bits > 8 || pay_bits == 0 || pay_bits > 16) {
            throw std::runtime_error("invalid QwenPU graph shape");
        }
        for (uint32_t i = 0; i < count; ++i) {
            auto op = read_layer(r, dims, width, ref_begin, ref_end);
            layers.push_back({gpu.add_layer(op), op.values * width * width});
        }
        const auto keep = r.matrix(dims);
        if (keep.size() != fam_bits + pay_bits + 2) {
            throw std::runtime_error("invalid QwenPU code, tau and mu shape");
        }
        std::vector<uint32_t> indices;
        for (const auto & row : keep) {
            if (row.size() != 1 || row.front().value != 1.0f) {
                throw std::runtime_error("QwenPU retained lanes must select exactly one residual coordinate");
            }
            indices.push_back(row.front().index);
        }
        gpu.set_keep(indices);
        kept.resize(indices.size());
        const auto family_count = r.number();
        if (family_count == 0 || family_count > (1u << fam_bits)) {
            throw std::runtime_error("invalid QwenPU token families");
        }
        for (uint32_t i = 0; i < family_count; ++i) {
            family entry{r.number(), r.number(), r.number() != 0, r.number() != 0, r.matrix(dims)};
            const uint32_t size = entry.control ? 1 : entry.count;
            size_t expected = size == 0 ? 0 : entry.lookup ? size : 1;
            if (!entry.lookup && size != 0) {
                for (auto mask = size - 1; mask; mask >>= 1) {
                    ++expected;
                }
            }
            if (entry.embedding.size() != expected ||
                    uint64_t(vocab) + entry.base + size > std::numeric_limits<llama_token>::max()) {
                throw std::runtime_error("invalid QwenPU token family");
            }
            families.push_back(std::move(entry));
        }
        stats.text_vocab = r.number();
        stats.chip_width = dims;
        stats.widened_width = r.number();
        stats.chip_layers = count;
        stats.prediction = -1;
        stats.last_text_position = -1;
        q_family = r.number();
        const auto printable_count = r.number();
        if (stats.text_vocab != vocab || stats.widened_width < language_width + keep.size() ||
            q_family >= family_count || families[q_family].control || families[q_family].count != 0 ||
            printable_count == 0 || printable_count > (1u << pay_bits)) {
            throw std::runtime_error("QwenPU graph does not match the Bonsai vocabulary or layout");
        }
        for (uint32_t i = 0; i < printable_count; ++i) {
            printable.push_back(r.number());
            if (printable.back() >= vocab) {
                throw std::runtime_error("invalid QwenPU printable text token");
            }
        }
        text_embedding = r.matrix(dims);
        if (text_embedding.size() != 1 + 16 * 8) {
            throw std::runtime_error("invalid QwenPU text feature table");
        }
        text_bytes.resize(size_t(vocab) * 16);
        r.read(text_bytes.data(), text_bytes.size());
        const auto special_count = r.number();
        if (special_count > vocab) {
            throw std::runtime_error("invalid QwenPU special text token count");
        }
        std::vector<uint32_t> special_ids;
        for (uint32_t i = 0; i < special_count; ++i) {
            special_ids.push_back(r.number());
        }
        auto special_rows = r.matrix(dims);
        if (special_rows.size() != special_ids.size()) {
            throw std::runtime_error("invalid QwenPU special text feature table");
        }
        for (size_t i = 0; i < special_ids.size(); ++i) {
            if (special_ids[i] >= vocab || !special.emplace(special_ids[i], std::move(special_rows[i])).second) {
                throw std::runtime_error("invalid QwenPU special text token");
            }
        }
        r.finish();
        embedding.resize(dims);
        init_backend();
        reset(nullptr);
        LLAMA_LOG_INFO("QwenPU forward: %u CUDA layers, language width %u, packed circuit width %u, logical residual width %u\n",
            count, language_width, dims, stats.widened_width);
    }

    void init_backend() {
        packed_device.context = this;
        packed_device.iface.get_name = [](ggml_backend_dev_t) { return "QWENPU"; };
        packed_device.iface.get_description = [](ggml_backend_dev_t) { return "Packed QwenPU layers on native CUDA buffers"; };
        packed_device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        packed_device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return ggml_backend_get_default_buffer_type(static_cast<impl *>(dev->context)->cuda_backend);
        };
        packed_device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
            return buft == ggml_backend_dev_buffer_type(dev);
        };
        packed_device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * node) {
            return node->op == GGML_OP_CUSTOM && std::strncmp(node->name, "qwenpu.", 7) == 0;
        };
        packed_backend.guid = &guid;
        packed_backend.device = &packed_device;
        packed_backend.context = this;
        packed_backend.iface.get_name = [](ggml_backend_t) { return "QWENPU-CUDA"; };
        packed_backend.iface.graph_compute = [](ggml_backend_t backend, ggml_cgraph * graph) {
            auto & self = *static_cast<impl *>(backend->context);
            try {
                // The native CUDA backend and the packed executor share one device's
                // buffers. Finish native producers before the custom split, then make
                // its outputs visible to the next native split.
                ggml_backend_synchronize(self.cuda_backend);
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                    auto * node = ggml_graph_node(graph, i);
                    if (node->op == GGML_OP_NONE) {
                        continue;
                    }
                    if (!ggml_backend_supports_op(backend, node)) {
                        throw std::runtime_error("non-QwenPU operation in packed split");
                    }
                    ggml_custom_op_params params;
                    std::memcpy(&params, node->op_params, sizeof(params));
                    if (params.userdata) {
                        const auto & op = *static_cast<const layer_info *>(params.userdata);
                        self.gpu.forward(op.index, node->data, node->src[0]->data);
                        ++self.stats.cuda_layer_calls;
                    } else {
                        self.gpu.gather_keep(node->data, node->src[0]->data);
                    }
                }
                self.gpu.synchronize();
                return GGML_STATUS_SUCCESS;
            } catch (const std::exception & e) {
                self.failed = true;
                LLAMA_LOG_ERROR("QwenPU CUDA split: %s\n", e.what());
                return GGML_STATUS_FAILED;
            }
        };
    }

    void reset(const char * name) {
        std::vector<row> initial(layers.size());
        if (name) {
            if (std::string(name).find(qwenpu_forward_program_prefix) != 0) {
                throw std::runtime_error("invalid QwenPU program tensor name");
            }
            reader r(path.c_str(), find_payload(metadata.get(), name), "QPS1");
            if (r.number() != layers.size()) {
                throw std::runtime_error("QwenPU snapshot layer count mismatch");
            }
            for (size_t i = 0; i < layers.size(); ++i) {
                auto matrix = r.matrix(layers[i].state_size);
                if (matrix.size() != 1) {
                    throw std::runtime_error("invalid QwenPU snapshot layer");
                }
                initial[i] = std::move(matrix.front());
            }
            r.finish();
        }
        failed = true;
        for (size_t i = 0; i < layers.size(); ++i) {
            std::vector<float> dense(layers[i].state_size, 0.0f);
            for (const auto & item : initial[i]) {
                dense[item.index] = item.value;
            }
            gpu.reset_layer(layers[i].index, dense.data(), dense.size(), name == nullptr);
        }
        if (!name) {
            stats.last_text_position = -1;
            stats.text_steps = 0;
            stats.chip_only_steps = 0;
            stats.cuda_layer_calls = 0;
        }
        stats.prediction = -1;
        stats.kept_l2 = 0.0f;
        failed = false;
    }

    const family * chip_family(llama_token token) const {
        const auto raw = uint32_t(token) - stats.text_vocab;
        const auto it = std::find_if(families.begin(), families.end(), [raw](const family & f) { return f.contains(raw); });
        return it == families.end() ? nullptr : &*it;
    }

    void embed(ggml_tensor * input, llama_token token) {
        std::fill(embedding.begin(), embedding.end(), 0.0f);
        const auto add = [&](const row & features) {
            for (const auto & item : features) {
                embedding[item.index] += item.value;
            }
        };
        if (uint32_t(token) < stats.text_vocab) {
            const auto it = special.find(token);
            if (it != special.end()) {
                add(it->second);
            } else {
                add(text_embedding.front());
                for (size_t i = 0; i < 16; ++i) {
                    auto bits = text_bytes[size_t(token) * 16 + i];
                    for (size_t b = 0; bits; ++b, bits >>= 1) {
                        if (bits & 1) {
                            add(text_embedding[1 + i * 8 + b]);
                        }
                    }
                }
            }
        } else {
            const auto & f = *chip_family(token);
            auto payload = uint32_t(token) - stats.text_vocab - f.base;
            add(f.embedding[f.lookup ? payload : 0]);
            if (!f.lookup) {
                for (size_t i = 1; payload; ++i, payload >>= 1) {
                    if (payload & 1) {
                        add(f.embedding[i]);
                    }
                }
            }
        }
        ggml_backend_tensor_set(input, embedding.data(), 0, embedding.size() * sizeof(float));
    }

    llama_token prediction(bool chip_only) const {
        if (kept[fam_bits + pay_bits] != 0.0f) {
            if (chip_only) {
                throw std::runtime_error("QwenPU gave up a chip-only position");
            }
            return -1;
        }
        uint32_t code = 0;
        for (uint32_t i = 0; i < fam_bits + pay_bits; ++i) {
            if (kept[i] == 0.0f) {
                throw std::runtime_error("QwenPU owned a position with an unset code lane");
            }
            code |= uint32_t(kept[i] > 0.0f) << i;
        }
        const auto id = code & ((1u << fam_bits) - 1);
        const auto payload = code >> fam_bits;
        if (id == q_family && payload < printable.size()) {
            return printable[payload];
        }
        if (id < families.size() && id != q_family) {
            const auto & f = families[id];
            if (f.control || payload < f.count) {
                return stats.text_vocab + f.base + (f.control ? 0 : payload);
            }
        }
        throw std::runtime_error("QwenPU predicted an invalid token");
    }

    struct input : llm_graph_input_i {
        impl & owner;
        ggml_tensor * tensor;
        input(impl & owner, ggml_tensor * tensor) : owner(owner), tensor(tensor) {}
        void set_input(const llama_ubatch * ubatch) override { owner.embed(tensor, ubatch->token[0]); }
        bool can_reuse(const llm_graph_params & params) override { return params.ubatch.n_tokens == 1 && params.ubatch.token; }
    };
};

llama_qwenpu_forward::llama_qwenpu_forward(ggml_backend_t cuda, const char * path, uint32_t vocab, uint32_t language_width)
    : p(std::make_unique<impl>(cuda, path, vocab, language_width)) {}

llama_qwenpu_forward::~llama_qwenpu_forward() = default;

ggml_backend_t llama_qwenpu_forward::backend() const {
    return &p->packed_backend;
}

ggml_tensor * llama_qwenpu_forward::build(const llm_graph_params & params) {
    auto * ctx = params.res->get_ctx();
    auto * graph = params.res->get_gf();
    auto * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p->stats.chip_width);
    ggml_set_name(cur, "qwenpu.input");
    ggml_set_input(cur);
    ggml_backend_sched_set_tensor_backend(params.sched, cur, backend());
    params.res->add_input(std::make_unique<impl::input>(*p, cur));
    for (auto & layer : p->layers) {
        cur = ggml_custom_4d(ctx, GGML_TYPE_F32, p->stats.chip_width, 1, 1, 1,
                             &cur, 1, gpu_only, 1, &layer);
        ggml_format_name(cur, "qwenpu.%s.%zu", layer.state_size ? "recurrent" : "ffn", layer.index);
        ggml_backend_sched_set_tensor_backend(params.sched, cur, backend());
    }
    cur = ggml_custom_4d(ctx, GGML_TYPE_F32, p->kept.size(), 1, 1, 1, &cur, 1, gpu_only, 1, nullptr);
    ggml_set_name(cur, "qwenpu.kept");
    ggml_backend_sched_set_tensor_backend(params.sched, cur, backend());
    params.res->t_qwenpu_kept = cur;
    ggml_set_output(cur);
    ggml_build_forward_expand(graph, cur);
    return cur;
}

ggml_tensor * llama_qwenpu_forward::chip_mean_square(
        ggml_context * ctx, ggml_backend_sched_t sched, ggml_tensor * kept) const {
    auto * square = ggml_sqr(ctx, kept);
    auto * sum = ggml_sum_rows(ctx, square);
    auto * mean = ggml_scale(ctx, sum, 1.0f / p->language_width);
    ggml_backend_sched_set_tensor_backend(sched, square, p->cuda_backend);
    ggml_backend_sched_set_tensor_backend(sched, sum, p->cuda_backend);
    ggml_backend_sched_set_tensor_backend(sched, mean, p->cuda_backend);
    ggml_set_name(mean, "qwenpu.kept_mean_square");
    return mean;
}

ggml_tensor * llama_qwenpu_forward::language_norm(
        ggml_context * ctx, ggml_tensor * text, ggml_tensor * chip_mean, float eps) const {
    // Zero-padded language weights with gamma multiplied by sqrt(D / W)
    // read x / sqrt((sum(x²) + sum(kept²)) / D + eps * W / D).
    auto * text_mean = ggml_mean(ctx, ggml_sqr(ctx, text));
    auto * variance = ggml_add(ctx, text_mean, chip_mean);
    auto * denominator = ggml_sqrt(ctx, ggml_scale_bias(ctx, variance, 1.0f, eps * p->stats.widened_width / p->language_width));
    return ggml_div(ctx, text, denominator);
}

bool llama_qwenpu_forward::accepts(llama_token token) const {
    return !p->failed && token >= 0 && (uint32_t(token) < p->stats.text_vocab || p->chip_family(token));
}

void llama_qwenpu_forward::reset(const char * program) {
    p->reset(program);
}

bool llama_qwenpu_forward::read_output(ggml_tensor * kept, bool chip_only, llama_pos pos) {
    try {
        ggml_backend_tensor_get(kept, p->kept.data(), 0, p->kept.size() * sizeof(float));
        float squared = 0;
        for (const auto v : p->kept) {
            squared += v * v;
        }
        if (!std::isfinite(squared)) {
            throw std::runtime_error("nonfinite QwenPU retained residual");
        }
        p->stats.prediction = p->prediction(chip_only);
        p->stats.kept_l2 = std::sqrt(squared);
        if (chip_only) {
            ++p->stats.chip_only_steps;
        } else {
            ++p->stats.text_steps;
            p->stats.last_text_position = pos;
        }
        return true;
    } catch (const std::exception & e) {
        p->failed = true;
        LLAMA_LOG_ERROR("QwenPU output: %s\n", e.what());
        return false;
    }
}

const llama_qwenpu_forward_info & llama_qwenpu_forward::info() const {
    return p->stats;
}
