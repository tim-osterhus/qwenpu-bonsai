#include "qwenpu-circuit.h"
#include "llama.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
struct payload_slice {
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct coefficient {
    uint32_t index;
    float value;
};
using row = std::vector<coefficient>;
using rows = std::vector<row>;

struct reader {
    std::ifstream file;
    uint64_t remaining;

    reader(const char * path, const char * format, payload_slice slice = {})
        : file(std::filesystem::u8path(path), std::ios::binary) {
        const auto size = std::filesystem::file_size(std::filesystem::u8path(path));
        if (slice.offset > size || slice.size > size - slice.offset) {
            throw std::runtime_error("QwenPU payload exceeds file bounds");
        }
        remaining = slice.size ? slice.size : size;
        char magic[4];
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.seekg(static_cast<std::streamoff>(slice.offset));
        read(magic, sizeof(magic));
        if (std::memcmp(magic, format, sizeof(magic)) != 0) {
            throw std::runtime_error("invalid QwenPU fixture header");
        }
    }

    void read(char * data, size_t size) {
        if (size > remaining) {
            throw std::runtime_error("QwenPU payload ended before its declared data");
        }
        file.read(data, size);
        remaining -= size;
    }

    uint32_t number() {
        uint32_t n;
        read(reinterpret_cast<char *>(&n), sizeof(n));
        return n;
    }

    rows matrix(uint32_t columns) {
        const auto n = number();
        if (n > 1000000) {
            throw std::runtime_error("invalid QwenPU matrix");
        }
        rows result(n);
        for (auto & r : result) {
            const auto count = number();
            if (count > columns) {
                throw std::runtime_error("invalid QwenPU sparse row");
            }
            for (uint32_t j = 0; j < count; ++j) {
                coefficient item{number(), 0};
                read(reinterpret_cast<char *>(&item.value), sizeof(item.value));
                if (item.index >= columns) {
                    throw std::runtime_error("invalid QwenPU coefficient index");
                }
                r.push_back(item);
            }
        }
        return result;
    }
};

float dot(const row & coefficients, const float * x) {
    float sum = 0;
    for (const auto & item : coefficients) {
        sum += x[item.index] * item.value;
    }
    return sum;
}

struct layer {
    uint32_t dims = 0, width = 0, heads = 0, values = 0, qhot = 0, ref_begin = 0, ref_end = 0;
    rows first, second, v, decay, beta, gate, out;
    std::vector<uint32_t> keys, beta_const;
    std::vector<float> state, lagged, q_values, k_values, projected;

    layer(reader & r, uint32_t d, uint32_t w, uint32_t begin, uint32_t end)
        : dims(d), width(w), ref_begin(begin), ref_end(end) {
        const auto kind = r.number();
        if (kind == 0) {
            first = r.matrix(dims);
            second = r.matrix(dims);
            out = r.matrix(static_cast<uint32_t>(first.size()));
            if (first.size() != second.size() || out.size() != dims) {
                throw std::runtime_error("invalid QwenPU FFN shape");
            }
            projected.resize(first.size());
            return;
        }
        heads = r.number();
        values = r.number();
        qhot = r.number();
        if (kind != 1 || heads == 0 || heads > 64 || values == 0 || values > 64 || qhot > width) {
            throw std::runtime_error("invalid QwenPU recurrent layer shape");
        }
        for (uint32_t i = 0; i < values; ++i) {
            keys.push_back(r.number());
            if (keys.back() >= heads) {
                throw std::runtime_error("invalid QwenPU recurrent head");
            }
        }
        for (uint32_t i = 0; i < values; ++i) {
            beta_const.push_back(r.number());
        }
        first = r.matrix(4 * dims);
        second = r.matrix(4 * dims);
        v = r.matrix(4 * dims);
        decay = r.matrix(dims);
        beta = r.matrix(dims);
        gate = r.matrix(dims);
        out = r.matrix(values * width);
        if (first.size() != heads * width || second.size() != heads * width ||
            v.size() != values * width || decay.size() != values || beta.size() != values ||
            gate.size() != values * width || out.size() != dims) {
            throw std::runtime_error("invalid QwenPU recurrent projection shape");
        }
        state.resize(values * width * width);
        lagged.resize(4 * dims);
        q_values.resize(heads * width);
        k_values.resize(heads * width);
        projected.resize(values * width);
    }

    void evaluate(float * dst, const float * src) {
        if (values == 0) {
            for (size_t i = 0; i < first.size(); ++i) {
                const auto rounded = ggml_bf16_to_fp32(ggml_fp32_to_bf16(dot(second[i], src)));
                projected[i] = std::max(0.0f, rounded) * dot(first[i], src);
            }
        } else {
            recurrent(src);
        }
        for (uint32_t d = 0; d < dims; ++d) {
            dst[d] = src[d] + dot(out[d], projected.data());
        }
    }

    void recurrent(const float * src) {
        std::copy_n(src, dims, lagged.data());
        for (size_t i = 0; i < q_values.size(); ++i) {
            q_values[i] = std::max(0.0f, dot(first[i], lagged.data()));
            k_values[i] = std::max(0.0f, dot(second[i], lagged.data()));
        }
        for (uint32_t h = 0; h < values; ++h) {
            auto * matrix = state.data() + h * width * width;
            if (!decay[h].empty()) {
                const auto a = dot(decay[h], src);
                const auto factor = a == 0 ? 1.0f : a == 110 ? 0.5f : 0.0f;
                if (factor != 1) {
                    for (uint32_t i = 0; i < width * width; ++i) {
                        matrix[i] *= factor;
                    }
                }
            }
            const auto * key = k_values.data() + keys[h] * width;
            const auto * maximum = std::max_element(key, key + width);
            if (*maximum > 0 && dot(beta[h], src) + beta_const[h] == 1) {
                auto * row_ptr = matrix + (maximum - key) * width;
                for (uint32_t col = 0; col < width; ++col) {
                    row_ptr[col] = std::max(0.0f, dot(v[h * width + col], lagged.data()));
                }
            }
            const auto * query = q_values.data() + keys[h] * width;
            auto * read = projected.data() + h * width;
            std::fill_n(read, width, 0.0f);
            uint32_t active = 0;
            for (uint32_t i = 0; i < width; ++i) {
                if (query[i] > 0) {
                    ++active;
                    for (uint32_t col = 0; col < width; ++col) {
                        read[col] += matrix[i * width + col];
                    }
                }
            }
            if (active > qhot) {
                std::fill_n(read, width, NAN);
                continue;
            }
            float sum = 0;
            for (uint32_t col = ref_begin; col < ref_end; ++col) {
                sum += read[col] * read[col];
            }
            const float norm = sum > 0 ? std::sqrt(sum / width) : 1.0f;
            for (uint32_t col = 0; col < width; ++col) {
                read[col] = sum > 0 ? read[col] / norm *
                    std::max(0.0f, dot(gate[h * width + col], src)) : 0.0f;
            }
        }
        std::memmove(lagged.data() + dims, lagged.data(), 3 * dims * sizeof(float));
    }

    static void compute(ggml_tensor * dst, const ggml_tensor * src, int, int, void * user) {
        static_cast<layer *>(user)->evaluate(static_cast<float *>(dst->data),
                                             static_cast<const float *>(src->data));
    }
};

struct family {
    uint32_t base, size, control;
    bool lookup;
    rows embedding;
};
} // namespace

struct llama_qwenpu_circuit::impl {
    ggml_backend_t backend;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * residual = nullptr;
    std::vector<layer> layers;
    uint32_t dims, fam_bits, pay_bits;
    llama_qwenpu_tokens tokens{};
    rows readout;
    std::vector<family> families;
    std::vector<float> data;
    bool ready = false;
    std::string checkpoint_path;
    std::array<payload_slice, qwenpu_tensor_names.size()> payloads{};

    impl(ggml_backend_t cpu, const char * path, bool checkpoint) : backend(cpu) {
        if (!backend) {
            throw std::runtime_error("QwenPU requires a GGML CPU backend");
        }
        if (checkpoint) {
            const std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
                gguf_init_from_file(path, {true, nullptr}), gguf_free);
            if (!metadata) {
                throw std::runtime_error("could not read the QwenPU checkpoint directory");
            }
            for (size_t i = 0; i < payloads.size(); ++i) {
                const auto tensor = gguf_find_tensor(metadata.get(), qwenpu_tensor_names[i]);
                if (tensor < 0 || gguf_get_tensor_type(metadata.get(), tensor) != GGML_TYPE_I8 ||
                    gguf_get_tensor_size(metadata.get(), tensor) == 0) {
                    throw std::runtime_error("missing or invalid QwenPU checkpoint payload");
                }
                payloads[i] = {gguf_get_data_offset(metadata.get()) + gguf_get_tensor_offset(metadata.get(), tensor),
                               gguf_get_tensor_size(metadata.get(), tensor)};
            }
            checkpoint_path = path;
        }
        reader r(path, "QPC2", payloads[0]);
        dims = r.number();
        const auto width = r.number();
        const auto ref_begin = r.number();
        const auto ref_end = r.number();
        fam_bits = r.number();
        pay_bits = r.number();
        tokens = {r.number(), r.number(), r.number(), r.number()};
        const auto count = r.number();
        if (dims == 0 || dims > 32768 || width != 128 || ref_begin >= ref_end || ref_end > width ||
            count == 0 || count > 64 || fam_bits > 8 || pay_bits > 16) {
            throw std::runtime_error("invalid QwenPU graph shape");
        }
        layers.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            layers.emplace_back(r, dims, width, ref_begin, ref_end);
        }
        readout = r.matrix(dims);
        const auto family_count = r.number();
        if (family_count > (1u << fam_bits) || readout.size() != fam_bits + pay_bits) {
            throw std::runtime_error("invalid QwenPU readout shape");
        }
        for (uint32_t i = 0; i < family_count; ++i) {
            family entry{r.number(), r.number(), r.number(), r.number() != 0, r.matrix(dims)};
            size_t expected = entry.control ? 1 : 0;
            if (entry.size) {
                expected = entry.lookup ? entry.size : 1;
                if (!entry.lookup) {
                    for (auto mask = entry.size - 1; mask; mask >>= 1) {
                        ++expected;
                    }
                }
            }
            if (entry.embedding.size() != expected) {
                throw std::runtime_error("invalid QwenPU embedding shape");
            }
            families.push_back(std::move(entry));
        }
        data.resize(dims);
        context.reset(ggml_init({(count + 2) * ggml_tensor_overhead() +
                                  ggml_graph_overhead_custom(count + 2, false),
                                 nullptr, true}));
        if (!context) {
            throw std::runtime_error("could not allocate QwenPU graph metadata");
        }
        graph = ggml_new_graph_custom(context.get(), count + 2, false);
        input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, dims);
        ggml_set_input(input);
        residual = input;
        for (auto & op : layers) {
            residual = ggml_map_custom1(context.get(), residual, layer::compute, 1, &op);
            ggml_set_name(residual, op.values == 0 ? "qwenpu.ffn" : "qwenpu.recurrent");
        }
        ggml_set_output(residual);
        ggml_build_forward_expand(graph, residual);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        if (!buffer) {
            throw std::runtime_error("could not allocate QwenPU graph tensors");
        }
    }

    void reset(const char * path, payload_slice slice = {}) {
        reader r(path, "QPS1", slice);
        if (r.number() != layers.size()) {
            throw std::runtime_error("QwenPU snapshot layer count mismatch");
        }
        rows state;
        for (const auto & op : layers) {
            auto matrix = r.matrix(static_cast<uint32_t>(op.state.size()));
            if (matrix.size() != 1) {
                throw std::runtime_error("invalid QwenPU snapshot layer");
            }
            state.push_back(std::move(matrix.front()));
        }
        for (size_t i = 0; i < layers.size(); ++i) {
            auto & op = layers[i];
            std::fill(op.state.begin(), op.state.end(), 0.0f);
            std::fill(op.lagged.begin(), op.lagged.end(), 0.0f);
            for (const auto & item : state[i]) {
                op.state[item.index] = item.value;
            }
        }
        ready = true;
    }

    void embed(uint32_t token) {
        for (const auto & entry : families) {
            if (token < entry.base || token - entry.base >= (entry.control ? 1 : entry.size)) {
                continue;
            }
            std::fill(data.begin(), data.end(), 0.0f);
            auto add = [&](const row & features) {
                for (const auto & item : features) {
                    data[item.index] += item.value;
                }
            };
            auto payload = token - entry.base;
            add(entry.embedding[entry.lookup ? payload : 0]);
            if (!entry.lookup) {
                for (size_t bit = 1; payload; payload >>= 1, ++bit) {
                    if (payload & 1) {
                        add(entry.embedding[bit]);
                    }
                }
            }
            return;
        }
        throw std::runtime_error("invalid QwenPU input token");
    }

    int32_t step(uint32_t token) {
        if (!ready) {
            throw std::runtime_error("QwenPU program state is not loaded");
        }
        embed(token);
        ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("QwenPU graph computation failed");
        }
        ggml_backend_tensor_get(residual, data.data(), 0, data.size() * sizeof(float));
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < readout.size(); ++bit) {
            const float value = std::nearbyint(dot(readout[bit], data.data()));
            if (value != 0 && value != 1) {
                throw std::runtime_error("QwenPU predicted a non-binary readout");
            }
            code |= static_cast<uint32_t>(value) << bit;
        }
        const auto id = code & ((1u << fam_bits) - 1);
        const auto payload = code >> fam_bits;
        if (id >= families.size() || (!families[id].control && payload >= families[id].size)) {
            throw std::runtime_error("QwenPU predicted an invalid trace token");
        }
        return families[id].base + (families[id].control ? 0 : payload);
    }
};

llama_qwenpu_circuit::llama_qwenpu_circuit(ggml_backend_t cpu, const char * path, bool checkpoint)
    : graph(std::make_unique<impl>(cpu, path, checkpoint)) {}

llama_qwenpu_circuit::~llama_qwenpu_circuit() = default;

void llama_qwenpu_circuit::reset(const char * path) {
    graph->reset(path);
}

void llama_qwenpu_circuit::reset_program(uint32_t program) {
    if (graph->checkpoint_path.empty() || program == 0 || program >= graph->payloads.size()) {
        throw std::runtime_error("invalid QwenPU checkpoint program");
    }
    graph->reset(graph->checkpoint_path.c_str(), graph->payloads[program]);
}

llama_qwenpu_tokens llama_qwenpu_circuit::tokens() const {
    return graph->tokens;
}

int32_t llama_qwenpu_circuit::step(uint32_t token) {
    return graph->step(token);
}

