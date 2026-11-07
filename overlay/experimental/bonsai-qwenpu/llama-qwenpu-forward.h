#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <memory>

inline constexpr const char * qwenpu_forward_version_key = "qwenpu.forward_version";
inline constexpr const char * qwenpu_forward_graph_tensor = "qwenpu.forward.graph";
inline constexpr const char * qwenpu_forward_program_prefix = "qwenpu.forward.program.";

struct llm_graph_params;

class llama_qwenpu_forward {
public:
    llama_qwenpu_forward(ggml_backend_t cuda, const char * path, uint32_t vocab, uint32_t language_width);
    ~llama_qwenpu_forward();

    ggml_backend_t backend() const;
    ggml_tensor * build(const llm_graph_params & params);
    ggml_tensor * chip_mean_square(ggml_context * ctx, ggml_backend_sched_t sched, ggml_tensor * kept) const;
    ggml_tensor * language_norm(ggml_context * ctx, ggml_tensor * text, ggml_tensor * chip_mean, float eps) const;

    bool accepts(llama_token token) const;
    void reset(const char * program);
    bool read_output(ggml_tensor * kept, bool chip_only, llama_pos pos);
    const llama_qwenpu_forward_info & info() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};
