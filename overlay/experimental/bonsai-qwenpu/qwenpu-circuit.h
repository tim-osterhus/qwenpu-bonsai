#pragma once

#include "ggml-backend.h"
#include <array>
#include <cstdint>
#include <memory>

inline constexpr const char * qwenpu_version_key = "qwenpu.format_version";
inline constexpr std::array<const char *, 4> qwenpu_tensor_names{
    "qwenpu.graph", "qwenpu.program.1", "qwenpu.program.2", "qwenpu.program.stories260k",
};

struct llama_qwenpu_tokens;

class llama_qwenpu_circuit {
public:
    llama_qwenpu_circuit(ggml_backend_t cpu, const char * path, bool checkpoint = false);
    ~llama_qwenpu_circuit();

    void reset(const char * path);
    void reset_program(uint32_t program);
    llama_qwenpu_tokens tokens() const;
    int32_t step(uint32_t token);

private:
    struct impl;
    std::unique_ptr<impl> graph;
};

