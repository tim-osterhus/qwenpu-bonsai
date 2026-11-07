#define NOMINMAX
#include <windows.h>

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

template<class Function>
Function symbol(HMODULE library, const char * name) {
    auto address = GetProcAddress(library, name);
    if (!address) {
        throw std::runtime_error(std::string("missing runtime symbol: ") + name);
    }
    return reinterpret_cast<Function>(address);
}

HMODULE load(const fs::path & path) {
    auto library = LoadLibraryExW(path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!library) {
        throw std::runtime_error("LoadLibraryExW failed with Windows error " + std::to_string(GetLastError()));
    }
    return library;
}

void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void phase(const char * name) {
    std::cout << "{\"phase\":\"" << name << "\"}" << std::endl;
}

std::string trim(const std::string & text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    return start == std::string::npos ? "" :
        text.substr(start, text.find_last_not_of(" \t\r\n") - start + 1);
}

int parse_vector_call(const std::string & reply) {
    const auto text = trim(reply);
    if (text == "<tool_call>\n<function=run_vector_fixture_one>\n</function>\n</tool_call>") {
        return 1;
    }
    if (text == "<tool_call>\n<function=run_vector_fixture_two>\n</function>\n</tool_call>") {
        return 2;
    }
    return 0;
}

std::string decode_hex(const std::string & line) {
    require(!line.empty() && line.size() % 2 == 0 && line.size() <= 480, "invalid chat input length");
    auto nibble = [](char ch) {
        if (ch >= '0' && ch <= '9') {
            return ch - '0';
        }
        require(ch >= 'a' && ch <= 'f', "invalid chat input");
        return ch - 'a' + 10;
    };
    std::string text;
    for (size_t i = 0; i < line.size(); i += 2) {
        text.push_back(static_cast<char>(16 * nibble(line[i]) + nibble(line[i + 1])));
    }
    return text;
}

int wmain(int argc, wchar_t ** argv) {
    try {
        require(argc >= 5 && argc <= 8, "usage: probe runtime-dir model.gguf report-dir mode [mode inputs]");
        const fs::path runtime = fs::absolute(argv[1]);
        const fs::path model_path = argv[2];
        const fs::path report = argv[3];
        const std::wstring mode = argv[4];
        const bool chat = mode == L"checkpoint-chat";
        const bool checkpoint = chat || mode == L"checkpoint" || mode == L"checkpoint-story" ||
            mode == L"checkpoint-autonomous";
        const bool story = mode == L"story" || mode == L"checkpoint-story";
        const bool candidate = mode == L"candidate" || mode == L"checkpoint";
        const bool autonomous = chat || mode == L"autonomous" || mode == L"checkpoint-autonomous";
        const int sidecar_args = checkpoint ? 0 : 1;
        require((mode == L"baseline" && argc == 5) ||
            (candidate && argc == 5 + sidecar_args) ||
            (story && argc == 6 + 2 * sidecar_args) ||
            (autonomous && !chat && argc == 6 + sidecar_args) || (chat && argc == 5),
            "invalid probe mode or arguments");

        require(SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS), "DLL search setup failed");
        require(AddDllDirectory(runtime.c_str()) != nullptr, "DLL directory registration failed");
        auto ggml = load(runtime / L"ggml.dll");
        auto llama = load(runtime / L"llama.dll");
        symbol<decltype(&ggml_backend_load_all_from_path)>(ggml, "ggml_backend_load_all_from_path")(
            runtime.u8string().c_str());

#define LLAMA(name) const auto name = symbol<decltype(&::name)>(llama, #name)
        LLAMA(llama_backend_init);
        LLAMA(llama_supports_gpu_offload);
        LLAMA(llama_model_default_params);
        LLAMA(llama_model_load_from_file);
        LLAMA(llama_model_free);
        LLAMA(llama_context_default_params);
        LLAMA(llama_init_from_model);
        LLAMA(llama_free);
        LLAMA(llama_model_get_vocab);
        LLAMA(llama_vocab_n_tokens);
        LLAMA(llama_vocab_is_eog);
        LLAMA(llama_tokenize);
        LLAMA(llama_token_to_piece);
        LLAMA(llama_batch_get_one);
        LLAMA(llama_decode);
        LLAMA(llama_get_logits_ith);
        LLAMA(llama_get_memory);
        LLAMA(llama_memory_seq_pos_max);
        LLAMA(llama_state_seq_get_size);
        LLAMA(llama_state_seq_get_data);
#undef LLAMA

        llama_backend_init();
        require(llama_supports_gpu_offload(), "the runtime has no GPU offload backend");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        phase("model load");
        std::unique_ptr<llama_model, decltype(llama_model_free)> model(
            llama_model_load_from_file(model_path.u8string().c_str(), mp), llama_model_free);
        require(model != nullptr, "could not load packed Bonsai model");

        auto cp = llama_context_default_params();
        cp.n_ctx = chat ? 4096 : 1024;
        cp.n_seq_max = 1;
        cp.n_batch = chat ? 768 : autonomous ? 384 : 128;
        cp.n_ubatch = 128;
        cp.n_threads = 1;
        cp.n_threads_batch = 1;
        std::unique_ptr<llama_context, decltype(llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        require(ctx != nullptr, "could not allocate Bonsai context");
        const auto * vocab = llama_model_get_vocab(model.get());
        const auto n_vocab = llama_vocab_n_tokens(vocab);
        require(n_vocab > 0, "empty Bonsai vocabulary");

        auto decode_parts = [&](std::initializer_list<std::pair<std::string, bool>> parts) {
            std::array<llama_token, 768> tokens;
            int count = 0;
            for (const auto & [text, special] : parts) {
                const int n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                    tokens.data() + count, static_cast<int32_t>(tokens.size()) - count, false, special);
                require(n > 0, "probe tokenization failed");
                count += n;
            }
            require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), count)) == 0,
                "Bonsai text decode failed");
        };
        auto decode = [&](const std::string & text) {
            decode_parts({{text, true}});
        };
        auto save_logits = [&](const char * filename) {
            const auto * logits = llama_get_logits_ith(ctx.get(), -1);
            require(logits != nullptr, "Bonsai logits unavailable");
            std::ofstream file(report / filename, std::ios::binary);
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file.write(reinterpret_cast<const char *>(logits), n_vocab * sizeof(float));
        };
        auto position = [&]() {
            return llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0);
        };
        auto cache = [&]() {
            const auto size = llama_state_seq_get_size(ctx.get(), 0);
            require(size > 0, "Bonsai sequence state is empty");
            std::vector<uint8_t> data(size);
            require(llama_state_seq_get_data(ctx.get(), data.data(), size, 0) == size,
                "Bonsai sequence state serialization failed");
            return data;
        };

        decltype(&llama_qwenpu_reset) reset = nullptr;
        decltype(&llama_qwenpu_reset_program) reset_program = nullptr;
        decltype(&llama_qwenpu_step) step = nullptr;
        fs::path fixture_dir;
        llama_qwenpu_tokens metadata{};
        if (candidate || story || autonomous) {
            phase("circuit load");
            if (checkpoint) {
                reset_program = symbol<decltype(reset_program)>(llama, "llama_qwenpu_reset_program");
            } else {
                const fs::path fixture = argv[5];
                fixture_dir = fixture.parent_path();
                auto load_graph = symbol<decltype(&llama_qwenpu_load)>(llama, "llama_qwenpu_load");
                reset = symbol<decltype(reset)>(llama, "llama_qwenpu_reset");
                require(load_graph(ctx.get(), fixture.u8string().c_str()) == 0, "QwenPU graph load failed");
            }
            step = symbol<decltype(step)>(llama, "llama_qwenpu_step");
            auto get_tokens = symbol<decltype(&llama_qwenpu_get_tokens)>(llama, "llama_qwenpu_get_tokens");
            require(get_tokens(ctx.get(), &metadata) == 0, "QwenPU graph metadata unavailable");
        }

        auto run_circuit = [&](uint32_t program, const fs::path & snapshot, int steps, std::ofstream * trace_file) {
            const int status = checkpoint ? reset_program(ctx.get(), program) :
                reset(ctx.get(), snapshot.u8string().c_str());
            require(status == 0, "QwenPU program reset failed");
            auto token = static_cast<int32_t>(metadata.start);
            const auto halt = metadata.halt;
            const auto byte_base = metadata.byte_base;
            const auto byte_count = metadata.byte_count;
            std::vector<int32_t> trace;
            std::string output;
            trace.reserve(steps);
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < steps; ++i) {
                int32_t next;
                if (step(ctx.get(), token, &next) != 0) {
                    throw std::runtime_error("QwenPU graph step failed at trace position " +
                        std::to_string(i + 1));
                }
                trace.push_back(next);
                if (trace_file) {
                    trace_file->write(reinterpret_cast<const char *>(&next), sizeof(next));
                }
                const auto value = static_cast<uint32_t>(next);
                if (value >= byte_base && value - byte_base < byte_count) {
                    output.push_back(static_cast<char>(value - byte_base));
                }
                if (value == halt) {
                    break;
                }
                token = next;
            }
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            return std::make_tuple(std::move(trace), std::move(output), seconds);
        };

        auto answer_text = [&](int limit = 12, bool special = false) {
            std::string answer;
            for (int i = 0; i < limit; ++i) {
                const auto * logits = llama_get_logits_ith(ctx.get(), -1);
                require(logits != nullptr, "Bonsai continuation logits unavailable");
                auto next = static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
                if (llama_vocab_is_eog(vocab, next)) {
                    break;
                }
                char piece[256];
                const int n = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, special);
                require(n >= 0, "Bonsai answer piece did not fit");
                answer.append(piece, n);
                require(llama_decode(ctx.get(), llama_batch_get_one(&next, 1)) == 0,
                    "Bonsai answer decode failed");
            }
            return answer;
        };
        auto write_array = [](const auto & values) {
            std::cout << "[";
            bool first = true;
            for (const auto value : values) {
                if (!first) {
                    std::cout << ",";
                }
                std::cout << +value;
                first = false;
            }
            std::cout << "]";
        };

        const std::string assistant = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        if (autonomous) {
            auto clear_memory = symbol<decltype(&llama_memory_clear)>(llama, "llama_memory_clear");
            auto synchronize = symbol<decltype(&llama_synchronize)>(llama, "llama_synchronize");
            const std::string system = R"(<|im_start|>system
# Tools

You have access to the following functions:
<tools>
{"type":"function","function":{"name":"run_vector_fixture_one","description":"Execute the first built-in QwenPU vector fixture and return its output digit.","parameters":{"type":"object","properties":{}}}}
{"type":"function","function":{"name":"run_vector_fixture_two","description":"Execute the second built-in QwenPU vector fixture and return its output digit.","parameters":{"type":"object","properties":{}}}}
</tools>

To call a function, reply only in this format, replacing function_name with the actual name:
<tool_call>
<function=function_name>
</function>
</tool_call>

Call only when the user asks for an actual run of one of the fixtures. Otherwise answer normally. If asked to quote tool syntax, put it in a code fence. After a tool result, use it to answer the original request.<|im_end|>
)";
            std::ifstream prompts;
            if (!chat) {
                prompts.open(fs::path(argv[5 + sidecar_args]));
                require(prompts.good(), "could not open autonomous prompts");
            } else {
                phase("ready");
            }
            std::istream & input = chat ? std::cin : prompts;
            std::string prompt;
            int case_number = 0;
            while (std::getline(input, prompt)) {
                if (trim(prompt).empty()) {
                    continue;
                }
                require(++case_number <= (chat ? 8 : 32), "too many autonomous prompts");
                if (!chat) {
                    synchronize(ctx.get());
                    clear_memory(llama_get_memory(ctx.get()), true);
                } else {
                    prompt = decode_hex(prompt);
                }
                const auto turn_start = position();
                const std::string prefix = chat && case_number > 1 ? "<|im_end|>\n" : system;
                decode_parts({{prefix + "<|im_start|>user\n", true}, {prompt, false},
                              {"<|im_end|>\n" + assistant, true}});
                const auto decision = answer_text(64, true);
                const int program = parse_vector_call(decision);
                const auto before = position();
                auto after = before;
                size_t cache_bytes = 0;
                bool cache_equal = true;
                std::vector<int32_t> trace;
                std::string output;
                std::string answer = decision;
                double circuit_seconds = 0;
                if (program) {
                    const auto saved = cache();
                    cache_bytes = saved.size();
                    const auto snapshot = fixture_dir / ("program-" + std::to_string(program) + ".qps");
                    std::tie(trace, output, circuit_seconds) = run_circuit(program, snapshot, 128, nullptr);
                    require(static_cast<uint32_t>(trace.back()) == metadata.halt, "QwenPU trace exceeded its bound");
                    after = position();
                    cache_equal = saved == cache();
                    decode("<|im_end|>\n<|im_start|>user\n<tool_response>\n" + output +
                        "\n</tool_response><|im_end|>\n" + assistant);
                    answer = answer_text(32);
                }
                std::cout << "{\"phase\":\"case\",\"case\":" << case_number
                          << ",\"program\":" << program << ",\"turn_start\":" << turn_start
                          << ",\"position_before\":" << before
                          << ",\"position_after\":" << after << ",\"position_final\":" << position()
                          << ",\"cache_equal\":" << (cache_equal ? "true" : "false")
                          << ",\"cache_bytes\":" << cache_bytes
                          << ",\"circuit_seconds\":" << circuit_seconds << ",\"decision_bytes\":";
                write_array(decision);
                std::cout << ",\"trace\":";
                write_array(trace);
                std::cout << ",\"output_bytes\":";
                write_array(output);
                std::cout << ",\"answer_bytes\":";
                write_array(answer);
                std::cout << "}" << std::endl;
            }
            require(!input.bad() && (chat || case_number > 0), "could not read autonomous prompts");
            phase("result");
            return 0;
        }
        if (story) {
            const int steps = std::stoi(argv[5 + 2 * sidecar_args]);
            require(steps > 0 && steps <= 150000, "story trace limit must be between 1 and 150000");
            decode("<|im_start|>user\nRun the stories260K guest.<|im_end|>\n" + assistant + "CALL_VECTOR");
            const auto before = position();
            const auto saved = cache();
            std::ofstream trace_file(report / "trace.i32", std::ios::binary);
            trace_file.exceptions(std::ios::failbit | std::ios::badbit);
            phase("story circuit");
            const fs::path snapshot = checkpoint ? fs::path{} : fs::path(argv[6]);
            const auto [trace, output, seconds] = run_circuit(3, snapshot, steps, &trace_file);
            const auto after = position();
            const auto cache_equal = saved == cache();
            std::cout << "{\"phase\":\"result\",\"position_before\":" << before
                      << ",\"position_after\":" << after << ",\"cache_equal\":"
                      << (cache_equal ? "true" : "false") << ",\"cache_bytes\":" << saved.size()
                      << ",\"trace_tokens\":" << trace.size() << ",\"circuit_seconds\":" << seconds
                      << ",\"output_bytes\":";
            write_array(output);
            std::cout << "}" << std::endl;
            return 0;
        }
        decode("<|im_start|>system\nYou are testing a local vector circuit. Request it by emitting CALL_VECTOR. "
               "When its tool result arrives, follow the request about which digits to report.<|im_end|>\n"
               "<|im_start|>user\nRun the first vector program and report the result.<|im_end|>\n" +
               assistant + "CALL_VECTOR");
        for (int program = 1; program <= 2; ++program) {
            const auto label = std::to_string(program);
            if (program == 2) {
                decode("<|im_end|>\n<|im_start|>user\nRun the second vector program.<|im_end|>\n" +
                       assistant + "CALL_VECTOR");
            }
            save_logits(("idle-" + label + ".f32").c_str());
            const auto before = position();
            auto after = before;
            bool cache_equal = true;
            size_t cache_size = 0;
            std::vector<int32_t> trace;
            // Stock Bonsai receives the reference output at the same position.
            std::string output = program == 1 ? "8" : "6";
            double circuit_seconds = 0;

            if (candidate) {
                const auto snapshot = fixture_dir / ("program-" + label + ".qps");
                const auto saved = cache();
                cache_size = saved.size();
                std::tie(trace, output, circuit_seconds) = run_circuit(program, snapshot, 128, nullptr);
                require(static_cast<uint32_t>(trace.back()) == metadata.halt, "QwenPU trace exceeded its bound");
                after = position();
                cache_equal = saved == cache();
            }

            const std::string instruction = program == 1
                ? ". Reply with just the digit it returned."
                : ". Reply with two digits, the first program's result followed by this program's result. Digits only.";
            decode("<|im_end|>\n<|im_start|>user\nThe local tool returned: " + output +
                   instruction + "<|im_end|>\n" + assistant);
            save_logits(("continuation-" + label + ".f32").c_str());
            const auto answer = answer_text();

            std::cout << "{\"phase\":\"program\",\"program\":" << program
                      << ",\"position_before\":" << before
                      << ",\"position_after\":" << after << ",\"position_final\":" << position()
                      << ",\"cache_bytes\":" << cache_size << ",\"cache_equal\":"
                      << (cache_equal ? "true" : "false") << ",\"circuit_seconds\":" << circuit_seconds
                      << ",\"trace\":";
            write_array(trace);
            std::cout << ",\"output_bytes\":";
            write_array(output);
            std::cout << ",\"answer_bytes\":";
            write_array(answer);
            std::cout << "}" << std::endl;
        }
        phase("result");

        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}


