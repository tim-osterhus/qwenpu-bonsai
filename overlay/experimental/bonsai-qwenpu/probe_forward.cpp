#define NOMINMAX
#include <windows.h>

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

template<class Function>
Function symbol(HMODULE library, const char * name) {
    auto address = GetProcAddress(library, name);
    if (!address) throw std::runtime_error(std::string("Missing runtime symbol: ") + name);
    return reinterpret_cast<Function>(address);
}

HMODULE load(const fs::path & path) {
    auto library = LoadLibraryExW(path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!library) throw std::runtime_error("Windows could not load " + path.u8string());
    return library;
}

void require(bool result, const char * message) {
    if (!result) throw std::runtime_error(message);
}

int wmain(int argc, wchar_t ** argv) {
    try {
        require(argc >= 10, "Usage: probe_forward runtime-dir model.gguf report-dir forward|stock|story max-steps digit-or-snapshot token...");
        const auto runtime = fs::absolute(argv[1]);
        const fs::path model_path = argv[2];
        const fs::path report = argv[3];
        const std::wstring mode = argv[4];
        const bool story = mode == L"story";
        require(mode == L"forward" || mode == L"stock" || story, "Invalid run mode");
        const int max_steps = std::stoi(argv[5]);
        require(max_steps > 0 && max_steps <= (story ? 250000 : 4096), "Invalid circuit step limit");
        fs::create_directories(report);
        auto ggml = load(runtime / "ggml.dll");
        auto llama = load(runtime / "llama.dll");
        auto load_backends = symbol<decltype(&ggml_backend_load_all_from_path)>(
            ggml, "ggml_backend_load_all_from_path");
        load_backends(runtime.u8string().c_str());

#define LLAMA(name) auto name = symbol<decltype(&::name)>(llama, #name)
        LLAMA(llama_backend_init);
        LLAMA(llama_supports_gpu_offload);
        LLAMA(llama_model_default_params);
        LLAMA(llama_model_load_from_file);
        LLAMA(llama_model_free);
        LLAMA(llama_context_default_params);
        LLAMA(llama_init_from_model);
        LLAMA(llama_free);
        LLAMA(llama_batch_get_one);
        LLAMA(llama_decode);
        LLAMA(llama_get_logits_ith);
        LLAMA(llama_model_get_vocab);
        LLAMA(llama_vocab_n_tokens);
        LLAMA(llama_tokenize);
        LLAMA(llama_token_to_piece);
        LLAMA(llama_get_memory);
        LLAMA(llama_memory_seq_pos_max);
        LLAMA(llama_state_seq_get_size);
        LLAMA(llama_state_seq_get_data);
#undef LLAMA

        llama_backend_init();
        require(llama_supports_gpu_offload(), "CUDA offload is unavailable");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        auto model = std::unique_ptr<llama_model, decltype(llama_model_free)>(
            llama_model_load_from_file(model_path.u8string().c_str(), mp), llama_model_free);
        require(model != nullptr, "Could not load checkpoint");
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024;
        cp.n_seq_max = 1;
        cp.n_batch = 128;
        cp.n_ubatch = 128;
        cp.n_threads = cp.n_threads_batch = 1;
        auto ctx = std::unique_ptr<llama_context, decltype(llama_free)>(
            llama_init_from_model(model.get(), cp), llama_free);
        require(ctx != nullptr, "Could not create Bonsai context");

        const auto * vocab = llama_model_get_vocab(model.get());
        const int vocabulary = llama_vocab_n_tokens(vocab);
        const auto digit = std::wstring(argv[6]);
        require(story ? digit == L"one" || digit == L"three" : digit == L"3" || digit == L"1",
                "Unknown fixture input");
        const auto prompt = story ? std::string("Run<tool_call></tool_call>") :
            std::string("Compute<tool_call>") + char(digit[0]) + "</tool_call>";
        llama_token tokenized[32];
        const auto token_count = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
                                                tokenized, 32, false, true);
        require(token_count == (story ? 3 : 4) && argc >= 7 + token_count, "Unexpected fixture tokenization");
        for (int i = 0; i < token_count; ++i) {
            require(tokenized[i] == std::stoi(argv[7 + i]), "GGUF tokenizer disagrees with the fixture inputs");
        }
        auto position = [&]() { return llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0); };
        auto cache = [&]() {
            const auto size = llama_state_seq_get_size(ctx.get(), 0);
            std::vector<uint8_t> bytes(size);
            require(size && llama_state_seq_get_data(ctx.get(), bytes.data(), size, 0) == size,
                    "Could not serialize Bonsai text cache");
            return bytes;
        };
        auto logits = [&]() {
            auto values = llama_get_logits_ith(ctx.get(), -1);
            require(values != nullptr, "Bonsai text logits unavailable");
            return values;
        };
        auto write_logits = [&](const char * filename) {
            std::ofstream file(report / filename, std::ios::binary);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.write(reinterpret_cast<const char *>(logits()), vocabulary * sizeof(float));
        };
        auto decode = [&](llama_token token) {
            require(llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) == 0, "llama_decode failed");
        };

        decltype(&llama_qwenpu_forward_get) get = nullptr;
        if (mode != L"stock") {
            get = symbol<decltype(get)>(llama, "llama_qwenpu_forward_get");
            auto reset = symbol<decltype(&llama_qwenpu_forward_reset)>(llama, "llama_qwenpu_forward_reset");
            const char * snapshot = !story ? "qwenpu.forward.program.fixture" : digit == L"one" ?
                "qwenpu.forward.program.stories260k.one" : "qwenpu.forward.program.stories260k.three";
            require(reset(ctx.get(), snapshot) == 0, "Circuit snapshot load failed");
        }
        auto info = [&]() {
            llama_qwenpu_forward_info data{};
            require(get(ctx.get(), &data) == 0, "Integrated forward result unavailable");
            return data;
        };

        for (int i = 7; i < argc; ++i) {
            const auto token = std::stoi(argv[i]);
            decode(token);
            if (get) {
                const auto state = info();
                std::cout << "{\"phase\":\"prefix\",\"id\":" << token << ",\"prediction\":" << state.prediction
                          << ",\"position\":" << position() << "}" << std::endl;
            }
            if (i == 7) {
                write_logits("idle.f32");
                if (get) {
                    auto first = info();
                    require(first.prediction < 0, "Circuit took ownership of ordinary text");
                    std::cout << "{\"phase\":\"architecture\",\"text_vocab\":" << first.text_vocab
                              << ",\"chip_width\":" << first.chip_width
                              << ",\"widened_width\":" << first.widened_width
                              << ",\"chip_layers\":" << first.chip_layers << "}" << std::endl;
                }
            }
        }
        if (!get) {
            write_logits("resumed.f32");
            return 0;
        }

        auto current = info();
        std::vector<uint8_t> before;
        bool positions_held = true;
        bool cache_held = true;
        uint64_t private_steps = 0;
        unsigned emitted_text = 0;
        std::vector<unsigned> echoed_bytes;
        int total = 0;
        const auto start = std::chrono::steady_clock::now();
        for (; current.prediction >= 0 && total < max_steps; ++total) {
            const auto next = current.prediction;
            const bool chip = next >= vocabulary;
            if (chip && before.empty()) before = cache();
            if (!chip && !before.empty()) {
                cache_held = cache_held && before == cache();
                before.clear();
            }
            const auto pos = position();
            decode(next);
            auto after = info();
            positions_held = positions_held && position() == pos + (chip ? 0 : 1);
            if (chip) ++private_steps;
            else {
                ++emitted_text;
                char bytes[32];
                const auto n = llama_token_to_piece(vocab, next, bytes, sizeof(bytes), 0, true);
                require(n > 0, "Cannot decode circuit-selected text token");
                for (int i = 0; i < n; ++i) echoed_bytes.push_back(static_cast<unsigned char>(bytes[i]));
            }
            std::cout << "{\"phase\":\"step\",\"id\":" << next
                      << ",\"prediction\":" << after.prediction << ",\"position\":" << position()
                      << ",\"chip_only_steps\":" << after.chip_only_steps
                      << ",\"cuda_layer_calls\":" << after.cuda_layer_calls << "}\n";
            current = after;
        }
        if (!before.empty()) cache_held = cache_held && before == cache();
        if (current.prediction < 0) write_logits("resumed.f32");
        std::cout << "{\"phase\":\"result\",\"released\":" << (current.prediction < 0 ? "true" : "false")
                  << ",\"chip_only_steps\":" << private_steps << ",\"echoed_text_tokens\":" << emitted_text
                  << ",\"cuda_layer_calls\":" << current.cuda_layer_calls
                  << ",\"position_correct\":" << (positions_held ? "true" : "false")
                  << ",\"cache_equal_during_chip\":" << (cache_held ? "true" : "false")
                  << ",\"elapsed_seconds\":"
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << ",\"echoed_bytes\":[";
        for (size_t i = 0; i < echoed_bytes.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << echoed_bytes[i];
        }
        std::cout << "]}" << std::endl;
        return current.prediction < 0 || (story && total == max_steps) ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
