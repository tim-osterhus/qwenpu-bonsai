// Load raw RV32/VX ELF data and transport the checkpoint's stock MTP feedback.
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;
using Floats = std::vector<float>;

constexpr uint32_t ram_end = 0x8000;
constexpr uint32_t drom_base = 0x20000;
constexpr uint32_t rom_base = 0x40000;
constexpr uint32_t rom_end = 0x80000;
constexpr uint32_t wrom_base = 0x100000, wrom_end = wrom_base + 72 * 4608;
constexpr uint32_t packed_wrom_base = 0x60000000;
constexpr uint32_t dot_wrom_base = 0x70000000;
constexpr int dot_payload = 2048;
constexpr size_t max_elf_bytes = 8 * 1024 * 1024;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static uint32_t little(const uint8_t * bytes, int count) {
    uint32_t result = 0;
    for (int i = 0; i < count; ++i) result |= uint32_t(bytes[i]) << (i * 8);
    return result;
}

struct Segment {
    uint32_t address;
    uint32_t flags;
    Bytes bytes;
};

struct Guest {
    uint32_t entry;
    std::vector<Segment> segments;
};

static Guest load_guest(const fs::path & path) {
    const auto size = fs::file_size(path);
    require(size >= 52 && size <= max_elf_bytes, "ELF file size outside the probe limit");
    Bytes data(static_cast<size_t>(size));
    std::ifstream input(path, std::ios::binary);
    require(bool(input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()))),
            "Could not read ELF file");
    const auto field = [&](size_t offset, int bytes = 4) { return little(data.data() + offset, bytes); };
    require(field(0) == 0x464c457f && field(4, 2) == 0x0101 && data[6] == 1 &&
            field(16, 2) == 2 && field(18, 2) == 243 && field(20) == 1 && field(40, 2) == 52 &&
            (field(36) & 0x9) == 0, "Expected a static little-endian RV32 ELF without RVC or RV32E");
    const auto ph_offset = field(28), ph_size = field(42, 2), ph_count = field(44, 2);
    require(ph_offset >= 52 && ph_size == 32 && ph_count > 0 && ph_count <= 128 &&
            uint64_t(ph_offset) + uint64_t(ph_count) * ph_size <= size, "Invalid ELF program-header table");
    Guest guest{field(24), {}};
    require((guest.entry & 3) == 0, "Unaligned ELF entry");
    bool has_entry = false;
    for (uint32_t i = 0; i < ph_count; ++i) {
        const size_t header = ph_offset + i * ph_size;
        const auto type = field(header);
        require(type != 2 && type != 3, "ELF uses dynamic linking");
        if (type != 1) continue;
        const auto offset = field(header + 4), address = field(header + 8);
        const auto filesz = field(header + 16), memsz = field(header + 20), flags = field(header + 24);
        require(filesz <= memsz && uint64_t(offset) + filesz <= size, "ELF PT_LOAD exceeds the file");
        if (memsz == 0) continue;
        const uint64_t end = uint64_t(address) + memsz;
        const bool ram = end <= ram_end && (flags & 3) == 2;
        const bool drom = address >= drom_base && end <= rom_base && (flags & 3) == 0;
        const bool rom = address >= rom_base && end <= rom_end && (flags & 3) == 1;
        const bool wrom = address >= wrom_base && end <= wrom_end && (flags & 3) == 0;
        require((flags & 4) && (ram || drom || rom || wrom), "ELF PT_LOAD outside the QwenPU memory map");
        require(std::none_of(guest.segments.begin(), guest.segments.end(), [&](const Segment & prior) {
            return address < uint64_t(prior.address) + prior.bytes.size() && prior.address < end;
        }), "Overlapping ELF PT_LOAD segments");
        Bytes bytes(memsz, 0);
        std::copy_n(data.data() + offset, filesz, bytes.data());
        guest.segments.push_back({address, flags, std::move(bytes)});
        if (rom && guest.entry >= address && uint64_t(guest.entry) + 4 <= uint64_t(address) + filesz) has_entry = true;
    }
    require(has_entry, "ELF entry is outside executable file bytes");
    return guest;
}

template<class T>
static T symbol(HMODULE module, const char * name) {
    const auto address = GetProcAddress(module, name);
    if (!address) throw std::runtime_error(std::string("Missing stock export: ") + name);
    return reinterpret_cast<T>(address);
}

static HMODULE load_dll(const fs::path & path) {
    const auto module = LoadLibraryExW(path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    require(module != nullptr, "Could not load stock Prism DLL");
    return module;
}

static std::string sha256(const void * data, size_t size) {
    std::array<uint8_t, 32> digest{};
    require(size <= ULONG_MAX && BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
        static_cast<PUCHAR>(const_cast<void *>(data)), static_cast<ULONG>(size),
        digest.data(), static_cast<ULONG>(digest.size())) == 0, "SHA-256 failed");
    std::ostringstream out;
    for (auto byte : digest) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
    return out.str();
}

static void put_bits(float * lane, uint32_t value, int count) {
    for (int i = 0; i < count; ++i) lane[i] = ((value >> i) & 1) ? 16.f : -16.f;
}

static uint32_t get_bits(const float * lane, int count, float & minimum) {
    uint32_t value = 0;
    for (int i = 0; i < count; ++i) {
        require(std::isfinite(lane[i]) && lane[i] != 0.f, "Ambiguous stock feedback bit");
        minimum = std::min(minimum, std::abs(lane[i]));
        if (lane[i] > 0.f) value |= uint32_t(1) << i;
    }
    return value;
}

static void write_file(const fs::path & path, const Bytes & bytes) {
    std::ofstream out(path, std::ios::binary);
    require(bool(out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()))),
            "Could not save stock cache snapshot");
}

enum Action : llama_token { seed_miss, seed_word, start, fetch, halt, retire, store, write_reg, fault, commit };
constexpr int write_address = 64, value_lane = 128, fresh_handle = 160, entry_lane = 192;
constexpr int read_address = 256, old_handle = 320, old_present = 344, pc_lane = 352, word_lane = 384;
constexpr int read_mark = 289, readback = 928;
constexpr int cause_lane = 416, handle_bits = 24;
constexpr int next_pc_lane = 424, rd_index = 456, rd_value = 464, rd_valid = 496;
constexpr int bus_valid = 497, bus_write = 498, branch_valid = 499, branch_taken = 500;
constexpr int bus_width = 501, bus_address = 512, bus_word = 544, exit_code = 576, putchar_valid = 615;
constexpr uint32_t seed_batch_size = 64;

static uint32_t setting(const char * name, uint32_t fallback, uint32_t maximum) {
    std::array<char, 64> text{};
    const auto count = GetEnvironmentVariableA(name, text.data(), static_cast<DWORD>(text.size()));
    if (count == 0) return fallback;
    require(count < text.size(), "VX probe limit is too long");
    size_t used = 0;
    const auto value = std::stoul(text.data(), &used);
    require(text[used] == 0 && (value > 0 || fallback == 0) && value <= maximum, "Invalid VX probe limit");
    return static_cast<uint32_t>(value);
}

struct BenchmarkWindow {
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;
    using Duration = Clock::duration;

    bool active = false;
    Time start{};
    llama_pos start_position = 0;
    uint32_t start_attempts = 0;
    uint32_t decode_calls = 0, output_calls = 0;
    Duration decode_call{}, logits_get{}, logits_scan{}, feedback_get{}, feedback_check_and_copy{};
    Duration witness_callback{}, kv_host{}, trace{};

    void open(Time now, uint32_t attempts, llama_pos pos) {
        start = now;
        start_attempts = attempts;
        start_position = pos;
        active = true;
    }

    Time stamp() const { return active ? Clock::now() : Time{}; }

    void split(Time & since, Duration & elapsed) const {
        if (!active) return;
        const auto now = Clock::now();
        elapsed += now - since;
        since = now;
    }

    void report(int run_index, uint32_t requested_start, uint32_t end_attempts, llama_pos end_position,
                llama_pos seed_positions, Time run_begin, Time seed_ready, Time end, const char * reason,
                int vocab, bool top2) const {
        const auto nanoseconds = [](Duration time) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(time).count();
        };
        const auto seconds = [](Duration time) { return std::chrono::duration<double>(time).count(); };
        std::ostringstream line;
        line << std::setprecision(std::numeric_limits<double>::max_digits10)
             << "stock VX benchmark {\"run_index\":" << run_index
             << ",\"requested_start_after\":" << requested_start
             << ",\"seed_positions\":" << seed_positions
             << ",\"seed_ready_elapsed_ns\":" << nanoseconds(seed_ready - run_begin)
             << ",\"end_attempts\":" << end_attempts << ",\"end_position\":" << end_position
             << ",\"end_elapsed_ns\":" << nanoseconds(end - run_begin) << ",\"stop_reason\":\"" << reason << '"';
        if (active) {
            const auto elapsed = end - start;
            const auto accounted = decode_call + logits_get + logits_scan + feedback_get +
                feedback_check_and_copy + kv_host + trace;
            line << ",\"start_attempts\":" << start_attempts << ",\"start_position\":" << start_position
                 << ",\"start_elapsed_ns\":" << nanoseconds(start - run_begin)
                 << ",\"wall_seconds\":" << seconds(elapsed)
                 << ",\"decode_calls\":" << decode_calls << ",\"output_calls\":" << output_calls
                 // Logical score and index payload, excluding feedback. This is not a bus counter.
                 << ",\"logits_f32_payload_bytes\":" << uint64_t(output_calls) * (top2 ? 2 : vocab) * sizeof(float)
                 << ",\"candidate_i32_payload_bytes\":" << uint64_t(output_calls) * (top2 ? 2 : 0) * sizeof(llama_token)
                 << ",\"seconds\":{\"decode_call_excluding_witness_callback\":" << seconds(decode_call - witness_callback)
                 << ",\"logits_get\":" << seconds(logits_get) << ",\"logits_scan\":" << seconds(logits_scan)
                 << ",\"feedback_get\":" << seconds(feedback_get)
                 << ",\"feedback_check_and_copy\":" << seconds(feedback_check_and_copy)
                 << ",\"kv_witness_callback\":" << seconds(witness_callback) << ",\"kv_host\":" << seconds(kv_host)
                 << ",\"trace_and_observation\":" << seconds(trace)
                 << ",\"other_transport_and_overhead\":" << seconds(elapsed - accounted) << '}';
        } else {
            line << ",\"start_attempts\":null,\"start_position\":null,\"start_elapsed_ns\":null,\"wall_seconds\":null"
                 << ",\"decode_calls\":null,\"output_calls\":null,\"logits_f32_payload_bytes\":null,\"seconds\":null";
        }
        std::cerr << line.str() << "}\n";
    }
};

struct KVWitness {
    decltype(&ggml_backend_tensor_get) get;
    std::string key_name, value_name;
    std::vector<int64_t> slots;
    Floats keys, values;
    bool enabled = false;
    uint32_t heads = 0;
    BenchmarkWindow * benchmark = nullptr;
    decltype(&ggml_backend_buffer_is_host) is_host = nullptr;
    bool audit_top2 = false, saw_top_k = false, saw_gather = false;

    void capture(ggml_tensor * tensor, Floats & destination) {
        require(tensor && tensor->type == GGML_TYPE_F32 && tensor->nb[0] == sizeof(float) &&
                tensor->nb[1] == size_t(tensor->ne[0]) * sizeof(float) &&
                ((tensor->ne[0] == 256 && tensor->ne[1] == heads) || tensor->ne[0] == 256 * heads),
                "Unexpected stock KV witness layout");
        const auto count = static_cast<size_t>(tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3]);
        destination.resize(count);
        get(tensor, destination.data(), 0, count * sizeof(float));
    }

    static bool observe(ggml_tensor * tensor, bool ask, void * user) {
        auto & tap = *static_cast<KVWitness *>(user);
        if (tap.audit_top2 && tensor->op == GGML_OP_TOP_K && tensor->ne[0] == 2 && !tap.saw_top_k) {
            if (!ask) {
                require(tensor->buffer && !tap.is_host(tensor->buffer), "Top-2 ran on a host backend");
                tap.saw_top_k = true;
            }
            return true;
        }
        if (tap.audit_top2 && tensor->op == GGML_OP_GET_ROWS && tensor->src[1] &&
            tensor->src[1]->op == GGML_OP_TOP_K && tensor->src[1]->ne[0] == 2 && !tap.saw_gather) {
            if (!ask) {
                require(tensor->buffer && !tap.is_host(tensor->buffer), "Top-2 gather ran on a host backend");
                tap.saw_gather = true;
            }
            return true;
        }
        if (!tap.enabled) return false;
        if (tap.value_name == tensor->name) {
            if (!ask) {
                auto * timer = tap.benchmark;
                auto phase = timer ? timer->stamp() : BenchmarkWindow::Time{};
                tap.capture(tensor, tap.values);
                if (timer) timer->split(phase, timer->witness_callback);
            }
            return true;
        }
        if (tensor->op == GGML_OP_SET_ROWS && tensor->src[2] && tap.key_name == tensor->src[2]->name) {
            if (!ask) {
                auto * timer = tap.benchmark;
                auto phase = timer ? timer->stamp() : BenchmarkWindow::Time{};
                const auto * indices = tensor->src[1];
                require(indices && indices->type == GGML_TYPE_I64 && indices->ne[0] > 0, "Missing stock cache slots");
                tap.slots.resize(static_cast<size_t>(indices->ne[0]));
                tap.get(indices, tap.slots.data(), 0, tap.slots.size() * sizeof(int64_t));
                tap.capture(tensor->src[0], tap.keys);
                if (timer) timer->split(phase, timer->witness_callback);
            }
            return true;
        }
        return false;
    }

    void clear() {
        slots.clear();
        keys.clear();
        values.clear();
    }

    void write(std::ostream & output, uint32_t first_position, uint32_t count) {
        const size_t stride = 256 * heads;
        require(slots.size() == count && keys.size() == count * stride && values.size() == count * stride,
                "Missing retained-row KV witness");
        for (uint32_t row = 0; row < count; ++row) {
            const auto * key = keys.data() + row * stride;
            const auto * value = values.data() + row * stride;
            const bool sentinel = key[196] > 0.f;
            float minimum = std::numeric_limits<float>::infinity();
            uint32_t address = 0, reg = 0;
            for (size_t lane = 0; lane < stride; ++lane) {
                require(std::isfinite(key[lane]) && std::isfinite(value[lane]), "Non-finite retained KV element");
                const bool active_key = sentinel ? lane == 196 : lane >= 64 && lane < 196;
                if (lane < 256) require(active_key ? key[lane] != 0.f : key[lane] == 0.f,
                        "Retained scalar key has wrong active lanes");
                if (lane < 57) require(value[lane] != 0.f, "Retained scalar value has wrong active lanes");
                if (active_key) minimum = std::min(minimum, std::abs(key[lane]));
                if (lane < 57) minimum = std::min(minimum, std::abs(value[lane]));
            }
            if (!sentinel) {
                for (int bit = 0; bit < 33; ++bit) {
                    const auto * repeated = key + 64 + 4 * bit;
                    for (int copy = 1; copy < 4; ++copy) require(repeated[copy] == repeated[0], "Key repetitions differ");
                    if (repeated[0] > 0.f) {
                        if (bit == 32) reg = 1;
                        else address |= uint32_t(1) << bit;
                    }
                }
            }
            const uint32_t word = get_bits(value, 32, minimum);
            const uint32_t handle = get_bits(value + 32, 24, minimum);
            const uint32_t hit = get_bits(value + 56, 1, minimum);
            require(minimum >= 1e-5f && slots[row] >= 0 && slots[row] <= UINT32_MAX,
                    "Retained KV value could underflow F16");
            require(sentinel ? !hit : hit && handle == first_position + row, "Retained row has wrong model handle");
            const std::array<uint32_t, 6> record{first_position + row, static_cast<uint32_t>(slots[row]), address,
                word, handle, reg | (uint32_t(sentinel) << 1) | (hit << 2)};
            output.write(reinterpret_cast<const char *>(record.data()), sizeof(record));
            require(bool(output.write(reinterpret_cast<const char *>(&minimum), sizeof(minimum))),
                    "Could not save retained KV witness");
        }
    }
};

struct VXObserver {
    std::array<std::vector<uint32_t>, 4> trackers;
    std::array<std::vector<uint32_t>, 2> extra_words;
    std::vector<uint32_t> wrom_index;
    std::vector<llama_token> read_tokens;
    uint32_t valid = 0;
    llama_token tile_token = -1;
    bool enabled = false;

    void load(const fs::path & path, uint32_t width) {
        std::ifstream input(path);
        std::string version;
        input >> version;
        require(version == "QPVXOBSERVER2", "Unknown VX observer format");
        const auto lanes = [&](std::vector<uint32_t> & target, uint32_t expected = 0) {
            uint32_t count = 0;
            input >> count;
            require(input && count > 0 && count <= 32 && (!expected || count == expected), "Invalid observer word width");
            target.resize(count);
            for (auto & lane : target) {
                input >> lane;
                require(input && lane < width, "Observer lane exceeds checkpoint width");
            }
        };
        for (auto & word : trackers) lanes(word, 32);
        input >> valid >> tile_token;
        require(input && valid < width && tile_token >= 2, "Invalid VX observer flags");
        lanes(wrom_index);
        for (auto & word : extra_words) lanes(word, 32);
        uint32_t read_count = 0;
        input >> read_count;
        require(input && read_count > 0 && read_count <= 1024, "Invalid observer read phase count");
        read_tokens.resize(read_count);
        for (auto & token : read_tokens) {
            input >> token;
            require(input && token >= 2, "Invalid observer read phase");
        }
        input >> std::ws;
        require(input.eof(), "Unexpected observer fields");
        enabled = true;
    }

    static uint32_t read(const float * feedback, const std::vector<uint32_t> & lanes, float & minimum) {
        uint32_t value = 0;
        for (size_t bit = 0; bit < lanes.size(); ++bit) {
            const auto digit = get_bits(feedback + lanes[bit], 1, minimum);
            value |= digit << bit;
        }
        return value;
    }

    void print(std::ostream & out, const float * feedback, float & minimum) const {
        if (!enabled) return;
        constexpr std::array<const char *, 4> names{"vmx", "vsum", "varg", "vbest"};
        out << ",\"vx_trackers\":{";
        for (size_t index = 0; index < trackers.size(); ++index) {
            if (index) out << ',';
            out << '"' << names[index] << "\":";
            if (index == 3 && !get_bits(feedback + valid, 1, minimum)) out << "null";
            else out << static_cast<int32_t>(read(feedback, trackers[index], minimum));
        }
        out << '}';
    }

    std::string summary(const float * feedback, float & minimum) const {
        constexpr std::array<const char *, 4> names{"vmx", "vsum", "varg", "vbest"};
        std::ostringstream out;
        for (size_t i = 0; i < trackers.size(); ++i) {
            if (i) out << ',';
            out << names[i] << '=';
            if (i == 3 && !get_bits(feedback + valid, 1, minimum)) out << "null";
            else out << static_cast<int32_t>(read(feedback, trackers[i], minimum));
        }
        return out.str();
    }
};

static void record(std::ostream & out, Action action, const float * feedback, float & minimum, const VXObserver & observer) {
    const auto read = [&](int lane, int count) { return get_bits(feedback + lane, count, minimum); };
    const auto pc = read(pc_lane, 32);
    out << "{\"pc\":" << pc << ",\"word\":" << read(word_lane, 32);
    out << ",\"next_pc\":" << read(next_pc_lane, 32);
    out << ",\"rd\":";
    if (read(rd_valid, 1)) {
        out << "{\"index\":" << read(rd_index, 5) << ",\"value\":" << read(rd_value, 32) << '}';
    } else out << "null";
    out << ",\"bus\":";
    if (read(bus_valid, 1)) {
        const auto code = feedback[bus_width] == 0.f && feedback[bus_width + 1] == 0.f ? 0u : read(bus_width, 2);
        require(code < 3, "Invalid checkpoint bus width");
        constexpr std::array<uint32_t, 3> widths{4, 1, 2};
        out << "{\"kind\":\"" << (read(bus_write, 1) ? "write" : "read") << "\",\"address\":"
            << read(bus_address, 32) << ",\"width\":" << widths[code] << ",\"value\":" << read(bus_word, 32) << '}';
    } else out << "null";
    out << ",\"branch_taken\":";
    if (read(branch_valid, 1)) out << (read(branch_taken, 1) ? "true" : "false");
    else out << "null";
    out << ",\"trap\":";
    if (action == fault) out << "{\"cause\":" << read(cause_lane, 3) << ",\"pc\":" << pc << '}';
    else out << "null";
    out << ",\"exit_code\":";
    if (action == halt) out << read(exit_code, 32);
    else out << "null";
    if (feedback[putchar_valid] > 0.f) out << ",\"output_symbol\":" << read(exit_code, 9);
    observer.print(out, feedback, minimum);
    out << '}';
}

int wmain(int argc, wchar_t ** argv) {
    try {
        if (argc == 3 && std::wstring(argv[1]) == L"--inspect-elf") {
            const auto guest = load_guest(argv[2]);
            std::cout << "{\"entry\":" << guest.entry << ",\"segments\":[";
            for (size_t i = 0; i < guest.segments.size(); ++i) {
                const auto & segment = guest.segments[i];
                if (i) std::cout << ',';
                std::cout << "{\"address\":" << segment.address << ",\"flags\":" << segment.flags
                          << ",\"bytes\":" << segment.bytes.size() << ",\"sha256\":\""
                          << sha256(segment.bytes.data(), segment.bytes.size()) << "\"}";
            }
            std::cout << "]}\n";
            return 0;
        }
        require(argc >= 7,
            "Usage: probe_stock_mtp_vx RUNTIME MODEL cpu|cuda tiny|packed|reference CAPACITY [PREFIX ELF ...]");
        const std::wstring device = argv[3], mode = argv[4];
        const bool cuda = device == L"cuda", packed = mode == L"packed", reference = mode == L"reference";
        require((cuda || device == L"cpu") && (packed || reference || mode == L"tiny"), "Invalid device or mode");
        require((reference && argc == 7) || (!reference && argc >= 8), "Unexpected argument count");
        size_t capacity_end = 0;
        const auto capacity = std::stoul(argv[5], &capacity_end);
        require(capacity_end == std::wcslen(argv[5]) && capacity >= 256 && capacity <= 131072 &&
                capacity % 256 == 0, "Invalid MTP cache capacity");
        const auto decode_limit = static_cast<llama_pos>(setting("QWENPU_MAX_DECODE_POSITIONS", 24000, 16000000));
        const auto seed_batch = setting("QWENPU_MTP_SEED_BATCH", seed_batch_size, seed_batch_size);
        const auto dot_wrom_rows = setting("QWENPU_DOT64_WROM_ROWS", 16, 4608);
        const bool flash_mtp = setting("QWENPU_MTP_FLASH_ATTENTION", 0, 1) != 0;
        const bool full_write_snapshots = setting("QWENPU_FULL_WRITE_SNAPSHOTS", 1, 2) == 1;
        const auto stop_after = setting("QWENPU_STOP_AFTER_ATTEMPTS", 0, 10000000);
        const bool trimmed_head = setting("QWENPU_TRIMMED_ACTION_HEAD", 0, 1) != 0;
        require(!trimmed_head || packed, "Trimmed head requires the packed guest mode");
        const bool top2 = setting("QWENPU_TOP2", 0, 1) != 0;
        const bool sample_audit = setting("QWENPU_TOP2_AUDIT", 0, 1) != 0;
        const bool kv_callback_off = setting("QWENPU_KV_CALLBACK_OFF", 0, 1) != 0;
        require(!kv_callback_off || !sample_audit, "CUDA score audit needs the evaluation callback");
        require(!(top2 || sample_audit) || (cuda && trimmed_head), "Top-2 study requires the CUDA trimmed head");
        const auto benchmark_start_after = setting("QWENPU_BENCH_START_AFTER", 0, 10000000);
        const bool time_actions = setting("QWENPU_BENCH_ACTION_TIMES", 0, 1) != 0;
        std::array<wchar_t, 32768> diagnostic_path{};
        const auto diagnostic_chars = GetEnvironmentVariableW(L"QWENPU_DIAGNOSTIC_READS", diagnostic_path.data(),
            static_cast<DWORD>(diagnostic_path.size()));
        require(diagnostic_chars < diagnostic_path.size(), "Diagnostic read path is too long");
        const auto runtime = fs::absolute(argv[1]);
        const auto ggml = load_dll(runtime / "ggml.dll");
#ifdef QWENPU_PRISM_B12128
        if (cuda) {
            // Keep the official runtime's private CUDA dependencies resolvable
            // before its plugin registry discovers ggml-cuda.dll.
            for (const auto * name : {"cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"}) {
                load_dll(runtime / name);
            }
        }
#endif
        const auto ggml_base = load_dll(runtime / "ggml-base.dll");
        const auto llama = load_dll(runtime / "llama.dll");
        symbol<decltype(&ggml_backend_load_all_from_path)>(ggml, "ggml_backend_load_all_from_path")(runtime.u8string().c_str());
#define LLAMA(name) auto name = symbol<decltype(&::name)>(llama, #name)
        LLAMA(llama_backend_init);
        LLAMA(llama_supports_gpu_offload);
        LLAMA(llama_model_default_params);
        LLAMA(llama_model_load_from_file);
        LLAMA(llama_model_free);
        LLAMA(llama_model_n_embd);
        LLAMA(llama_model_n_layer);
        LLAMA(llama_model_get_vocab);
        LLAMA(llama_vocab_n_tokens);
        LLAMA(llama_tokenize);
        LLAMA(llama_context_default_params);
        LLAMA(llama_init_from_model);
        LLAMA(llama_free);
        LLAMA(llama_n_ctx);
        LLAMA(llama_decode);
        LLAMA(llama_get_logits);
        LLAMA(llama_get_logits_ith);
        LLAMA(llama_sampler_chain_init);
        LLAMA(llama_sampler_chain_default_params);
        LLAMA(llama_sampler_chain_add);
        LLAMA(llama_sampler_init_top_k);
        LLAMA(llama_sampler_free);
        LLAMA(llama_set_sampler);
        LLAMA(llama_get_sampled_logits_ith);
        LLAMA(llama_get_sampled_logits_count_ith);
        LLAMA(llama_get_sampled_candidates_ith);
        LLAMA(llama_get_sampled_candidates_count_ith);
        LLAMA(llama_get_memory);
        LLAMA(llama_memory_clear);
        LLAMA(llama_memory_seq_rm);
        LLAMA(llama_memory_seq_pos_max);
        LLAMA(llama_state_seq_get_size_ext);
        LLAMA(llama_state_seq_get_data_ext);
#undef LLAMA
        const auto set_nextn = symbol<void (*)(llama_context *, bool, bool)>(llama,
            "?llama_set_embeddings_nextn@@YAXPEAUllama_context@@_N1@Z");
        const auto get_nextn = symbol<float * (*)(llama_context *, int32_t)>(llama,
            "?llama_get_embeddings_nextn_ith@@YAPEAMPEAUllama_context@@H@Z");
        llama_backend_init();
        require(!cuda || llama_supports_gpu_offload(), "Stock CUDA offload unavailable");
#ifdef QWENPU_PRISM_B12128
        if (cuda) {
            const auto by_type = symbol<decltype(&ggml_backend_dev_by_type)>(ggml, "ggml_backend_dev_by_type");
            require(by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr, "No stock CUDA GPU device was registered");
        }
#endif
        auto mp = llama_model_default_params();
#ifdef QWENPU_PRISM_B12128
        mp.load_mtp = !reference;
#endif
        mp.n_gpu_layers = cuda ? 999 : 0;
        const auto model = std::unique_ptr<llama_model, decltype(llama_model_free)>(
            llama_model_load_from_file(fs::path(argv[2]).u8string().c_str(), mp), llama_model_free);
        require(bool(model), "Stock model load failed");
        const auto width = llama_model_n_embd(model.get());
        const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const bool with_text = packed || reference;
        const auto layers = llama_model_n_layer(model.get());
        const bool model_9b = width == 4096 && layers == 32;
        const bool model_27b = width == 5120 && layers == 64;
        require(with_text && (model_9b || model_27b) && vocab == 248320,
                "Unexpected stock Bonsai geometry");
        VXObserver observer;
        std::array<wchar_t, 32768> observer_path{};
        const auto observer_chars = GetEnvironmentVariableW(L"QWENPU_VX_OBSERVER", observer_path.data(),
            static_cast<DWORD>(observer_path.size()));
        require(observer_chars < observer_path.size(), "VX observer path is too long");
        if (observer_chars) observer.load(observer_path.data(), static_cast<uint32_t>(width));
        KVWitness witness{symbol<decltype(&ggml_backend_tensor_get)>(ggml_base, "ggml_backend_tensor_get"),
            "cache_k_l" + std::to_string(layers),
            "mtp_Vcur-" + std::to_string(layers), {}, {}, {}, false, 4u};
        witness.audit_top2 = top2 && sample_audit;
        if (witness.audit_top2) {
            witness.is_host = symbol<decltype(&ggml_backend_buffer_is_host)>(ggml_base, "ggml_backend_buffer_is_host");
        }
        auto make_context = [&](llama_context_type type, uint32_t cells) {
            auto params = llama_context_default_params();
            params.ctx_type = type;
            params.n_ctx = cells;
            params.n_batch = params.n_ubatch = type == LLAMA_CONTEXT_TYPE_MTP ? seed_batch : seed_batch_size;
            params.n_threads = params.n_threads_batch = 2;
            params.flash_attn_type = type == LLAMA_CONTEXT_TYPE_MTP && flash_mtp ?
                LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            params.type_k = params.type_v = GGML_TYPE_F16;
            if (type == LLAMA_CONTEXT_TYPE_MTP) {
                if (top2) params.n_outputs_max_per_seq = 1;
                if (!kv_callback_off) {
                    params.cb_eval = KVWitness::observe;
                    params.cb_eval_user_data = &witness;
                }
            }
            auto result = std::unique_ptr<llama_context, decltype(llama_free)>(
                llama_init_from_model(model.get(), params), llama_free);
            require(bool(result), "Stock context allocation failed");
            return result;
        };
        auto state = [&](llama_context * ctx, llama_state_seq_flags flags = 0) {
            Bytes bytes(llama_state_seq_get_size_ext(ctx, 0, flags));
            require(!bytes.empty() && llama_state_seq_get_data_ext(ctx, bytes.data(), bytes.size(), 0, flags) == bytes.size(),
                    "Stock sequence state unavailable");
            return bytes;
        };
        auto position = [&](llama_context * ctx) { return llama_memory_seq_pos_max(llama_get_memory(ctx), 0); };
        const auto check_scores = [&](const float * logits) {
            require(logits && std::all_of(logits, logits + vocab, [](float v) { return std::isfinite(v); }),
                    "Missing or non-finite stock logits");
        };
        auto scores = [&](llama_context * ctx) {
            const auto * logits = llama_get_logits_ith(ctx, -1);
            check_scores(logits);
            return logits;
        };
        auto text_state = [&](llama_context * ctx) {
            const auto full = state(ctx), recurrent = state(ctx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            require(full.size() > recurrent.size() && recurrent.size() >= 8 &&
                    std::equal(recurrent.begin() + 8, recurrent.end(), full.end() - recurrent.size() + 8),
                    "Unexpected stock Bonsai serialization layout");
            std::ostringstream out;
            out << "{\"position\":" << position(ctx) << ",\"state_sha256\":\"" << sha256(full.data(), full.size())
                << "\",\"kv_sha256\":\"" << sha256(full.data() + 8, full.size() - recurrent.size())
                << "\",\"recurrent_sha256\":\"" << sha256(recurrent.data(), recurrent.size())
                << "\",\"logits_sha256\":\"" << sha256(scores(ctx), size_t(vocab) * sizeof(float)) << "\"}";
            return out.str();
        };
        std::unique_ptr<llama_context, decltype(llama_free)> text(nullptr, llama_free);
        std::string parked = "null";
        auto park_text = [&]() {
            llama_memory_clear(llama_get_memory(text.get()), true);
            std::array<llama_token, 6> tokens{33, 32, 34, 198, 35, 36};
            const llama_batch batch{static_cast<int32_t>(tokens.size()), tokens.data(), nullptr, nullptr, nullptr, nullptr, nullptr};
            require(llama_decode(text.get(), batch) == 0, "Stock text prefill failed");
            return text_state(text.get());
        };
        if (with_text) {
            text = make_context(LLAMA_CONTEXT_TYPE_DEFAULT, 256);
            parked = park_text();
        }
        std::vector<std::string> visible_results;
        auto print_handoffs = [&]() {
            std::cout << '[';
            for (size_t index = 0; index < visible_results.size(); ++index) {
                require(park_text() == parked, "Text reset changed the parked state");
                const auto & visible = visible_results[index];
                const std::string message = " QwenPU result: " + visible;
                const auto * vocabulary = llama_model_get_vocab(model.get());
                std::vector<llama_token> tokens(message.size() + 16);
                const auto count = llama_tokenize(vocabulary, message.data(), static_cast<int32_t>(message.size()),
                    tokens.data(), static_cast<int32_t>(tokens.size()), false, false);
                require(count > 0 && count < 192 && size_t(count) <= tokens.size(), "Result tokenization failed");
                tokens.resize(count);
                if (index) std::cout << ',';
                std::cout << "{\"visible_utf8\":[";
                for (size_t i = 0; i < visible.size(); ++i) {
                    if (i) std::cout << ',';
                    std::cout << unsigned(static_cast<unsigned char>(visible[i]));
                }
                std::cout << "],\"token_ids\":[";
                for (size_t i = 0; i < tokens.size(); ++i) {
                    if (i) std::cout << ',';
                    std::cout << tokens[i];
                    const llama_batch batch{1, &tokens[i], nullptr, nullptr, nullptr, nullptr, nullptr};
                    require(llama_decode(text.get(), batch) == 0, "Visible result decode failed");
                }
                std::cout << "],\"text_after_result\":" << text_state(text.get());
                const auto * logits = scores(text.get());
                llama_token next = static_cast<llama_token>(std::max_element(logits, logits + vocab) - logits);
                const llama_batch batch{1, &next, nullptr, nullptr, nullptr, nullptr, nullptr};
                require(llama_decode(text.get(), batch) == 0, "Next text token decode failed");
                std::cout << ",\"next_token\":" << next << ",\"text_after_next\":" << text_state(text.get()) << '}';
            }
            require(park_text() == parked, "Handoff cleanup changed parked text");
            std::cout << ']';
        };
        if (reference) {
            std::array<wchar_t, 32768> results_path{};
            const auto length = GetEnvironmentVariableW(L"QWENPU_HANDOFF_RESULTS", results_path.data(),
                static_cast<DWORD>(results_path.size()));
            require(length < results_path.size(), "Reference results path is too long");
            if (length) {
                std::ifstream input(fs::path(results_path.data()), std::ios::binary);
                require(bool(input), "Could not open matched-history results");
                std::array<uint8_t, 4> header{};
                while (input.read(reinterpret_cast<char *>(header.data()), header.size())) {
                    const auto size = little(header.data(), 4);
                    require(size > 0 && size <= 4096, "Invalid matched-history result size");
                    std::string value(size, '\0');
                    require(bool(input.read(value.data(), size)), "Truncated matched-history result");
                    visible_results.push_back(std::move(value));
                }
                require(input.eof() && input.gcount() == 0, "Truncated matched-history framing");
            }
            std::cout << "{\"parked_text\":" << parked << ",\"handoffs\":";
            print_handoffs();
            std::cout << "}\n";
            return 0;
        }

        std::unique_ptr<llama_sampler, decltype(llama_sampler_free)> sampler(nullptr, llama_sampler_free);
        auto mtp = make_context(LLAMA_CONTEXT_TYPE_MTP, capacity);
        std::cerr << "stock KV callback: " << (kv_callback_off ? "off" : "on") << '\n';
        if (top2) {
            sampler.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            require(bool(sampler), "Could not create the stock sampler chain");
            auto * selector = llama_sampler_init_top_k(2);
            require(selector != nullptr, "Could not create stock top-k(2)");
            llama_sampler_chain_add(sampler.get(), selector);
            require(llama_set_sampler(mtp.get(), 0, sampler.get()), "Could not attach stock top-k(2)");
        }
        require(llama_n_ctx(mtp.get()) == capacity, "Unexpected stock MTP cache capacity");
        set_nextn(mtp.get(), true, true);
        auto * memory = llama_get_memory(mtp.get());
        const auto prefix = fs::path(argv[6]).wstring();
        const bool capture_feedback = GetEnvironmentVariableW(L"QWENPU_SCALAR_CAPTURE_FEEDBACK", nullptr, 0) != 0;
        float * raw_score_canary = nullptr;
        constexpr float canary_value = -1234567.f;
        bool checked_raw_score_canary = false;
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"context_cells\":" << capacity << ",\"seed_batch\":" << seed_batch
                  << ",\"flash_attention_requested\":" << (flash_mtp ? "true" : "false")
                  << ",\"parked_text\":" << parked << ",\"runs\":[";
        for (int i = 7; i < argc; ++i) {
            const auto begin = std::chrono::steady_clock::now();
            const auto guest = load_guest(argv[i]);
            const auto suffix = prefix + L".run-" + std::to_wstring(i - 7);
            llama_memory_clear(memory, true);
            require(position(mtp.get()) == -1, "Guest reset left an occupied cell");
            Floats hidden(width, 0.f);
            llama_pos pos = 0;
            std::ofstream kv_witness(fs::path(suffix + L".kv-witness.bin"), std::ios::binary);
            require(bool(kv_witness), "Could not open retained KV witness stream");
            std::ofstream score_audit;
            if (sample_audit) {
                score_audit.open(fs::path(suffix + L".score-audit.csv"));
                require(bool(score_audit), "Could not open score evidence stream");
                score_audit << std::setprecision(std::numeric_limits<float>::max_digits10)
                            << "position,winner,runner,winner_score,runner_score,margin,runner_ties\n";
            }
            BenchmarkWindow benchmark;
            std::vector<int64_t> action_times;
            auto previous_action_end = BenchmarkWindow::Time{};
            auto decode = [&](llama_token input, bool retained, bool emit) {
                require(pos < decode_limit, "Guest exceeded the stock decode budget");
                if (emit && input >= 313 && input <= 316) {
                    float margin = std::numeric_limits<float>::infinity();
                    const auto word = VXObserver::read(hidden.data(), observer.wrom_index, margin);
                    require(word % 9 == 0 && word / 9 < dot_wrom_rows, "VMAT exceeds the seeded wide-WROM range");
                }
                int8_t output = emit ? 1 : 0;
                const llama_batch batch{1, &input, hidden.data(), &pos, nullptr, nullptr, &output};
                witness.clear();
                witness.enabled = retained;
                witness.benchmark = benchmark.active ? &benchmark : nullptr;
                auto phase = benchmark.stamp();
                // The existing scheduler may wait for GPU work inside this call, including around witness capture.
                const int status = llama_decode(mtp.get(), batch);
                benchmark.split(phase, benchmark.decode_call);
                witness.benchmark = nullptr;
                if (benchmark.active) {
                    ++benchmark.decode_calls;
                    benchmark.output_calls += emit;
                }
                if (status != 0 || position(mtp.get()) != pos) {
                    throw std::runtime_error("Stock decode failed at position " + std::to_string(pos) +
                                             ", action " + std::to_string(input) + ", status " + std::to_string(status));
                }
                if (retained && !kv_callback_off) witness.write(kv_witness, static_cast<uint32_t>(pos), 1);
                benchmark.split(phase, benchmark.kv_host);
                llama_token next = -1;
                float margin = 0.f;
                if (emit) {
                    // The getters can wait for transfers queued in decode. Time the existing waits in place.
                    const auto * logits = top2 ? llama_get_sampled_logits_ith(mtp.get(), -1) :
                        llama_get_logits_ith(mtp.get(), -1);
                    const auto * ids = top2 ? llama_get_sampled_candidates_ith(mtp.get(), -1) : nullptr;
                    benchmark.split(phase, benchmark.logits_get);
                    constexpr int guard = 513;
                    int runner_id = -1;
                    int runner_ties = 0;
                    float winner_score = 0.f, runner_score = -std::numeric_limits<float>::infinity();
                    if (top2) {
                        require(logits && ids && llama_get_sampled_logits_count_ith(mtp.get(), -1) == 2 &&
                                llama_get_sampled_candidates_count_ith(mtp.get(), -1) == 2,
                                "Stock sampler did not emit exactly two scores and IDs");
                        require(ids[0] >= 0 && ids[0] <= guard && ids[1] >= 0 && ids[1] <= guard && ids[0] != ids[1] &&
                                std::isfinite(logits[0]) && std::isfinite(logits[1]), "Invalid stock top-2 candidates");
                        const int winner_index = logits[0] > logits[1] ? 0 : 1;
                        next = ids[winner_index];
                        runner_id = ids[1 - winner_index];
                        winner_score = logits[winner_index];
                        runner_score = logits[1 - winner_index];
                        require(runner_id != guard || runner_score == 0.f, "Nonzero trimmed head guard");
                    } else if (trimmed_head) {
                        require(logits && vocab == 248320, "Missing trimmed action scores");
                        require(std::all_of(logits, logits + guard + 1, [](float v) { return std::isfinite(v); }),
                                "Non-finite score in retained action IDs");
                        require(std::all_of(logits + guard + 1, logits + vocab, [](float v) {
                                    return v == -std::numeric_limits<float>::infinity();
                                }), "Expected the stock trimmed action mask");
                        require(logits[guard] == 0.f, "Nonzero trimmed head guard");
                    } else {
                        check_scores(logits);
                    }
                    if (!top2) {
                        next = static_cast<llama_token>(std::max_element(logits, logits + vocab) - logits);
                        winner_score = logits[next];
                    }
                    require(!trimmed_head || next != guard, "Trimmed head guard won");
                    benchmark.split(phase, benchmark.logits_scan);
                    const auto * feedback = get_nextn(mtp.get(), -1);
                    benchmark.split(phase, benchmark.feedback_get);
                    require(feedback && std::all_of(feedback, feedback + width, [](float x) { return std::isfinite(x); }),
                            "Invalid stock MTP feedback");
                    benchmark.split(phase, benchmark.feedback_check_and_copy);
                    if (!top2) {
                        for (int row = 0; row < vocab; ++row) {
                            if (row == next) continue;
                            if (logits[row] > runner_score) {
                                runner_score = logits[row];
                                runner_id = row;
                                runner_ties = 1;
                            } else if (sample_audit && logits[row] == runner_score) {
                                ++runner_ties;
                            }
                        }
                    }
                    margin = winner_score - runner_score;
                    require(margin > 0.f, "Stock action is tied");
                    benchmark.split(phase, benchmark.logits_scan);
                    hidden.assign(feedback, feedback + width);
                    benchmark.split(phase, benchmark.feedback_check_and_copy);
                    if (sample_audit) {
                        if (top2 && !checked_raw_score_canary) {
                            auto * raw_scores = llama_get_logits(mtp.get());
                            require(raw_scores && raw_scores != logits, "Raw and sampled host buffers are not distinct");
                            if (!raw_score_canary) {
                                raw_score_canary = raw_scores;
                                std::fill_n(raw_scores, vocab, canary_value);
                            } else {
                                require(raw_scores == raw_score_canary && std::all_of(raw_scores, raw_scores + vocab,
                                    [canary_value](float value) { return value == canary_value; }),
                                    "Sampled decode overwrote the raw host score buffer");
                                checked_raw_score_canary = true;
                            }
                        }
                        score_audit << pos << ',' << next << ',' << runner_id << ',' << winner_score << ','
                                    << runner_score << ',' << margin << ',' << runner_ties << '\n';
                        require(bool(score_audit), "Could not save score evidence");
                    }
                }
                if (!retained) require(llama_memory_seq_rm(memory, 0, pos, pos + 1), "Scratch-row removal failed");
                benchmark.split(phase, benchmark.kv_host);
                ++pos;
                return std::pair{next, margin};
            };
            decode(seed_miss, true, false);
            struct Seed { uint32_t address, word; bool reg; int dot_row = -1; };
            std::vector<Seed> seeds;
            Bytes raw_weights(wrom_end - wrom_base, 0);
            for (const auto & segment : guest.segments) {
                if (segment.address >= wrom_base) {
                    std::copy(segment.bytes.begin(), segment.bytes.end(), raw_weights.begin() + (segment.address - wrom_base));
                    continue;
                }
                require((segment.address & 3) == 0, "PT_LOAD start must be word aligned in this probe");
                for (size_t offset = 0; offset < segment.bytes.size(); offset += 4) {
                    const auto length = std::min<size_t>(4, segment.bytes.size() - offset);
                    const auto word = little(segment.bytes.data() + offset, static_cast<int>(length));
                    if (word) seeds.push_back({segment.address + static_cast<uint32_t>(offset), word, false});
                }
            }
            for (size_t cell = 0; cell < raw_weights.size(); cell += 8) {
                uint32_t packed_word = 0;
                for (int nibble = 0; nibble < 8; ++nibble) packed_word |= uint32_t(raw_weights[cell + nibble] & 15) << (4 * nibble);
                if (packed_word) seeds.push_back({packed_wrom_base + static_cast<uint32_t>(cell / 2), packed_word, false});
            }
            for (uint32_t row = 0; row < dot_wrom_rows; ++row) {
                const auto word = row * 9;
                seeds.push_back({dot_wrom_base + 4 * word, 0, false, static_cast<int>(row)});
            }
            seeds.push_back({2, ram_end, true});
            require(seeds.size() + 8 < capacity && seeds.size() + 1 < static_cast<size_t>(decode_limit),
                    "ELF exceeds available stock cache or decode budget");
            Floats seed_inputs(size_t(width) * seed_batch_size);
            std::array<llama_token, seed_batch_size> seed_tokens{};
            std::array<llama_pos, seed_batch_size> seed_positions{};
            std::array<int8_t, seed_batch_size> seed_outputs{};
            seed_tokens.fill(seed_word);
            for (size_t first = 0; first < seeds.size(); first += seed_batch) {
                const auto count = static_cast<int32_t>(std::min<size_t>(seed_batch, seeds.size() - first));
                std::fill(seed_inputs.begin(), seed_inputs.end(), 0.f);
                for (int row = 0; row < count; ++row) {
                    auto * input = seed_inputs.data() + size_t(row) * width;
                    const auto & item = seeds[first + row];
                    put_bits(input + write_address, item.address, 32);
                    put_bits(input + write_address + 32, item.reg ? 1 : 0, 1);
                    put_bits(input + value_lane, item.word, 32);
                    put_bits(input + fresh_handle, static_cast<uint32_t>(pos + row), handle_bits);
                    if (item.dot_row >= 0) {
                        const auto offset = static_cast<size_t>(item.dot_row) * 72;
                        for (int lane = 0; lane < 64; ++lane) {
                            const auto nibble = (raw_weights[offset + lane] & 15u) ^ 8u;
                            for (int bit = 0; bit < 4; ++bit) {
                                input[dot_payload + 64 * (lane / 16) + 16 * bit + lane % 16] = (nibble >> bit & 1u) ? 16.f : 0.f;
                            }
                        }
                        input[dot_payload + 256] = 16.f;
                    }
                    seed_positions[row] = pos + row;
                }
                witness.clear();
                witness.enabled = true;
                const llama_batch batch{count, seed_tokens.data(), seed_inputs.data(), seed_positions.data(),
                                        nullptr, nullptr, seed_outputs.data()};
                require(llama_decode(mtp.get(), batch) == 0, "Stock raw ELF seed batch failed");
                if (!kv_callback_off) witness.write(kv_witness, static_cast<uint32_t>(pos), static_cast<uint32_t>(count));
                pos += count;
            }
            const auto seed_state = state(mtp.get());
            write_file(suffix + L".seed-kv.bin", seed_state);
            std::ofstream retained(fs::path(suffix + L".writes-kv.bin"), std::ios::binary);
            require(bool(retained), "Could not open stock write snapshots");
            std::ofstream feedback_dump;
            if (capture_feedback) {
                feedback_dump.open(fs::path(suffix + L".feedback-f32.bin"), std::ios::binary);
                require(bool(feedback_dump), "Could not open stock feedback capture");
            }
            if (i != 7) std::cout << ',';
            std::cout << "{\"entry\":" << guest.entry << ",\"seed_positions\":" << pos
                      << ",\"seed_sha256\":\"" << sha256(seed_state.data(), seed_state.size())
                      << "\",\"full_write_snapshots\":" << (full_write_snapshots ? "true" : "false") << ",\"segments\":[";
            for (size_t part = 0; part < guest.segments.size(); ++part) {
                const auto & segment = guest.segments[part];
                if (part) std::cout << ',';
                std::cout << "{\"address\":" << segment.address << ",\"flags\":" << segment.flags
                          << ",\"mapped_bytes\":" << segment.bytes.size() << ",\"sha256\":\""
                          << sha256(segment.bytes.data(), segment.bytes.size()) << "\"}";
            }
            std::cout << "],\"diagnostic_reads\":" << (diagnostic_chars ? "true" : "false") << ",\"actions\":[";
            std::fill(hidden.begin(), hidden.end(), 0.f);
            put_bits(hidden.data() + entry_lane, guest.entry, 32);
            llama_token action = start;
            int event_count = 0;
            uint32_t retirement_count = 0;
            bool stopped_at_limit = false;
            std::string visible;
            const auto seed_ready = BenchmarkWindow::Clock::now();
            const auto seeded_positions = pos;
            auto progress_time = seed_ready;
            BenchmarkWindow::Time benchmark_end{};
            const char * stop_reason = "attempt_limit";
            if (!diagnostic_chars && benchmark_start_after == 0) benchmark.open(seed_ready, retirement_count, pos);
            float minimum = std::numeric_limits<float>::infinity();
            if (diagnostic_chars) {
                std::ifstream requests(fs::path(diagnostic_path.data()), std::ios::binary);
                require(bool(requests), "Could not open the diagnostic address stream");
                std::array<uint8_t, 8> request{};
                while (requests.read(reinterpret_cast<char *>(request.data()), request.size())) {
                    const auto address = little(request.data(), 4), reg = little(request.data() + 4, 4);
                    require(reg < 2, "Invalid diagnostic address namespace");
                    std::fill(hidden.begin(), hidden.end(), 0.f);
                    put_bits(hidden.data() + read_address, address, 32);
                    put_bits(hidden.data() + read_address + 32, reg, 1);
                    put_bits(hidden.data() + read_mark, 1, 1);
                    const auto at = pos;
                    const auto [next, margin] = decode(fetch, false, true);
                    const auto read = [&](int lane, int count) { return get_bits(hidden.data() + lane, count, minimum); };
                    if (event_count++) std::cout << ',';
                    std::cout << "{\"position\":" << at << ",\"diagnostic_address\":" << address
                              << ",\"register\":" << (reg ? "true" : "false") << ",\"word\":" << read(readback, 32)
                              << ",\"handle\":";
                    if (read(readback + 56, 1)) std::cout << read(readback + 32, 24);
                    else std::cout << "null";
                    std::cout << ",\"next\":" << next << ",\"logit_margin\":" << margin << '}';
                }
                require(requests.eof() && requests.gcount() == 0, "Truncated diagnostic address stream");
            } else while (true) {
                const bool writing = action == store || action == write_reg;
                const bool terminal = action == halt || action == fault;
                // Active observer markers are refreshed signs. Small inactive CUDA residues are not requests.
                const bool reading = observer.enabled ?
                    std::find(observer.read_tokens.begin(), observer.read_tokens.end(), action) != observer.read_tokens.end() :
                    hidden[read_mark] > 0.25f;
                const bool tile_read = observer.enabled && action == observer.tile_token;
                auto phase = benchmark.stamp();
                if (capture_feedback) {
                    require(bool(feedback_dump.write(reinterpret_cast<const char *>(hidden.data()),
                        static_cast<std::streamsize>(hidden.size() * sizeof(float)))), "Could not save stock feedback");
                }
                const auto read = [&](int lane, int count) { return get_bits(hidden.data() + lane, count, minimum); };
                const auto tile_address = tile_read ? read(read_address, 32) : 0u;
                const auto tile_wrom_word = tile_read ? VXObserver::read(hidden.data(), observer.wrom_index, minimum) : 0u;
                if (event_count++) std::cout << ',';
                std::cout << "{\"position\":" << pos << ",\"action\":" << action << ",\"read_marker\":" << hidden[read_mark];
                if (action != start) std::cout << ",\"feedback_sha256\":\""
                    << sha256(hidden.data(), hidden.size() * sizeof(float)) << '"';
                if (action == fetch) std::cout << ",\"fetch_pc\":" << read(read_address, 32);
                if (reading) {
                    std::cout << ",\"query\":{\"address\":" << read(read_address, 32) << ",\"register\":"
                              << (read(read_address + 32, 1) ? "true" : "false") << '}';
                }
                if (writing) {
                    std::cout << ",\"write\":{\"address\":" << read(write_address, 32) << ",\"register\":"
                              << (read(write_address + 32, 1) ? "true" : "false") << ",\"value\":"
                              << read(value_lane, 32) << '}';
                    benchmark.split(phase, benchmark.trace);
                    put_bits(hidden.data() + fresh_handle, static_cast<uint32_t>(pos), handle_bits);
                    phase = benchmark.stamp();
                }
                if (action == retire) {
                    std::cout << ",\"removed_handle\":";
                    if (read(old_present, 1)) {
                        const auto old = static_cast<llama_pos>(read(old_handle, handle_bits));
                        benchmark.split(phase, benchmark.trace);
                        require(old > 0 && old + 1 < pos && llama_memory_seq_rm(memory, 0, old, old + 1),
                                "Invalid checkpoint-issued removal handle");
                        benchmark.split(phase, benchmark.kv_host);
                        std::cout << old;
                    } else std::cout << "null";
                }
                if (action == commit || terminal) {
                    ++retirement_count;
                    if (hidden[putchar_valid] > 0.f) {
                        const auto output = read(exit_code, 9);
                        require(output < 256, "Guest output symbol is not a byte");
                        visible.push_back(static_cast<char>(output));
                    }
                    std::cout << ",\"retirement\":";
                    record(std::cout, static_cast<Action>(action), hidden.data(), minimum, observer);
                }
                if (terminal) {
                    if (action == halt && with_text) {
                        if (visible.empty()) visible = observer.enabled ? observer.summary(hidden.data(), minimum) :
                            std::to_string(read(exit_code, 32));
                        visible_results.push_back(visible);
                    }
                    std::cout << '}';
                    benchmark.split(phase, benchmark.trace);
                    benchmark_end = BenchmarkWindow::Clock::now();
                    stop_reason = action == halt ? "halt" : "fault";
                    break;
                }
                benchmark.split(phase, benchmark.trace);
                const auto [next, margin] = decode(action, writing, true);
                require(next >= start && next < vocab, "Invalid stock guest action token");
                phase = benchmark.stamp();
                if (writing && full_write_snapshots) {
                    const auto written = state(mtp.get());
                    require(written.size() <= UINT32_MAX, "Write snapshot exceeds record framing");
                    const auto size = static_cast<uint32_t>(written.size());
                    retained.write(reinterpret_cast<const char *>(&size), sizeof(size));
                    require(bool(retained.write(reinterpret_cast<const char *>(written.data()), size)),
                            "Could not save stock write snapshot");
                    benchmark.split(phase, benchmark.kv_host);
                    std::cout << ",\"write_kv_sha256\":\"" << sha256(written.data(), written.size()) << '"';
                }
                if (reading) {
                    const bool hit = read(readback + 32 + handle_bits, 1) != 0;
                    std::cout << ",\"response\":{\"value\":" << read(readback, 32) << ",\"handle\":";
                    if (hit) std::cout << read(readback + 32, handle_bits);
                    else std::cout << "null";
                    std::cout << '}';
                }
                if (tile_read) {
                    std::cout << ",\"extra_queries\":[{\"address\":" << (tile_address | 4u)
                              << ",\"register\":false,\"response\":{\"value\":"
                              << VXObserver::read(hidden.data(), observer.extra_words[0], minimum)
                              << "}},{\"address\":" << (packed_wrom_base + 4 * tile_wrom_word)
                              << ",\"register\":false,\"response\":{\"value\":"
                              << VXObserver::read(hidden.data(), observer.extra_words[1], minimum) << "}}]";
                }
                std::cout << ",\"next\":" << next << ",\"logit_margin\":" << margin << '}';
                benchmark.split(phase, benchmark.trace);
                if (time_actions && benchmark.active) {
                    const auto previous = action_times.empty() ? benchmark.start : previous_action_end;
                    action_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(phase - previous).count());
                    previous_action_end = phase;
                }
                action = next;
                if (stop_after && retirement_count >= stop_after) {
                    stopped_at_limit = true;
                    benchmark_end = BenchmarkWindow::Clock::now();
                    break;
                }
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - progress_time).count() >= 5) {
                    phase = now;
                    std::cerr << "stock VX: position " << pos << ", retired attempts " << retirement_count
                              << ", elapsed " << std::chrono::duration<double>(now - begin).count() << " s\n";
                    std::cout.flush();
                    benchmark.split(phase, benchmark.trace);
                    progress_time = now;
                }
                // A counted commit is complete after its decode and event. The next iteration starts the window.
                if (!benchmark.active && retirement_count == benchmark_start_after) {
                    benchmark.open(BenchmarkWindow::Clock::now(), retirement_count, pos);
                }
            }
            const auto final_state = state(mtp.get());
            if (time_actions) {
                require(action_times.size() == benchmark.decode_calls, "Action timings disagree with decode count");
                std::ofstream times(fs::path(suffix + L".action-ns.bin"), std::ios::binary);
                times.write(reinterpret_cast<const char *>(action_times.data()),
                            static_cast<std::streamsize>(action_times.size() * sizeof(action_times.front())));
                require(bool(times), "Could not save action timings");
            }
            retained.close();
            require(bool(retained), "Could not finish stock write snapshots");
            kv_witness.close();
            require(bool(kv_witness), "Could not finish retained KV witness stream");
            write_file(suffix + L".final-kv.bin", final_state);
            const auto after = with_text ? text_state(text.get()) : "null";
            require(after == parked, "Stock scalar modified parked Bonsai state");
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            std::cout << "],\"final_sha256\":\"" << sha256(final_state.data(), final_state.size())
                      << "\",\"final_feedback_sha256\":\"" << sha256(hidden.data(), hidden.size() * sizeof(float))
                      << "\",\"decode_positions\":" << pos << ",\"minimum_observed_bit\":" << minimum
                      << ",\"attempts_limit_reached\":" << (stopped_at_limit ? "true" : "false")
                      << ",\"retirement_attempts\":" << retirement_count
                      << ",\"wall_seconds\":" << elapsed << ",\"text_after\":" << after << '}';
            std::cout.flush();
            if (!diagnostic_chars) {
                benchmark.report(i - 7, benchmark_start_after, retirement_count, pos, seeded_positions,
                                 begin, seed_ready, benchmark_end, stop_reason, vocab, top2);
            }
            std::cerr << "stock scalar: completed input " << i - 7 << " in " << elapsed << " s\n";
        }
        PROCESS_MEMORY_COUNTERS memory_counters{};
        require(GetProcessMemoryInfo(GetCurrentProcess(), &memory_counters, sizeof(memory_counters)) != 0,
                "Could not read peak working set");
        if (witness.audit_top2) {
            require(witness.saw_top_k && witness.saw_gather, "Stock sampler graph did not expose both CUDA nodes");
            require(checked_raw_score_canary, "Raw score canary was not checked");
            std::cerr << "stock top-2 CUDA audit: TOP_K and GET_ROWS device buffers verified\n";
            std::cerr << "stock top-2 CUDA audit: separate raw host score buffer remained unchanged\n";
        }
        std::cout << "],\"handoffs\":";
        if (with_text) print_handoffs();
        else std::cout << "[]";
        std::cout << ",\"peak_working_set_bytes\":" << memory_counters.PeakWorkingSetSize << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "stock VX: " << error.what() << '\n';
        return 1;
    }
}
