// minimal safetensors reader (mmap + header parse)
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct StTensor {
    std::string dtype;           // "BF16", "F32", "I8", "U8", ...
    std::vector<int64_t> shape;
    uint64_t off0 = 0, off1 = 0; // byte range inside the data section
    uint64_t bytes() const { return off1 - off0; }
};

struct StFile {
    int fd = -1;
    uint64_t size = 0;
    uint8_t* map = nullptr;
    uint64_t data_off = 0;
    std::map<std::string, StTensor> t;
    std::string pfx;  // optional key prefix: get/has fall back to pfx + name (fine-tunes that keep ComfyUI's full keys)
    bool open(const char* path);
    void close();
    const StTensor& get(const std::string& name) const;  // aborts if missing
    bool has(const std::string& name) const { return t.count(name) > 0 || (!pfx.empty() && t.count(pfx + name) > 0); }
    const uint8_t* ptr(const StTensor& x) const { return map + data_off + x.off0; }
};
