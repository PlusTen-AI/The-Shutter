#include "st.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
// Just enough JSON for safetensors headers: objects, arrays, strings, numbers.
struct J {
    const char* p;
    const char* e;
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
    bool lit(char c) { ws(); if (p < e && *p == c) { p++; return true; } return false; }
    std::string str() {
        ws();
        if (*p != '"') { fprintf(stderr, "[engine] json: string expected\n"); abort(); }
        p++;
        std::string s;
        while (p < e && *p != '"') { if (*p == '\\') { p++; } s += *p++; }
        p++;
        return s;
    }
    double num() { ws(); char* q; double v = strtod(p, &q); p = q; return v; }
    void skip() {  // skip any value
        ws();
        if (*p == '"') { str(); return; }
        if (*p == '{') { p++; if (lit('}')) return; do { str(); lit(':'); skip(); } while (lit(',')); lit('}'); return; }
        if (*p == '[') { p++; if (lit(']')) return; do { skip(); } while (lit(',')); lit(']'); return; }
        while (p < e && *p != ',' && *p != '}' && *p != ']') p++;
    }
};
}  // namespace

bool StFile::open(const char* path) {
    fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat sb;
    fstat(fd, &sb);
    size = sb.st_size;
    map = (uint8_t*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { map = nullptr; return false; }
    madvise(map, size, MADV_SEQUENTIAL | MADV_WILLNEED);
    uint64_t n;
    memcpy(&n, map, 8);
    data_off = 8 + n;
    J j{(const char*)map + 8, (const char*)map + 8 + n};
    j.lit('{');
    if (j.lit('}')) return true;
    do {
        std::string key = j.str();
        j.lit(':');
        if (key == "__metadata__") { j.skip(); continue; }
        StTensor x;
        j.lit('{');
        do {
            std::string f = j.str();
            j.lit(':');
            if (f == "dtype") x.dtype = j.str();
            else if (f == "shape") { j.lit('['); if (!j.lit(']')) { do { x.shape.push_back((int64_t)j.num()); } while (j.lit(',')); j.lit(']'); } }
            else if (f == "data_offsets") { j.lit('['); x.off0 = (uint64_t)j.num(); j.lit(','); x.off1 = (uint64_t)j.num(); j.lit(']'); }
            else j.skip();
        } while (j.lit(','));
        j.lit('}');
        t[key] = x;
    } while (j.lit(','));
    return true;
}

void StFile::close() {
    if (map) munmap(map, size);
    if (fd >= 0) ::close(fd);
    map = nullptr;
    fd = -1;
}

const StTensor& StFile::get(const std::string& name) const {
    auto it = t.find(name);
    if (it == t.end() && !pfx.empty()) it = t.find(pfx + name);
    if (it == t.end()) { fprintf(stderr, "[engine] missing tensor %s\n", name.c_str()); abort(); }
    return it->second;
}
