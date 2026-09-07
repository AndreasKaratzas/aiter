// SPDX-License-Identifier: MIT
// Integrity of explicitly admitted precompiled kernel resources. This is not
// publisher authentication: the process owner selects the expected index hash.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace aiter::kernels {

inline std::string sha256(const void* input, size_t length)
{
    constexpr std::array<uint32_t, 64> constants = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    std::array<uint32_t, 8> state = {0x6a09e667,
                                     0xbb67ae85,
                                     0x3c6ef372,
                                     0xa54ff53a,
                                     0x510e527f,
                                     0x9b05688c,
                                     0x1f83d9ab,
                                     0x5be0cd19};
    auto rotate                   = [](uint32_t value, unsigned bits) {
        return (value >> bits) | (value << (32 - bits));
    };
    const auto* bytes   = static_cast<const unsigned char*>(input);
    const size_t blocks = length / 64 + ((length % 64) < 56 ? 1 : 2);
    for(size_t block = 0; block < blocks; ++block)
    {
        std::array<unsigned char, 64> data{};
        const size_t offset = block * 64;
        for(size_t i = 0; i < 64; ++i)
        {
            if(offset + i < length)
                data[i] = bytes[offset + i];
            else if(offset + i == length)
                data[i] = 0x80;
        }
        if(block + 1 == blocks)
            for(unsigned i = 0; i < 8; ++i)
                data[63 - i] = static_cast<unsigned char>((uint64_t(length) * 8) >> (i * 8));
        std::array<uint32_t, 64> words{};
        for(unsigned i = 0; i < 16; ++i)
            words[i] = (uint32_t(data[4 * i]) << 24) | (uint32_t(data[4 * i + 1]) << 16) |
                       (uint32_t(data[4 * i + 2]) << 8) | uint32_t(data[4 * i + 3]);
        for(unsigned i = 16; i < 64; ++i)
        {
            auto x = words[i - 15], y = words[i - 2];
            words[i] = words[i - 16] + (rotate(x, 7) ^ rotate(x, 18) ^ (x >> 3)) + words[i - 7] +
                       (rotate(y, 17) ^ rotate(y, 19) ^ (y >> 10));
        }
        auto work = state;
        for(unsigned i = 0; i < 64; ++i)
        {
            auto a = work[0], b = work[1], c = work[2], e = work[4], f = work[5], g = work[6];
            uint32_t first = work[7] + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) +
                             ((e & f) ^ (~e & g)) + constants[i] + words[i];
            uint32_t second =
                (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            work = {first + second, a, b, c, work[3] + first, e, f, g};
        }
        for(unsigned i = 0; i < 8; ++i)
            state[i] += work[i];
    }
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for(auto word : state)
        result << std::setw(8) << word;
    return result.str();
}

inline void verify_object(const std::string& root,
                          const std::string& relative,
                          const void* bytes,
                          size_t size,
                          const char* expected_index)
{
    if(!expected_index || std::strlen(expected_index) != 64)
        throw std::runtime_error("kernel resources require explicit verified admission");
    if(relative.empty() || relative.front() == '/' || relative.find("..") != std::string::npos ||
       relative.find_first_of("\\\n\r\t ") != std::string::npos)
        throw std::runtime_error("kernel resource path is invalid");
    std::ifstream file(root + "/objects.sha256", std::ios::binary);
    if(!file)
        throw std::runtime_error("admitted kernel index is unavailable");
    std::string index{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if(sha256(index.data(), index.size()) != expected_index)
        throw std::runtime_error("admitted kernel index integrity mismatch");
    std::istringstream lines(index);
    std::string hash, path;
    size_t expected_size;
    while(lines >> hash >> expected_size >> path)
        if(path == relative)
        {
            if(expected_size != size || sha256(bytes, size) != hash)
                throw std::runtime_error("admitted kernel object integrity mismatch: " + relative);
            return;
        }
    throw std::runtime_error("kernel object is absent from admitted index: " + relative);
}

} // namespace aiter::kernels

// Weak inline definition is shared by translation units in one extension.
extern "C" __attribute__((visibility("default"), used)) inline int
aiter_kernel_resource_abi_version()
{ return 1; }
