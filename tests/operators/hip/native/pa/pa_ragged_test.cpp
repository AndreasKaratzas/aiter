// SPDX-License-Identifier: MIT
// A numerical consumer of the legacy native ragged-attention bridge.
#include "pa_ragged.h"
#include <iostream>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check_hip(hipError_t result)
{
    if(result != hipSuccess)
        throw std::runtime_error(hipGetErrorString(result));
}

template <typename T> class DeviceBuffer
{
public:
    explicit DeviceBuffer(const std::vector<T>& values) : count(values.size())
    {
        check_hip(hipMalloc(&pointer, count * sizeof(T)));
        try
        {
            check_hip(hipMemcpy(pointer, values.data(), count * sizeof(T), hipMemcpyHostToDevice));
        }
        catch(...)
        {
            hipFree(pointer);
            throw;
        }
    }
    ~DeviceBuffer() { hipFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    std::vector<T> read() const
    {
        std::vector<T> values(count);
        check_hip(hipMemcpy(values.data(), pointer, count * sizeof(T), hipMemcpyDeviceToHost));
        return values;
    }
    T* pointer = nullptr;
    size_t count;
};

void check_attention()
{
    check_hip(hipSetDevice(0));
    constexpr int sequences = 2, heads = 8, dimension = 128;
    constexpr int block_size = 16, blocks = 4, partitions = 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    const std::vector<int> pages{1, 0, 3, 2};
    const std::vector<int> lengths{21, 32};
    std::vector<half> query(sequences * heads * dimension);
    std::vector<half> keys(blocks * block_size * dimension);
    std::vector<half> values(keys.size());
    for(size_t i = 0; i < query.size(); ++i)
        query[i] = __float2half(0.5f * std::sin(static_cast<float>(i) * 0.17f));
    for(size_t i = 0; i < keys.size(); ++i)
    {
        keys[i] = __float2half(0.5f * std::cos(static_cast<float>(i) * 0.07f));
        values[i] = __float2half(0.5f * std::sin(static_cast<float>(i) * 0.11f));
    }
    DeviceBuffer<half> d_query(query), d_keys(keys), d_values(values);
    DeviceBuffer<half> output(std::vector<half>(query.size(), __float2half(NAN)));
    const size_t workspace_bytes = sequences * heads * partitions *
                                   (2 * sizeof(float) + dimension * sizeof(half));
    DeviceBuffer<unsigned char> workspace(std::vector<unsigned char>(workspace_bytes, 0));
    DeviceBuffer<int> indptr(std::vector<int>{0, 2, 4}), indices(pages);
    DeviceBuffer<int> last_page(std::vector<int>{5, 16});
    DeviceBuffer<float> scales(std::vector<float>{1.0f});
    aiter::paged_attention_ragged(
        std::nullopt, d_query.pointer, d_keys.pointer, d_values.pointer,
        workspace.pointer, indptr.pointer, indices.pointer, last_page.pointer,
        scales.pointer, scales.pointer, nullptr, output.pointer, nullptr,
        scale, sequences, 1, heads, partitions, dimension, block_size, 0.0f,
        heads * dimension, dimension * block_size, dimension * block_size,
        dimension, "_Float16", "_Float16", "auto", "_Float16", nullptr);
    check_hip(hipGetLastError());
    check_hip(hipDeviceSynchronize());
    const auto actual = output.read();
    for(int sequence = 0; sequence < sequences; ++sequence)
        for(int head = 0; head < heads; ++head)
        {
            std::vector<double> weights(lengths[sequence], 0.0);
            for(int token = 0; token < lengths[sequence]; ++token)
            {
                const int block = pages[sequence * 2 + token / block_size];
                for(int dim = 0; dim < dimension; ++dim)
                {
                    const int key_index = block * dimension * block_size +
                                          (token % block_size) * dimension + dim;
                    weights[token] += __half2float(query[(sequence * heads + head) * dimension + dim]) *
                                      __half2float(keys[key_index]);
                }
                weights[token] *= scale;
            }
            const double maximum = *std::max_element(weights.begin(), weights.end());
            double denominator = 0.0;
            for(double& weight : weights)
            {
                weight = std::exp(weight - maximum);
                denominator += weight;
            }
            for(int dim = 0; dim < dimension; ++dim)
            {
                double expected = 0.0;
                for(int token = 0; token < lengths[sequence]; ++token)
                {
                    const int block = pages[sequence * 2 + token / block_size];
                    const int value_index = block * dimension * block_size + (token % block_size) * dimension + dim;
                    expected += weights[token] * __half2float(values[value_index]) / denominator;
                }
                const float observed = __half2float(actual[(sequence * heads + head) * dimension + dim]);
                if(!std::isfinite(observed) ||
                   std::abs(observed - expected) > 0.002 + 0.01 * std::abs(expected))
                    throw std::runtime_error("Ragged attention differs from CPU reference at sequence=" +
                        std::to_string(sequence) + " head=" + std::to_string(head) +
                        " dimension=" + std::to_string(dim));
            }
        }
}
} // namespace

int main()
{
    try
    {
        check_attention();
        std::cout << "validated_values=2048" << std::endl;
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
