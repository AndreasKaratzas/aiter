#include <fmt/core.h>
#include "pa_ragged.h"
#include <vector>
#include "../utils.h"

namespace aiter {

#define MD_NAME "pa_ragged"

void paged_attention_ragged(std::optional<std::string> folder,
                            void* query_ptr,
                            void* key_cache_ptr,
                            void* value_cache_ptr,
                            void* workspace_buffer_ptr,
                            int* kv_indptr_ptr,
                            int* kv_page_indices_ptr,
                            int* kv_last_page_lens_ptr,
                            const float* k_scale_ptr,
                            const float* v_scale_ptr,
                            const float* fp8_out_scale_ptr,
                            void* out_ptr,
                            const float* alibi_slopes_ptr,
                            const float scale,
                            const int num_seqs,
                            const int num_kv_heads,
                            const int num_heads,
                            const int max_num_partitions,
                            const int head_size,
                            const int block_size,
                            const float logits_soft_cap,
                            const int q_stride,
                            const int kv_block_stride,
                            const int kv_head_stride,
                            const int kv_seq_stride,
                            const std::string dtype,
                            const std::string kv_dtype,
                            const std::string kv_cache_dtype,
                            const std::string out_dtype,
                            const hipStream_t stream)
{
    const int gqa_ratio = num_heads / num_kv_heads;
    int device = 0;
    int warp_size = 0;
    const auto device_status = hipGetDevice(&device);
    if(device_status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(device_status));
    const auto warp_status = hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device);
    if(warp_status != hipSuccess || warp_size <= 0)
        throw std::runtime_error("Cannot resolve HIP wave size for ragged attention");
    const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, warp_size);
    const std::string alibi_enabled = alibi_slopes_ptr ? "true" : "false";
    std::list<std::string> args{std::to_string(gqa_ratio),
                                std::to_string(head_size),
                                std::to_string(npar_loops),
                                dtype,
                                kv_dtype,
                                kv_cache_dtype,
                                out_dtype,
                                std::to_string(block_size),
                                alibi_enabled};
    const std::string func_name = get_default_func_name(MD_NAME, args);

    if(!folder)
    {
        folder = func_name;
    }

    if(not_built(folder.value()))
    {
        args.push_back(func_name);
        execute_cmd(R"(python3 -m aiter.ops._native.pa.pa_ragged \
                                    --gqa_ratio={} \
                                    --head_size={} \
                                    --npar_loops={} \
                                    --dtype={} \
                                    --kv_dtype={} \
                                    --fp8_kv_dtype={} \
                                    --out_dtype={} \
                                    --block_size={} \
                                    --alibi_enabled={} \
                                    --func_name={})", args);
    }
    run_lib(func_name,
            folder.value(),
            out_ptr,
            workspace_buffer_ptr,
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            kv_indptr_ptr,
            kv_page_indices_ptr,
            kv_last_page_lens_ptr,
            alibi_slopes_ptr,
            static_cast<const float*>(nullptr), // unquantized query
            k_scale_ptr,
            v_scale_ptr,
            fp8_out_scale_ptr,
            scale,
            logits_soft_cap,
            num_seqs,
            num_kv_heads,
            num_heads,
            max_num_partitions,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            kv_seq_stride,
            reinterpret_cast<void*>(stream));
}
#undef MD_NAME
#undef DIVIDE_ROUND_UP
} // namespace aiter
