// SPDX-License-Identifier: MIT
#include "aiter/aiter.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {
thread_local char last_error[2048]{};
using RmsLaunch   = void (*)(size_t, size_t, size_t, float, int, int, int, int, int, int, size_t);
using GemmPrepare = int (*)(int, int, int, int, void**);
using GemmLaunch =
    int (*)(const void*, const void*, const void*, const float*, const float*, void*, void*);
using GemmDestroy  = void (*)(void*);
using BackendError = const char* (*)();

aiter_status fail(aiter_status status, const char* message)
{
    std::strncpy(last_error, message ? message : "unknown error", sizeof(last_error) - 1);
    last_error[sizeof(last_error) - 1] = '\0';
    return status;
}

template <typename T>
T symbol(void* library, const char* name)
{
    dlerror();
    void* address     = dlsym(library, name);
    const char* error = dlerror();
    if(error || !address)
        throw std::runtime_error(error ? error : "missing backend symbol");
    return reinterpret_cast<T>(address);
}

void* open_library(const char* path)
{
    if(!path || !*path)
        throw std::invalid_argument("backend_library is required");
    std::ifstream stream(path, std::ios::binary);
    if(!stream)
        throw std::invalid_argument("backend_library cannot be opened");
    std::string bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    if(bytes.find("amdhsa--gfx950") == std::string::npos)
        throw std::invalid_argument("backend_library contains no gfx950 code object");
    // dlopen caches mutable pathnames. Load a sealed snapshot of the bytes we
    // inspected, and retain its file descriptor so that pathname cannot later
    // identify different code. Exact-byte keys also share one resident snapshot
    // between plans without depending on a filename or a hash collision.
    struct Libraries
    {
        std::mutex mutex;
        std::map<std::string, int> files;
    };
    // Captured GPU graphs can outlive plans and C++ static destruction order.
    // The OS releases this small per-artifact registry at process termination.
    static auto* resident = new Libraries;
    std::lock_guard<std::mutex> lock(resident->mutex);
    auto found = resident->files.find(bytes);
    int fd     = found == resident->files.end() ? -1 : found->second;
    struct File
    {
        int value;
        ~File()
        {
            if(value >= 0)
                close(value);
        }
    } fresh{-1};
    if(fd < 0)
    {
        fd = memfd_create("aiter-backend", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        if(fd < 0)
            throw std::runtime_error("cannot create immutable backend snapshot");
        fresh.value   = fd;
        size_t offset = 0;
        while(offset < bytes.size())
        {
            auto count = write(fd, bytes.data() + offset, bytes.size() - offset);
            if(count < 0 && errno == EINTR)
                continue;
            if(count <= 0)
                throw std::runtime_error("cannot write backend snapshot");
            offset += static_cast<size_t>(count);
        }
        if(fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) < 0)
            throw std::runtime_error("cannot seal backend snapshot");
    }
    const auto snapshot = "/proc/self/fd/" + std::to_string(fd);
    void* library       = dlopen(snapshot.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if(!library)
        throw std::runtime_error(dlerror());
    if(fresh.value >= 0)
    {
        // Retain the pathname even if allocation of the registry entry fails.
        fresh.value = -1;
        try
        {
            resident->files.emplace(std::move(bytes), fd);
        }
        catch(...)
        {
            dlclose(library);
            throw;
        }
    }
    return library;
}

bool dimension(int64_t value) { return value > 0 && value <= INT_MAX; }
bool overlaps(const aiter_buffer& a, size_t an, const aiter_buffer& b, size_t bn)
{
    auto ap = reinterpret_cast<uintptr_t>(a.data);
    auto bp = reinterpret_cast<uintptr_t>(b.data);
    return ap < bp + bn && bp < ap + an;
}
} // namespace

struct aiter_plan
{
    enum class Kind
    {
        RmsNorm,
        Gemm
    } kind;
    void* library = nullptr;
    int device    = 0;
    aiter_rmsnorm_desc rms{};
    aiter_gemm_desc gemm{};
    RmsLaunch rms_launch       = nullptr;
    GemmLaunch gemm_launch     = nullptr;
    GemmDestroy gemm_destroy   = nullptr;
    BackendError backend_error = nullptr;
    void* backend_plan         = nullptr;
    size_t sizes[5]{};
    size_t count = 0;
    ~aiter_plan()
    {
        if(backend_plan && gemm_destroy)
            gemm_destroy(backend_plan);
        if(library)
            dlclose(library);
    }
};

extern "C" uint32_t aiter_abi_version(void) { return AITER_ABI_VERSION; }
extern "C" const char* aiter_last_error(void) { return last_error; }
extern "C" const char* aiter_status_string(aiter_status status)
{
    switch(status)
    {
    case AITER_SUCCESS: return "success";
    case AITER_INVALID_ARGUMENT: return "invalid argument";
    case AITER_UNSUPPORTED: return "unsupported";
    case AITER_RUNTIME_ERROR: return "runtime error";
    default: return "unknown status";
    }
}

extern "C" aiter_status
aiter_rmsnorm_prepare(const aiter_rmsnorm_desc* d, const char* library, aiter_plan** output)
{
    if(!output)
        return fail(AITER_INVALID_ARGUMENT, "output plan pointer is required");
    *output = nullptr;
    if(!d || d->struct_size < sizeof(*d) || d->abi_version != AITER_ABI_VERSION || d->reserved ||
       !dimension(d->rows) || !dimension(d->hidden) || !dimension(d->input_row_stride) ||
       d->input_row_stride < d->hidden || d->dtype < AITER_FP16 || d->dtype > AITER_FP32 ||
       !std::isfinite(d->epsilon) || d->epsilon <= 0 || d->device < 0)
        return fail(AITER_INVALID_ARGUMENT, "invalid RMSNorm descriptor");
    try
    {
        hipDeviceProp_t properties{};
        auto status = hipGetDeviceProperties(&properties, d->device);
        if(status != hipSuccess)
            return fail(AITER_RUNTIME_ERROR, hipGetErrorString(status));
        if(std::strncmp(properties.gcnArchName, "gfx950", 6) != 0)
            return fail(AITER_UNSUPPORTED, "this native backend profile requires gfx950");
        auto plan     = std::make_unique<aiter_plan>();
        plan->kind    = aiter_plan::Kind::RmsNorm;
        plan->device  = d->device;
        plan->rms     = *d;
        plan->library = open_library(library);
        auto version  = symbol<int (*)()>(plan->library, "aiter_rmsnorm_backend_abi_version");
        if(version() != 1)
            return fail(AITER_UNSUPPORTED, "unsupported RMSNorm backend ABI");
        plan->rms_launch     = symbol<RmsLaunch>(plan->library, "rms_norm_opus");
        const size_t element = d->dtype == AITER_FP32 ? 4 : 2;
        plan->sizes[0] =
            (size_t(d->rows - 1) * size_t(d->input_row_stride) + size_t(d->hidden)) * element;
        plan->sizes[1] = size_t(d->hidden) * element;
        plan->sizes[2] = size_t(d->rows) * size_t(d->hidden) * element;
        plan->count    = 3;
        *output        = plan.release();
        last_error[0]  = '\0';
        return AITER_SUCCESS;
    }
    catch(const std::invalid_argument& e)
    {
        return fail(AITER_INVALID_ARGUMENT, e.what());
    }
    catch(const std::exception& e)
    {
        return fail(AITER_RUNTIME_ERROR, e.what());
    }
    catch(...)
    {
        return fail(AITER_RUNTIME_ERROR, "unknown backend exception");
    }
}

extern "C" aiter_status
aiter_gemm_prepare(const aiter_gemm_desc* d, const char* library, aiter_plan** output)
{
    if(!output)
        return fail(AITER_INVALID_ARGUMENT, "output plan pointer is required");
    *output = nullptr;
    if(!d || d->struct_size < sizeof(*d) || d->abi_version != AITER_ABI_VERSION ||
       !dimension(d->m) || !dimension(d->n) || !dimension(d->k) || d->device < 0 ||
       (d->output_dtype != AITER_FP16 && d->output_dtype != AITER_BF16))
        return fail(AITER_INVALID_ARGUMENT, "invalid blockscale GEMM descriptor");
    if(d->m % 16 || d->n % 128 || d->k % 256)
        return fail(AITER_UNSUPPORTED, "native CK tile requires M%16=N%128=K%256=0");
    try
    {
        hipDeviceProp_t properties{};
        auto status = hipGetDeviceProperties(&properties, d->device);
        if(status != hipSuccess)
            return fail(AITER_RUNTIME_ERROR, hipGetErrorString(status));
        if(std::strncmp(properties.gcnArchName, "gfx950", 6) != 0)
            return fail(AITER_UNSUPPORTED, "E4M3FN CK SDK profile requires gfx950");
        int current_device = -1;
        if(hipGetDevice(&current_device) != hipSuccess || current_device != d->device)
            return fail(AITER_INVALID_ARGUMENT, "select the descriptor device before preparation");
        auto plan     = std::make_unique<aiter_plan>();
        plan->kind    = aiter_plan::Kind::Gemm;
        plan->device  = d->device;
        plan->gemm    = *d;
        plan->library = open_library(library);
        auto version  = symbol<int (*)()>(plan->library, "aiter_ck_blockscale_abi_version");
        if(version() != 1)
            return fail(AITER_UNSUPPORTED, "unsupported CK backend ABI");
        auto prepare        = symbol<GemmPrepare>(plan->library, "aiter_ck_blockscale_prepare");
        plan->gemm_launch   = symbol<GemmLaunch>(plan->library, "aiter_ck_blockscale_launch");
        plan->gemm_destroy  = symbol<GemmDestroy>(plan->library, "aiter_ck_blockscale_destroy");
        plan->backend_error = symbol<BackendError>(plan->library, "aiter_ck_blockscale_last_error");
        int result =
            prepare(int(d->m), int(d->n), int(d->k), int(d->output_dtype), &plan->backend_plan);
        if(result)
            return fail(result == 2 ? AITER_UNSUPPORTED : AITER_RUNTIME_ERROR,
                        plan->backend_error());
        plan->sizes[0] = size_t(d->m) * size_t(d->k);
        plan->sizes[1] = size_t(d->n) * size_t(d->k);
        plan->sizes[2] = size_t(d->m) * size_t(d->k / 128) * 4;
        plan->sizes[3] = size_t(d->n / 128) * size_t(d->k / 128) * 4;
        plan->sizes[4] = size_t(d->m) * size_t(d->n) * 2;
        plan->count    = 5;
        *output        = plan.release();
        last_error[0]  = '\0';
        return AITER_SUCCESS;
    }
    catch(const std::invalid_argument& e)
    {
        return fail(AITER_INVALID_ARGUMENT, e.what());
    }
    catch(const std::exception& e)
    {
        return fail(AITER_RUNTIME_ERROR, e.what());
    }
    catch(...)
    {
        return fail(AITER_RUNTIME_ERROR, "unknown backend exception");
    }
}

extern "C" aiter_status
aiter_execute(const aiter_plan* plan, const aiter_buffer* buffers, size_t count, void* stream)
{
    if(!plan || !buffers || count != plan->count)
        return fail(AITER_INVALID_ARGUMENT, "wrong plan or buffer count");
    for(size_t i = 0; i < count; ++i)
    {
        auto address = reinterpret_cast<uintptr_t>(buffers[i].data);
        if(!address || address % 16 || buffers[i].size_bytes < plan->sizes[i] ||
           address > std::numeric_limits<uintptr_t>::max() - plan->sizes[i])
            return fail(AITER_INVALID_ARGUMENT,
                        "buffer is null, unaligned, too short, or overflows");
    }
    for(size_t i = 0; i + 1 < count; ++i)
        if(overlaps(buffers[i], plan->sizes[i], buffers[count - 1], plan->sizes[count - 1]))
            return fail(AITER_INVALID_ARGUMENT, "output overlaps an input");
    int device  = -1;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        return fail(AITER_RUNTIME_ERROR, hipGetErrorString(status));
    if(device != plan->device)
        return fail(AITER_INVALID_ARGUMENT, "calling thread selected the wrong device");
    for(size_t i = 0; i < count; ++i)
    {
        hipPointerAttribute_t attributes{};
        status = hipPointerGetAttributes(&attributes, buffers[i].data);
        if(status != hipSuccess || attributes.device != plan->device ||
           attributes.type != hipMemoryTypeDevice)
            return fail(AITER_INVALID_ARGUMENT,
                        "every buffer must be device memory on the plan device");
    }
    if(stream)
    {
        hipDevice_t stream_device = -1;
        status = hipStreamGetDevice(reinterpret_cast<hipStream_t>(stream), &stream_device);
        if(status != hipSuccess || stream_device != plan->device)
            return fail(AITER_INVALID_ARGUMENT, "stream belongs to another device or is invalid");
    }
    try
    {
        if(plan->kind == aiter_plan::Kind::RmsNorm)
        {
            const auto& d = plan->rms;
            plan->rms_launch(reinterpret_cast<size_t>(buffers[2].data),
                             reinterpret_cast<size_t>(buffers[0].data),
                             reinterpret_cast<size_t>(buffers[1].data),
                             d.epsilon,
                             int(d.rows),
                             int(d.hidden),
                             int(d.input_row_stride),
                             int(d.dtype),
                             0,
                             0,
                             reinterpret_cast<size_t>(stream));
            status = hipGetLastError();
            if(status != hipSuccess)
                return fail(AITER_RUNTIME_ERROR, hipGetErrorString(status));
        }
        else
        {
            int result = plan->gemm_launch(plan->backend_plan,
                                           buffers[0].data,
                                           buffers[1].data,
                                           static_cast<float*>(buffers[2].data),
                                           static_cast<float*>(buffers[3].data),
                                           buffers[4].data,
                                           stream);
            if(result)
                return fail(AITER_RUNTIME_ERROR, plan->backend_error());
        }
        last_error[0] = '\0';
        return AITER_SUCCESS;
    }
    catch(const std::exception& e)
    {
        return fail(AITER_RUNTIME_ERROR, e.what());
    }
    catch(...)
    {
        return fail(AITER_RUNTIME_ERROR, "unknown backend exception");
    }
}

extern "C" void aiter_plan_destroy(aiter_plan* plan) { delete plan; }
