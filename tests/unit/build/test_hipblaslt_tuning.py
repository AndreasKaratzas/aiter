# SPDX-License-Identifier: MIT
"""Compile the production tuner against controlled hipBLASLt candidate lists."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from common.paths import source_root


STUBS = r"""
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
// HIP exposes mixed-type min/max overloads used by the rotating-buffer code.
template<class A, class B> auto min(A a, B b) { return a < b ? a : b; }
template<class A, class B> auto max(A a, B b) { return a > b ? a : b; }
using hipblasStatus_t = int;
using hipblasLtHandle_t = int;
using hipblasLtMatmulDesc_t = int;
using hipblasLtMatrixLayout_t = int;
using hipStream_t = int;
using hipEvent_t = int;
using hipDataType = int;
constexpr int HIPBLAS_STATUS_SUCCESS = 0;
constexpr int HIPBLAS_STATUS_NOT_SUPPORTED = 1;
constexpr int hipblaslt_handle = 0, preference = 0;
struct Algorithm { int data = -1; };
struct hipblasLtMatmulHeuristicResult_t {
    Algorithm algo;
    size_t workspaceSize = 0;
    // Match the installed hipBLASLt default: unreturned slots look successful.
    int state = HIPBLAS_STATUS_SUCCESS;
};
int admitted, fail_mode, events = 0, buffers = 0, launches = 0;
std::set<int> visited;
#define CHECK_HIPBLAS_ERROR(expr) do { if ((expr) != 0) throw std::runtime_error("status"); } while (0)
#define CHECK_HIP_ERROR(expr) CHECK_HIPBLAS_ERROR(expr)
int hipblasLtMatmulAlgoGetHeuristic(int, int, int, int, int, int, int,
    int requested, hipblasLtMatmulHeuristicResult_t* values, int* count) {
    *count = admitted;
    for (int i = 0; i < admitted && i < requested; ++i) {
        // Failed-state results deliberately have no valid algorithm.
        values[i].state = fail_mode == 4 || (fail_mode == 3 && i == 0) ? 1 : 0;
        if (values[i].state == 0) values[i].algo.data = i;
    }
    return 0;
}
int hipEventCreate(int* event) { *event = ++events; return 0; }
int hipEventDestroy(int) { --events; return 0; }
int hipMalloc(void** p, size_t) { *p = malloc(1); ++buffers; return 0; }
int hipFree(void* p) { free(p); --buffers; return 0; }
int hipStreamSynchronize(int) { return 0; }
int realDataTypeSize(int) { return 1; }
void pre_gpu_time(int, double&, int) {}
void post_gpu_time(int, int, double& time, int) { time = 1; }
int hipblasLtMatmul(int, int, const void*, const void*, int, const void*, int,
    const void*, const void*, int, void*, int, const Algorithm* algo, void*, size_t, int) {
    if (algo->data < 0 || algo->data >= admitted)
        throw std::runtime_error("uninitialized algorithm launched");
    visited.insert(algo->data);
    ++launches;
    // mode1: every candidate fails; mode2: the first fails and later ones work.
    return fail_mode == 1 || (fail_mode == 2 && algo->data == 0) ? 1 : 0;
}
"""

MAIN = r"""
int main(int argc, char** argv) {
    admitted = std::atoi(argv[1]);
    fail_mode = std::atoi(argv[2]);
    std::vector<hipblasLtMatmulHeuristicResult_t> selected(1);
    selected[0].algo.data = 999;  // Must not survive an unsuccessful tuning call.
    const auto status = hipblasLt_online_tuning(0, 16, 16, 16, 0, 0, 0, 0,
        nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr, selected,
        1, 1, 1, 1, 0, 0, 0);
    if (events || buffers) return 2;
    if (admitted <= 0 || admitted > 32 || fail_mode == 4) {
        if (status != HIPBLAS_STATUS_NOT_SUPPORTED || !selected.empty() || launches) return 3;
    } else if (fail_mode == 1) {
        if (status != HIPBLAS_STATUS_NOT_SUPPORTED || !selected.empty()) return 4;
    } else {
        if (status != HIPBLAS_STATUS_SUCCESS || selected.size() != 1) return 5;
        if (selected[0].algo.data < (fail_mode == 2 || fail_mode == 3 ? 1 : 0) ||
            selected[0].algo.data >= admitted) return 6;
    }
    const auto valid_count = fail_mode == 4 ? 0 : admitted - (fail_mode == 3 ? 1 : 0);
    if (admitted > 0 && admitted <= 32 && visited.size() != size_t(valid_count)) return 7;
    return 0;
}
"""


class HipblasLtTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        native = (source_root() / "csrc/blas/hipbsolgemm.cu").read_text()
        # Extract the real function, preserving every branch being exercised.
        start = native.index("hipblasStatus_t hipblasLt_online_tuning(")
        body = native.index("{", start)
        depth = 1
        end = body + 1
        while depth:
            depth += (native[end] == "{") - (native[end] == "}")
            end += 1
        source = root / "tuner.cpp"
        source.write_text(STUBS + native[start:end] + MAIN)
        cls.binary = root / "tuner"
        subprocess.run(
            ["c++", "-std=c++17", "-O0", str(source), "-o", str(cls.binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def check_candidates(self, count, failure_mode=0):
        result = subprocess.run(
            [str(self.binary), str(count), str(failure_mode)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_zero_candidates_leave_no_resources_or_stale_selection(self):
        self.check_candidates(0)

    def test_partial_and_full_lists_launch_only_returned_algorithms(self):
        for count in (1, 2, 31, 32):
            with self.subTest(count=count):
                self.check_candidates(count)

    def test_failed_candidates_are_not_selected(self):
        self.check_candidates(2, failure_mode=1)
        self.check_candidates(2, failure_mode=2)

    def test_invalid_library_count_is_rejected_before_allocating(self):
        for count in (-1, 33):
            with self.subTest(count=count):
                self.check_candidates(count)

    def test_failed_heuristic_states_are_never_launched(self):
        self.check_candidates(2, failure_mode=3)
        self.check_candidates(2, failure_mode=4)
