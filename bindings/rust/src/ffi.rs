// SPDX-License-Identifier: MIT
//! Exact ABI 1 layout. This module is private; consumers use typed descriptors.
use std::ffi::{c_char, c_void};

#[repr(C)]
pub(crate) struct RmsNormDesc {
    pub struct_size: u32,
    pub abi_version: u32,
    pub device: i32,
    pub dtype: i32,
    pub rows: i64,
    pub hidden: i64,
    pub input_row_stride: i64,
    pub epsilon: f32,
    pub reserved: u32,
}

#[repr(C)]
pub(crate) struct GemmDesc {
    pub struct_size: u32,
    pub abi_version: u32,
    pub device: i32,
    pub output_dtype: i32,
    pub m: i64,
    pub n: i64,
    pub k: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Buffer {
    pub data: *mut c_void,
    pub size_bytes: usize,
}

extern "C" {
    pub(crate) fn aiter_abi_version() -> u32;
    pub(crate) fn aiter_last_error() -> *const c_char;
    pub(crate) fn aiter_status_string(status: i32) -> *const c_char;
    pub(crate) fn aiter_rmsnorm_prepare(
        desc: *const RmsNormDesc,
        library: *const c_char,
        plan: *mut *mut c_void,
    ) -> i32;
    pub(crate) fn aiter_gemm_prepare(
        desc: *const GemmDesc,
        library: *const c_char,
        plan: *mut *mut c_void,
    ) -> i32;
    pub(crate) fn aiter_execute(
        plan: *const c_void,
        buffers: *const Buffer,
        count: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn aiter_plan_destroy(plan: *mut c_void);
}
