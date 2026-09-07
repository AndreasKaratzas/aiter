// SPDX-License-Identifier: MIT
//! Typed descriptors and owned plans over AITER's native C ABI.
//!
//! Python and Rust use the same native implementations. Rust does not select
//! kernels independently, own HIP allocations, or add a compiler dependency.
//! Enqueue is explicitly unsafe: raw asynchronous GPU lifetimes cannot be
//! inferred from a temporary Rust slice. See [`Plan::enqueue`].
#![deny(unsafe_op_in_unsafe_fn)]

mod ffi;

use std::{
    ffi::{c_void, CStr, CString},
    fmt,
    marker::PhantomData,
    mem::size_of,
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

const ABI_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Fp16,
    Bf16,
    Fp32,
}

impl DType {
    fn native(self) -> i32 {
        match self {
            Self::Fp16 => 0,
            Self::Bf16 => 1,
            Self::Fp32 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    InvalidArgument,
    Unsupported,
    RuntimeError,
    Unknown(i32),
}

#[derive(Debug)]
pub struct Error {
    pub status: Status,
    message: String,
}

impl Error {
    fn argument(message: impl Into<String>) -> Self {
        Self {
            status: Status::InvalidArgument,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(output, "{:?}: {}", self.status, self.message)
    }
}
impl std::error::Error for Error {}

fn check(status: i32) -> Result<(), Error> {
    if status == 0 {
        return Ok(());
    }
    // SDK error text is thread-local and copied before another SDK call.
    let message = unsafe {
        let mut pointer = ffi::aiter_last_error();
        if pointer.is_null() || *pointer == 0 {
            pointer = ffi::aiter_status_string(status);
        }
        if pointer.is_null() {
            "unknown native error".into()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(Error {
        status: match status {
            1 => Status::InvalidArgument,
            2 => Status::Unsupported,
            3 => Status::RuntimeError,
            other => Status::Unknown(other),
        },
        message,
    })
}

fn library_path(path: &Path) -> Result<CString, Error> {
    if unsafe { ffi::aiter_abi_version() } != ABI_VERSION {
        return Err(Error {
            status: Status::Unsupported,
            message: "native SDK ABI differs from this frontend".into(),
        });
    }
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| Error::argument("provider path contains NUL"))
}

fn dimension(value: usize, name: &str) -> Result<i64, Error> {
    if value == 0 {
        return Err(Error::argument(format!("{name} must be positive")));
    }
    i64::try_from(value).map_err(|_| Error::argument(format!("{name} exceeds ABI range")))
}

/// RMSNorm metadata. Preparation copies this descriptor into the native plan.
pub struct RmsNorm {
    descriptor: ffi::RmsNormDesc,
}

impl RmsNorm {
    pub fn new(device: u32, dtype: DType, rows: usize, hidden: usize) -> Result<Self, Error> {
        let hidden = dimension(hidden, "hidden")?;
        Ok(Self {
            descriptor: ffi::RmsNormDesc {
                struct_size: size_of::<ffi::RmsNormDesc>() as u32,
                abi_version: ABI_VERSION,
                device: i32::try_from(device)
                    .map_err(|_| Error::argument("device exceeds ABI range"))?,
                dtype: dtype.native(),
                rows: dimension(rows, "rows")?,
                hidden,
                input_row_stride: hidden,
                epsilon: 1e-6,
                reserved: 0,
            },
        })
    }

    pub fn epsilon(mut self, value: f32) -> Result<Self, Error> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::argument("epsilon must be finite and positive"));
        }
        self.descriptor.epsilon = value;
        Ok(self)
    }

    pub fn input_row_stride(mut self, value: usize) -> Result<Self, Error> {
        let stride = dimension(value, "input_row_stride")?;
        if stride < self.descriptor.hidden {
            return Err(Error::argument("input rows overlap"));
        }
        self.descriptor.input_row_stride = stride;
        Ok(self)
    }

    pub fn prepare(&self, provider: impl AsRef<Path>) -> Result<Plan<3>, Error> {
        let path = library_path(provider.as_ref())?;
        let mut raw = std::ptr::null_mut();
        check(unsafe { ffi::aiter_rmsnorm_prepare(&self.descriptor, path.as_ptr(), &mut raw) })?;
        Plan::from_native(raw)
    }
}

/// Ordinary E4M3FN blockscale GEMM; layouts and target restrictions come from ABI 1.
pub struct Gemm {
    descriptor: ffi::GemmDesc,
}

impl Gemm {
    pub fn new(
        device: u32,
        output_dtype: DType,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            descriptor: ffi::GemmDesc {
                struct_size: size_of::<ffi::GemmDesc>() as u32,
                abi_version: ABI_VERSION,
                device: i32::try_from(device)
                    .map_err(|_| Error::argument("device exceeds ABI range"))?,
                output_dtype: output_dtype.native(),
                m: dimension(m, "m")?,
                n: dimension(n, "n")?,
                k: dimension(k, "k")?,
            },
        })
    }

    pub fn prepare(&self, provider: impl AsRef<Path>) -> Result<Plan<5>, Error> {
        let path = library_path(provider.as_ref())?;
        let mut raw = std::ptr::null_mut();
        check(unsafe { ffi::aiter_gemm_prepare(&self.descriptor, path.as_ptr(), &mut raw) })?;
        Plan::from_native(raw)
    }
}

/// Borrowed device allocation. This wrapper neither allocates nor frees GPU memory.
pub struct DeviceBuffer<'a> {
    raw: ffi::Buffer,
    _borrow: PhantomData<&'a mut [u8]>,
}

impl<'a> DeviceBuffer<'a> {
    /// Borrow a HIP allocation from its actual owner.
    ///
    /// # Safety
    /// `data` must identify at least `size_bytes` of live device storage for `'a`.
    /// The owner must respect exclusive mutable access while this borrow exists.
    /// Enqueued work has additional lifetime obligations documented on `enqueue`.
    pub unsafe fn from_raw_parts(data: *mut c_void, size_bytes: usize) -> Result<Self, Error> {
        if data.is_null() || size_bytes == 0 {
            return Err(Error::argument("device buffer is empty"));
        }
        Ok(Self {
            raw: ffi::Buffer { data, size_bytes },
            _borrow: PhantomData,
        })
    }
}

/// Own a native preparation handle. The buffer count is part of its Rust type.
///
/// A plan is not Clone, Send or Sync. This frontend does not claim automatic HIP
/// device selection or cross-thread context ownership. Destroying a handle does
/// not unload code used by a graph: executable retention belongs to the SDK.
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<aiter::Plan<3>>();
/// ```
///
/// ```compile_fail
/// fn require_sync<T: Sync>() {}
/// require_sync::<aiter::Plan<3>>();
/// ```
pub struct Plan<const BUFFERS: usize> {
    raw: NonNull<c_void>,
    _thread: PhantomData<Rc<()>>,
}

impl<const BUFFERS: usize> Plan<BUFFERS> {
    fn from_native(raw: *mut c_void) -> Result<Self, Error> {
        let raw = NonNull::new(raw).ok_or_else(|| Error {
            status: Status::RuntimeError,
            message: "native preparation returned an empty handle".into(),
        })?;
        Ok(Self {
            raw,
            _thread: PhantomData,
        })
    }

    /// Validate buffers and enqueue the retained native implementation.
    ///
    /// RMSNorm order is input, weight, output. GEMM order is x, w, x_scale,
    /// w_scale, output. The SDK checks spans, alignment, devices and aliases.
    /// Success means enqueue, not completion. This call does not synchronize.
    ///
    /// # Safety
    /// Select the plan's HIP device on this thread and supply a valid stream
    /// from that device (null denotes its default stream). Keep every allocation
    /// and the stream valid until GPU completion, including future graph replays.
    /// Do not read, overwrite, free or give conflicting access to these buffers
    /// while queued work uses them. Rust cannot infer those asynchronous lifetimes
    /// from this temporary slice; the caller must establish completion explicitly.
    pub unsafe fn enqueue(
        &self,
        buffers: &mut [DeviceBuffer<'_>; BUFFERS],
        stream: *mut c_void,
    ) -> Result<(), Error> {
        let native: [ffi::Buffer; BUFFERS] = std::array::from_fn(|index| buffers[index].raw);
        check(unsafe { ffi::aiter_execute(self.raw.as_ptr(), native.as_ptr(), BUFFERS, stream) })
    }
}

impl<const BUFFERS: usize> Drop for Plan<BUFFERS> {
    fn drop(&mut self) {
        unsafe { ffi::aiter_plan_destroy(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_abi_one() {
        assert_eq!(unsafe { ffi::aiter_abi_version() }, ABI_VERSION);
        assert_eq!(size_of::<ffi::RmsNormDesc>(), 48);
        assert_eq!(size_of::<ffi::GemmDesc>(), 40);
        assert_eq!(size_of::<ffi::Buffer>(), 16);
        assert_eq!(std::mem::align_of::<ffi::RmsNormDesc>(), 8);
        assert_eq!(DType::Fp16.native(), 0);
        assert_eq!(DType::Bf16.native(), 1);
        assert_eq!(DType::Fp32.native(), 2);
    }

    #[test]
    fn field_offsets_match_the_c_header() {
        // Address projection does not read uninitialized memory. This works on
        // our Rust 1.75 minimum without the newer offset_of! standard macro.
        macro_rules! offset {
            ($ty:ty, $field:ident) => {{
                let storage = std::mem::MaybeUninit::<$ty>::uninit();
                let base = storage.as_ptr();
                unsafe { std::ptr::addr_of!((*base).$field) as usize - base as usize }
            }};
        }
        assert_eq!(offset!(ffi::RmsNormDesc, struct_size), 0);
        assert_eq!(offset!(ffi::RmsNormDesc, abi_version), 4);
        assert_eq!(offset!(ffi::RmsNormDesc, device), 8);
        assert_eq!(offset!(ffi::RmsNormDesc, dtype), 12);
        assert_eq!(offset!(ffi::RmsNormDesc, rows), 16);
        assert_eq!(offset!(ffi::RmsNormDesc, hidden), 24);
        assert_eq!(offset!(ffi::RmsNormDesc, input_row_stride), 32);
        assert_eq!(offset!(ffi::RmsNormDesc, epsilon), 40);
        assert_eq!(offset!(ffi::RmsNormDesc, reserved), 44);
        assert_eq!(offset!(ffi::GemmDesc, struct_size), 0);
        assert_eq!(offset!(ffi::GemmDesc, abi_version), 4);
        assert_eq!(offset!(ffi::GemmDesc, device), 8);
        assert_eq!(offset!(ffi::GemmDesc, output_dtype), 12);
        assert_eq!(offset!(ffi::GemmDesc, m), 16);
        assert_eq!(offset!(ffi::GemmDesc, n), 24);
        assert_eq!(offset!(ffi::GemmDesc, k), 32);
        assert_eq!(offset!(ffi::Buffer, data), 0);
        assert_eq!(offset!(ffi::Buffer, size_bytes), 8);
    }

    #[test]
    fn malformed_metadata_is_rejected_without_gpu_work() {
        assert!(RmsNorm::new(0, DType::Bf16, 0, 16).is_err());
        assert!(RmsNorm::new(u32::MAX, DType::Bf16, 1, 16).is_err());
        assert!(RmsNorm::new(0, DType::Bf16, 1, 16)
            .unwrap()
            .input_row_stride(8)
            .is_err());
        assert!(Gemm::new(0, DType::Bf16, 16, 128, 0).is_err());
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(RmsNorm::new(0, DType::Bf16, 1, 16)
                .unwrap()
                .epsilon(value)
                .is_err());
        }
    }
}
