// SPDX-License-Identifier: MIT
//! Execute the installed SDK through Rust with owned test allocations and explicit waits.
use aiter::{DType, DeviceBuffer, Gemm, RmsNorm, Status};
use std::{error::Error, ffi::c_void, path::PathBuf};

#[link(name = "amdhip64")]
extern "C" {
    fn hipSetDevice(device: i32) -> i32;
    fn hipMalloc(pointer: *mut *mut c_void, bytes: usize) -> i32;
    fn hipFree(pointer: *mut c_void) -> i32;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn hipDeviceSynchronize() -> i32;
}

fn hip(code: i32) -> Result<(), Box<dyn Error>> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("HIP status {code}").into())
    }
}

struct Allocation {
    pointer: *mut c_void,
    bytes: usize,
}

impl Allocation {
    fn new(values: &[u8]) -> Result<Self, Box<dyn Error>> {
        let mut pointer = std::ptr::null_mut();
        hip(unsafe { hipMalloc(&mut pointer, values.len()) })?;
        let allocation = Self {
            pointer,
            bytes: values.len(),
        };
        hip(unsafe { hipMemcpy(pointer, values.as_ptr().cast(), values.len(), 1) })?;
        Ok(allocation)
    }

    fn borrow(&mut self) -> DeviceBuffer<'_> {
        // Allocation owns these bytes and its mutable borrow prevents free/reuse.
        unsafe { DeviceBuffer::from_raw_parts(self.pointer, self.bytes) }.unwrap()
    }

    fn read(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut values = vec![0; self.bytes];
        hip(unsafe { hipMemcpy(values.as_mut_ptr().cast(), self.pointer, self.bytes, 2) })?;
        Ok(values)
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe { hipFree(self.pointer) };
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let prefix = PathBuf::from(std::env::var_os("AITER_SDK_DIR").ok_or("AITER_SDK_DIR missing")?);
    let library = [prefix.join("lib"), prefix.join("lib64")]
        .into_iter()
        .find(|path| path.join("libaiter.so").is_file())
        .ok_or("AITER SDK libaiter.so is missing")?;
    let providers = library.join("aiter");
    hip(unsafe { hipSetDevice(0) })?;

    let values: Vec<f32> = (0..64).map(|index| (index as f32 - 31.0) / 16.0).collect();
    let encoded: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let weights: Vec<u8> = (0..16).flat_map(|_| 1.0f32.to_ne_bytes()).collect();
    let mut input = Allocation::new(&encoded)?;
    let mut weight = Allocation::new(&weights)?;
    let mut output = Allocation::new(&[0; 256])?;
    let rms = RmsNorm::new(0, DType::Fp32, 4, 16)?
        .prepare(providers.join("libaiter_rmsnorm_backend.so"))?;

    {
        // A too-small span is rejected by the shared C SDK before enqueue.
        let short = unsafe { DeviceBuffer::from_raw_parts(output.pointer, 1) }?;
        let error = unsafe {
            rms.enqueue(
                &mut [input.borrow(), weight.borrow(), short],
                std::ptr::null_mut(),
            )
        }
        .err()
        .ok_or("short output unexpectedly accepted")?;
        assert_eq!(error.status, Status::InvalidArgument);
    }
    for _ in 0..3 {
        // All owners remain alive, on the selected device, until the explicit wait.
        unsafe {
            rms.enqueue(
                &mut [input.borrow(), weight.borrow(), output.borrow()],
                std::ptr::null_mut(),
            )
        }?;
        hip(unsafe { hipDeviceSynchronize() })?;
    }
    let bytes = output.read()?;
    for row in 0..4 {
        let mean = values[row * 16..(row + 1) * 16]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / 16.0;
        for column in 0..16 {
            let index = row * 16 + column;
            let actual = f32::from_ne_bytes(bytes[index * 4..index * 4 + 4].try_into()?);
            let expected = values[index] / (mean + 1e-6).sqrt();
            assert!(
                (actual - expected).abs() <= 2e-5,
                "RMSNorm {index}: {actual} != {expected}"
            );
        }
    }

    let mut x = Allocation::new(&vec![0x38; 16 * 256])?; // E4M3FN 1.0
    let mut w = Allocation::new(&vec![0x38; 128 * 256])?;
    let mut xs = Allocation::new(
        &(0..32)
            .flat_map(|_| 1.0f32.to_ne_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut ws = Allocation::new(
        &(0..2)
            .flat_map(|_| 1.0f32.to_ne_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut y = Allocation::new(&vec![0; 16 * 128 * 2])?;
    let gemm = Gemm::new(0, DType::Bf16, 16, 128, 256)?
        .prepare(providers.join("libaiter_ck_backend.so"))?;
    unsafe {
        gemm.enqueue(
            &mut [x.borrow(), w.borrow(), xs.borrow(), ws.borrow(), y.borrow()],
            std::ptr::null_mut(),
        )
    }?;
    hip(unsafe { hipDeviceSynchronize() })?;
    for word in y.read()?.chunks_exact(2) {
        assert_eq!(
            f32::from_bits(u32::from(u16::from_ne_bytes(word.try_into()?)) << 16),
            256.0
        );
    }
    println!("PASS Rust native SDK: 64 RMSNorm values, 2048 GEMM values, invalid-span rejection, repeated enqueue and explicit completion");
    Ok(())
}
