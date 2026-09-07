// SPDX-License-Identifier: MIT
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=AITER_SDK_DIR");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "linux",
        "AITER native ABI 1 requires Linux"
    );
    let prefix = PathBuf::from(
        env::var_os("AITER_SDK_DIR").expect("Set AITER_SDK_DIR to a CMake-installed AITER SDK"),
    );
    let prefix = prefix.canonicalize().expect("AITER_SDK_DIR does not exist");
    assert!(
        prefix.join("include/aiter/aiter.h").is_file(),
        "AITER SDK header is missing"
    );
    let library = [prefix.join("lib"), prefix.join("lib64")]
        .into_iter()
        .find(|path| path.join("libaiter.so").is_file())
        .expect("AITER SDK libaiter.so is missing");
    println!(
        "cargo:rerun-if-changed={}",
        prefix.join("include/aiter/aiter.h").display()
    );
    println!("cargo:rustc-link-search=native={}", library.display());
    println!("cargo:rustc-link-lib=dylib=aiter");
    let rocm = PathBuf::from(env::var_os("ROCM_PATH").unwrap_or_else(|| "/opt/rocm".into()));
    println!(
        "cargo:rustc-link-search=native={}",
        rocm.join("lib").display()
    );
}
