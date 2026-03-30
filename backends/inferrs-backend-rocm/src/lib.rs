/// Probe whether a ROCm (AMD GPU) device is available and functional.
///
/// candle-core's `cuda` feature covers both NVIDIA CUDA and AMD ROCm/HIP —
/// the same `Device::new_cuda(0)` call works when the library was compiled
/// with the ROCm toolchain (`hipcc`, `ROCM_PATH` set at build time).
///
/// Returns 0 on success, non-zero on failure.
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    match candle_core::Device::new_cuda(0) {
        Ok(_) => 0,
        Err(_) => 1,
    }
}
