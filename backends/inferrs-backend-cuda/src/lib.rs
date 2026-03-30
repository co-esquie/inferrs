/// Probe whether a CUDA device is available and functional.
///
/// Returns 0 on success (CUDA device 0 is usable), non-zero on failure.
/// This function is `dlopen`'d by the main `inferrs` binary at runtime so
/// that the binary itself does not link against CUDA at compile time.
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    match candle_core::Device::new_cuda(0) {
        Ok(_) => 0,
        Err(_) => 1,
    }
}
