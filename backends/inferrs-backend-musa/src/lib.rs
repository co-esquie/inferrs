//! Probe whether a Moore Threads GPU (MUSA) device is available and
//! functional on this system.
//!
//! # Design
//!
//! MUSA is the GPU compute platform from Moore Threads Technology.  It
//! presents an API surface intentionally compatible with NVIDIA CUDA; the SDK
//! ships a runtime library (`libmusart.so` on Linux, `musa.dll` on Windows)
//! that exposes a `musaGetDeviceCount` symbol analogous to `cudaGetDeviceCount`.
//!
//! This plugin does **not** link against the MUSA SDK at compile time.
//! Instead it opens the MUSA runtime library at probe time via `libloading`,
//! which wraps `dlopen` on Linux and `LoadLibraryW` on Windows.  This means:
//!
//! * Building the plugin requires only the Rust toolchain — no MUSA SDK.
//! * Shipping the plugin in release archives is safe on any host; a missing
//!   MUSA driver simply causes the probe to return non-zero.
//!
//! The probe sequence is:
//! 1. Try each candidate library name in priority order (versioned first).
//! 2. Resolve `musaGetDeviceCount` from the opened library.
//! 3. Call it; return 0 (available) only if the call succeeds and returns
//!    at least one device.
//!
//! # Supported platforms
//!
//! Moore Threads currently ships the MUSA SDK for **Linux x86_64** and
//! **Linux aarch64**.  Windows x86_64 support is included because Moore
//! Threads has announced Windows drivers for the MTT S80 / S4000 / S5000
//! series.
//!
//! Android and macOS are not supported by Moore Threads hardware.
//! Windows aarch64 has no known MUSA SDK; the probe returns 1 there.

/// Probe whether a MUSA device (Moore Threads GPU) is available.
///
/// Returns 0 if at least one MUSA device is usable, non-zero otherwise.
///
/// The `inferrs` binary `dlopen`s this plugin at runtime so the binary itself
/// carries no compile-time dependency on the MUSA SDK.
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    // Linux x86_64 and aarch64: primary targets shipping the MUSA SDK.
    // Windows x86_64: announced support from Moore Threads.
    #[cfg(any(
        target_os = "linux",
        all(target_os = "windows", target_arch = "x86_64")
    ))]
    {
        probe_musa()
    }

    // macOS, Windows aarch64, Android, and all other platforms: not supported.
    #[cfg(not(any(
        target_os = "linux",
        all(target_os = "windows", target_arch = "x86_64")
    )))]
    {
        1
    }
}

/// Try candidate MUSA library names in order; for the first one that can be
/// opened, attempt to resolve and call `musaGetDeviceCount`.
#[cfg(any(
    target_os = "linux",
    all(target_os = "windows", target_arch = "x86_64")
))]
fn probe_musa() -> i32 {
    // Candidate names searched in priority order.
    //
    // Linux:   versioned sonames are preferred (the linker chose them for the
    //          installed ABI); unversioned names are fallbacks for custom installs
    //          that lack symlinks.
    //
    //   libmusart.so — MUSA **runtime** library; exposes musaGetDeviceCount
    //                  directly without requiring a separate musaInit() call.
    //   libmusa.so   — MUSA **driver** API library (lower-level).
    //
    // Windows: Moore Threads DLLs follow the <name>.dll / <name>64.dll pattern
    //          used by CUDA (nvcuda.dll → musa.dll, cudart64_*.dll → musa64.dll).
    #[cfg(target_os = "linux")]
    const CANDIDATES: &[&str] = &[
        "libmusart.so.1",
        "libmusart.so",
        "libmusa.so.1",
        "libmusa.so",
    ];

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    const CANDIDATES: &[&str] = &["musa.dll", "musart.dll", "musa64.dll"];

    type MusaGetDeviceCountFn = unsafe extern "C" fn(*mut i32) -> i32;

    for lib_name in CANDIDATES {
        // SAFETY: We are attempting to open a well-known system library by
        // name.  libloading uses RTLD_LOCAL | RTLD_NOW on Linux and
        // LoadLibraryW on Windows.
        let lib = match unsafe { libloading::Library::new(lib_name) } {
            Ok(l) => l,
            Err(_) => continue,
        };

        // SAFETY: We know the symbol name and its C ABI signature from the
        // MUSA SDK.  The library is kept alive for the duration of the call.
        let get_device_count: libloading::Symbol<MusaGetDeviceCountFn> =
            match unsafe { lib.get(b"musaGetDeviceCount\0") } {
                Ok(sym) => sym,
                Err(_) => continue,
            };

        let mut count: i32 = 0;
        // SAFETY: count is valid stack memory; musaGetDeviceCount writes
        // exactly one i32 to the pointer.
        let err = unsafe { get_device_count(&mut count) };

        // musaSuccess == 0.  Only commit to a result when the call itself
        // succeeded; a non-zero error code (driver mismatch, init failure,
        // etc.) means this library is unusable — try the next candidate.
        if err == 0 {
            return if count > 0 { 0 } else { 1 };
        }
    }

    // No MUSA library found on this system.
    1
}
