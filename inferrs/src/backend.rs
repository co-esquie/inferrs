//! Linux GPU backend discovery via `dlopen`.
//!
//! The `inferrs` binary is compiled CPU-only (no CUDA/ROCm linked at build
//! time).  At startup it searches for backend plugin `.so` files alongside the
//! running executable and probes each one in priority order.  Each plugin
//! exports a single C-ABI function:
//!
//! ```c
//! int inferrs_backend_probe(void);  // 0 = available, non-zero = not available
//! ```
//!
//! If a probe succeeds the matching `candle_core::Device` variant is returned.
//! The caller (`resolve_device`) uses this to construct the actual device.
//!
//! Plugin search order (highest priority first):
//!   1. CUDA   (`libinferrs_backend_cuda.so`)    → `Device::new_cuda(0)`
//!   2. ROCm   (`libinferrs_backend_rocm.so`)    → `Device::new_cuda(0)` (HIP)
//!   3. Vulkan (`libinferrs_backend_vulkan.so`)  → CPU fallback with warning
//!   4. CPU    (always available)

#[cfg(target_os = "linux")]
mod linux {
    use std::path::PathBuf;

    use libloading::{Library, Symbol};

    /// The detected GPU backend, in priority order.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BackendKind {
        Cuda,
        Rocm,
        /// Vulkan is detected but candle 0.8 has no Vulkan Device variant yet.
        /// Falls back to CPU while logging the detection.
        Vulkan,
        Cpu,
    }

    /// Probe the backend plugins and return the highest-priority available kind.
    ///
    /// The plugins are searched next to the running executable first, then in
    /// the same directory as the executable's parent (for dev builds under
    /// `target/<target>/release/`), and finally in `/usr/lib/inferrs/`.
    pub fn detect_backend() -> BackendKind {
        let search_dirs = plugin_search_dirs();

        // Priority order: CUDA → ROCm → Vulkan → CPU
        let candidates: &[(&str, BackendKind)] = &[
            ("libinferrs_backend_cuda.so", BackendKind::Cuda),
            ("libinferrs_backend_rocm.so", BackendKind::Rocm),
            ("libinferrs_backend_vulkan.so", BackendKind::Vulkan),
        ];

        for (lib_name, kind) in candidates {
            if probe_plugin(&search_dirs, lib_name) {
                return *kind;
            }
        }

        BackendKind::Cpu
    }

    /// Try to load `lib_name` from each search directory and call
    /// `inferrs_backend_probe()`.  Returns `true` if the probe returns 0.
    fn probe_plugin(search_dirs: &[PathBuf], lib_name: &str) -> bool {
        type ProbeFn = unsafe extern "C" fn() -> i32;

        for dir in search_dirs {
            let path = dir.join(lib_name);
            if !path.exists() {
                continue;
            }

            // SAFETY: We are loading a well-known plugin whose ABI we control.
            let lib = match unsafe { Library::new(&path) } {
                Ok(l) => l,
                Err(e) => {
                    tracing::debug!("Failed to load {}: {e}", path.display());
                    continue;
                }
            };

            // SAFETY: We know the exported symbol name and its signature.
            let probe: Symbol<ProbeFn> = match unsafe { lib.get(b"inferrs_backend_probe\0") } {
                Ok(sym) => sym,
                Err(e) => {
                    tracing::debug!("Symbol not found in {}: {e}", path.display());
                    continue;
                }
            };

            // SAFETY: Calling a C function with no arguments is safe.
            let result = unsafe { probe() };
            if result == 0 {
                tracing::debug!("Backend probe succeeded: {}", path.display());
                // Keep `lib` alive until probe result is used — it's dropped here
                // which is fine because we only needed the probe call.
                drop(lib);
                return true;
            }
            tracing::debug!(
                "Backend probe returned {result} (unavailable): {}",
                path.display()
            );
        }

        false
    }

    /// Directories to search for backend plugins, in priority order.
    fn plugin_search_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();

        // 1. Same directory as the running executable.
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                dirs.push(parent.to_path_buf());
            }
        }

        // 2. System-wide install location.
        dirs.push(PathBuf::from("/usr/lib/inferrs"));
        dirs.push(PathBuf::from("/usr/local/lib/inferrs"));

        dirs
    }
}

#[cfg(target_os = "linux")]
pub use linux::{detect_backend, BackendKind};

/// On non-Linux platforms (macOS, Windows) there is no plugin system —
/// Metal and CUDA are linked directly.
#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
pub fn detect_backend() -> BackendKind {
    BackendKind::Cpu
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
}
