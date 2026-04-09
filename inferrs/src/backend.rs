//! GPU/NPU backend discovery via dynamic loading (`dlopen` on Linux/Android,
//! `LoadLibraryW` on Windows).
//!
//! The `inferrs` binary is compiled with the `cuda` feature (so
//! `Device::new_cuda()` is available) but candle-core is patched to use
//! cudarc's `fallback-dynamic-loading` instead of `dynamic-linking`.  This
//! means CUDA/cuBLAS/cuRAND libraries are **not** hard-linked into the binary;
//! they are opened on demand when a CUDA device is first used.
//!
//! At startup the binary searches for backend plugin files alongside the
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
//! ## Platform support matrix
//!
//! | Platform                  | CUDA | MUSA | ROCm | CANN | Hexagon | Vulkan |
//! |---------------------------|------|------|------|------|---------|--------|
//! | Linux x86_64              | ✓    | ✓    | ✓    | ✓    | —       | ✓      |
//! | Linux aarch64             | ✓    | ✓    | ✓    | ✓    | ✓       | ✓      |
//! | Windows x86_64            | ✓    | ✓    | ✓    | —    | —       | ✓      |
//! | Windows aarch64           | —    | —    | —    | —    | ✓       | ✓      |
//! | macOS x86_64 / aarch64    | —    | —    | —    | —    | —       | ✓      |
//! | Android aarch64           | —    | —    | —    | ✓    | ✓       | ✓      |
//!
//! ROCm on Windows is supported from ROCm 5.5+ (HIP SDK for Windows).
//! ROCm on Linux aarch64 is supported on hardware such as AMD MI300A APUs
//! and Radeon-equipped AArch64 platforms.
//! CANN (Huawei Ascend NPU) is not supported on Windows (Huawei SDK constraint).
//! Hexagon (Qualcomm HTP NPU) is only present on Snapdragon SoCs (aarch64).
//!
//! Plugin search order (highest priority first):
//!
//! **Linux x86_64 / aarch64:**
//!   1. CUDA    (`.so`)  → `Device::new_cuda(0)`
//!   2. MUSA    (`.so`)  → `Device::new_cuda(0)` (Moore Threads)
//!   3. ROCm    (`.so`)  → `Device::new_cuda(0)` (HIP)
//!   4. CANN    (`.so`)  → CPU fallback with info log (pending candle CANN Device)
//!   5. Hexagon (`.so`)  → CPU fallback with info log (pending candle Hexagon Device)
//!   6. Vulkan  (`.so`)  → CPU fallback with info log
//!   7. CPU     (always available)
//!
//! **Windows x86_64:**
//!   1. CUDA    (`.dll`) → `Device::new_cuda(0)`
//!   2. MUSA    (`.dll`) → `Device::new_cuda(0)` (Moore Threads)
//!   3. ROCm    (`.dll`) → `Device::new_cuda(0)` (HIP SDK for Windows)
//!   4. Vulkan  (`.dll`) → CPU fallback with info log
//!   5. CPU     (always available)
//!
//! **Windows aarch64:**
//!   1. Hexagon (`.dll`) → CPU fallback with info log (pending candle Hexagon Device)
//!   2. Vulkan  (`.dll`) → CPU fallback with info log
//!   3. CPU     (always available)
//!
//! **macOS x86_64 / aarch64:**
//!   Metal is tried first (linked directly).  If Metal fails:
//!   1. Vulkan (`.dylib`, via MoltenVK) → CPU fallback with info log
//!   2. CPU    (always available)
//!
//! **Android aarch64:**
//!   1. CANN    (`.so`)  → CPU fallback with info log (pending candle CANN Device)
//!   2. Hexagon (`.so`)  → CPU fallback with info log (pending candle Hexagon Device)
//!   3. Vulkan  (`.so`)  → CPU fallback with info log
//!   4. CPU     (always available)

// ── Shared helpers ────────────────────────────────────────────────────────────
//
// `probe_plugin` and `exe_dir` are used by every platform module.
// They are compiled only on platforms that actually use the plugin system.

/// Try to load `lib_name` from each directory in `search_dirs` and call
/// `inferrs_backend_probe()`.  Returns `true` if the probe returns 0.
///
/// This single implementation is shared by every platform module; only the
/// search-directory list and the candidate file names differ per platform.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "windows",
))]
fn probe_plugin(search_dirs: &[std::path::PathBuf], lib_name: &str) -> bool {
    use libloading::{Library, Symbol};
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
            // `lib` is dropped here — that is fine because we only needed
            // the probe call; the plugin does not need to stay loaded.
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

/// Return the directory that contains the running executable, if available.
/// This is the highest-priority search location on every platform.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "windows",
))]
fn exe_dir() -> Option<std::path::PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

// ── Linux ─────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod linux {
    use std::path::PathBuf;

    /// The detected GPU/NPU backend, in priority order.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BackendKind {
        Cuda,
        /// Moore Threads GPU (MUSA SDK).  Uses the same `Device::new_cuda(0)`
        /// path in candle-core as NVIDIA CUDA, because the MUSA SDK mirrors
        /// the CUDA API.  The plugin probes for `libmusart.so` at runtime.
        Musa,
        Rocm,
        /// Huawei Ascend NPU via CANN (Compute Architecture for Neural Networks).
        ///
        /// candle-core does not yet have a native CANN `Device` variant.
        /// A successful probe causes an info-level log message; the binary
        /// then falls back to CPU.  Full acceleration will be enabled once
        /// candle integrates CANN support.
        ///
        /// Supported CANN SDK architectures: x86_64, aarch64.
        Cann,
        /// Qualcomm Hexagon HTP NPU.
        /// Falls back to CPU while candle gains a Hexagon device variant.
        /// Only present on aarch64 Snapdragon SoCs; the plugin won't exist
        /// on x86_64 so the probe skips it silently.
        Hexagon,
        /// Vulkan is detected but candle 0.8 has no Vulkan Device variant yet.
        /// Falls back to CPU while logging the detection.
        Vulkan,
        Cpu,
    }

    /// Probe the backend plugins and return the highest-priority available kind.
    pub fn detect_backend() -> BackendKind {
        let search_dirs = plugin_search_dirs();

        // Priority order: CUDA → MUSA → ROCm → CANN → Hexagon → Vulkan → CPU.
        // Both x86_64 and aarch64 Linux support CUDA, MUSA, and ROCm.
        //
        // MUSA (Moore Threads) mirrors the CUDA API; its plugin probes
        // `libmusart.so` without any compile-time dependency, so it is safe
        // to ship on systems without a MUSA driver installed.
        //
        // CANN (Huawei Ascend NPU) is placed after CUDA/MUSA/ROCm so that a
        // system with both a GPU and an Ascend NPU prefers the GPU.  CANN is
        // placed before Vulkan because it represents dedicated neural-network
        // silicon rather than a general graphics API.
        //
        // Hexagon (Qualcomm HTP) is placed after CANN for the same reason.
        // On x86_64 Linux the Hexagon plugin won't exist, so it is skipped.
        //
        // The CANN plugin is arch-gated at build time (x86_64 / aarch64 only)
        // so on unsupported architectures the `.so` simply won't exist.
        let candidates: &[(&str, BackendKind)] = &[
            ("libinferrs_backend_cuda.so", BackendKind::Cuda),
            ("libinferrs_backend_musa.so", BackendKind::Musa),
            ("libinferrs_backend_rocm.so", BackendKind::Rocm),
            ("libinferrs_backend_cann.so", BackendKind::Cann),
            ("libinferrs_backend_hexagon.so", BackendKind::Hexagon),
            ("libinferrs_backend_vulkan.so", BackendKind::Vulkan),
        ];

        for (lib_name, kind) in candidates {
            if super::probe_plugin(&search_dirs, lib_name) {
                return *kind;
            }
        }

        BackendKind::Cpu
    }

    fn plugin_search_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = super::exe_dir().into_iter().collect();
        dirs.push(PathBuf::from("/usr/lib/inferrs"));
        dirs.push(PathBuf::from("/usr/local/lib/inferrs"));
        dirs
    }
}

#[cfg(target_os = "linux")]
pub use linux::{detect_backend, BackendKind};

// ── Android ───────────────────────────────────────────────────────────────────
// Android aarch64 hosts both Huawei Ascend NPUs (CANN, in some edge devices)
// and Qualcomm Hexagon NPUs (Snapdragon SoCs).  CUDA and ROCm are unavailable.
// Vulkan is also available on most Android devices since API 24.

#[cfg(target_os = "android")]
mod android {
    use std::path::PathBuf;

    /// The detected NPU/GPU backend on Android.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BackendKind {
        /// Huawei Ascend NPU via CANN (aarch64 only).
        Cann,
        /// Qualcomm Hexagon HTP NPU (Snapdragon SoCs).
        Hexagon,
        /// Vulkan GPU acceleration (available since API 24).
        Vulkan,
        Cpu,
    }

    pub fn detect_backend() -> BackendKind {
        // Priority: CANN → Hexagon → Vulkan → CPU.
        // CANN is probed first; on Snapdragon devices the CANN plugin won't
        // exist so the probe falls through to Hexagon, then Vulkan.
        let candidates: &[(&str, BackendKind)] = &[
            ("libinferrs_backend_cann.so", BackendKind::Cann),
            ("libinferrs_backend_hexagon.so", BackendKind::Hexagon),
            ("libinferrs_backend_vulkan.so", BackendKind::Vulkan),
        ];

        let search_dirs = plugin_search_dirs();
        for (lib_name, kind) in candidates {
            if super::probe_plugin(&search_dirs, lib_name) {
                return *kind;
            }
        }

        BackendKind::Cpu
    }

    fn plugin_search_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = super::exe_dir().into_iter().collect();
        // Qualcomm vendor library path (standard on Snapdragon Android).
        dirs.push(PathBuf::from("/vendor/lib64/inferrs"));
        // Common Android data-app / on-device dev directories.
        dirs.push(PathBuf::from("/data/local/tmp/inferrs"));
        dirs
    }
}

#[cfg(target_os = "android")]
pub use android::{detect_backend, BackendKind};

// ── macOS ─────────────────────────────────────────────────────────────────────
//
// Metal is linked directly and is always preferred.  Vulkan is also available
// via MoltenVK (a Vulkan 1.3 portability layer over Metal).  We probe for it
// so that the main binary can log its availability and future wgpu/Vulkan code
// paths can use it.
//
// Architectures: x86_64-apple-darwin, aarch64-apple-darwin.

#[cfg(target_os = "macos")]
mod macos {
    use std::path::PathBuf;

    /// The detected GPU backend on macOS (Metal is handled directly in
    /// `auto_device` before the plugin system is invoked).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BackendKind {
        /// Vulkan via MoltenVK is detected; candle 0.8 has no Vulkan Device yet.
        Vulkan,
        Cpu,
    }

    pub fn detect_backend() -> BackendKind {
        if super::probe_plugin(&plugin_search_dirs(), "libinferrs_backend_vulkan.dylib") {
            return BackendKind::Vulkan;
        }
        BackendKind::Cpu
    }

    fn plugin_search_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = super::exe_dir().into_iter().collect();
        // Standard macOS library paths, including Homebrew locations where
        // MoltenVK is typically installed.
        dirs.push(PathBuf::from("/usr/local/lib"));
        dirs.push(PathBuf::from("/opt/homebrew/lib")); // Homebrew Apple Silicon
        dirs.push(PathBuf::from("/usr/local/opt/molten-vk/lib")); // Homebrew Intel
        dirs
    }
}

#[cfg(target_os = "macos")]
pub use macos::{detect_backend, BackendKind};

// ── Windows x86_64 ───────────────────────────────────────────────────────────
// CUDA and ROCm are available on Windows x86_64 only.  ROCm on Windows is
// supported from ROCm 5.5+ (HIP SDK for Windows).  Vulkan is available on
// both x86_64 and aarch64 (see the Windows aarch64 section below).
// CANN and Hexagon are not supported on Windows x86_64.

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
mod windows_x86_64 {
    use std::path::PathBuf;

    /// The detected GPU backend, in priority order.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BackendKind {
        Cuda,
        /// Moore Threads GPU (MUSA SDK).  Uses `Device::new_cuda(0)` because
        /// MUSA mirrors the CUDA API.  The plugin probes `musa.dll` at runtime.
        Musa,
        /// ROCm/HIP device (AMD GPU via ROCm 5.5+ HIP SDK for Windows).
        Rocm,
        /// Vulkan is detected but candle 0.8 has no Vulkan Device variant yet.
        /// Falls back to CPU while logging the detection.
        Vulkan,
        Cpu,
    }

    pub fn detect_backend() -> BackendKind {
        let search_dirs = plugin_search_dirs();

        // Priority order: CUDA → MUSA → ROCm → Vulkan → CPU.
        // ROCm on Windows x86_64 is supported via AMD's HIP SDK (ROCm 5.5+).
        // MUSA Windows support is announced by Moore Threads.
        // CANN is not supported on Windows (Huawei SDK constraint).
        // Hexagon does not exist on x86_64.
        let candidates: &[(&str, BackendKind)] = &[
            ("inferrs_backend_cuda.dll", BackendKind::Cuda),
            ("inferrs_backend_musa.dll", BackendKind::Musa),
            ("inferrs_backend_rocm.dll", BackendKind::Rocm),
            ("inferrs_backend_vulkan.dll", BackendKind::Vulkan),
        ];

        for (lib_name, kind) in candidates {
            if super::probe_plugin(&search_dirs, lib_name) {
                return *kind;
            }
        }

        BackendKind::Cpu
    }

    fn plugin_search_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = super::exe_dir().into_iter().collect();
        if let Ok(pf) = std::env::var("ProgramFiles") {
            dirs.push(PathBuf::from(pf).join("inferrs"));
        }
        dirs
    }
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub use windows_x86_64::{detect_backend, BackendKind};

// ── Windows aarch64 (Snapdragon X / 8cx) ─────────────────────────────────────
// Qualcomm ARM64 Windows devices ship a Hexagon NPU and Vulkan-capable GPU,
// but no CUDA/ROCm/CANN.
// Priority: Hexagon → Vulkan → CPU.

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
mod windows_aarch64 {
    use std::path::PathBuf;

    /// The detected NPU / GPU backend, in priority order.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BackendKind {
        /// Hexagon HTP NPU (Qualcomm).  Falls back to CPU while candle gains
        /// a Hexagon device variant.
        Hexagon,
        /// Vulkan GPU.  Falls back to CPU while candle 0.8 has no Vulkan Device.
        Vulkan,
        Cpu,
    }

    pub fn detect_backend() -> BackendKind {
        // Priority: Hexagon → Vulkan → CPU  (no CUDA/ROCm/CANN on ARM64 Windows)
        let candidates: &[(&str, BackendKind)] = &[
            ("inferrs_backend_hexagon.dll", BackendKind::Hexagon),
            ("inferrs_backend_vulkan.dll", BackendKind::Vulkan),
        ];

        let search_dirs = plugin_search_dirs();
        for (lib_name, kind) in candidates {
            if super::probe_plugin(&search_dirs, lib_name) {
                return *kind;
            }
        }

        BackendKind::Cpu
    }

    fn plugin_search_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = super::exe_dir().into_iter().collect();
        if let Ok(pf) = std::env::var("ProgramFiles") {
            dirs.push(PathBuf::from(pf).join("inferrs"));
        }
        dirs
    }
}

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
pub use windows_aarch64::{detect_backend, BackendKind};

// ── Any remaining platform (e.g. FreeBSD, bare-metal) ────────────────────────

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "aarch64"),
)))]
#[allow(dead_code)]
pub fn detect_backend() -> BackendKind {
    BackendKind::Cpu
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "aarch64"),
)))]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
}
