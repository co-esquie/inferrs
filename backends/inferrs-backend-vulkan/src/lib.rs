/// Probe whether a Vulkan-capable driver is available on this system.
///
/// This is implemented by attempting to open the Vulkan loader library at
/// runtime using `dlopen` (POSIX) or `LoadLibraryW` (Windows).  The backend
/// shared library itself does **not** link against Vulkan at compile time, so
/// loading this plugin on a system without Vulkan will not fail — only the
/// probe call will return non-zero.
///
/// NOTE: candle-core 0.8 does not yet have a Vulkan/wgpu `Device` variant.
/// The main `inferrs` binary uses a successful probe to log that Vulkan is
/// available and will accelerate inference once candle gains wgpu support.
/// Until then the binary falls back to CPU after logging the detection.
///
/// Platform library names tried, in order:
///
/// | Platform              | Library names                                         |
/// |-----------------------|-------------------------------------------------------|
/// | Linux  (any arch)     | `libvulkan.so.1`, `libvulkan.so`                      |
/// | Android (any arch)    | `libvulkan.so`  (OS-provided since API 24)            |
/// | macOS  (any arch)     | `libvulkan.1.dylib`, `libvulkan.dylib`,               |
/// |                       | `libMoltenVK.dylib`  (MoltenVK portability layer)     |
/// | Windows (x86_64/ARM)  | `vulkan-1.dll`  (via `LoadLibraryW`)                  |
///
/// Returns 0 if any candidate library can be opened, 1 otherwise.
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    if try_load_vulkan() {
        0
    } else {
        1
    }
}

// ── Linux ─────────────────────────────────────────────────────────────────────
//
// Covers x86_64 and aarch64 (and any other Linux architecture).
// Android is a separate target_os so it does not fall into this branch.

#[cfg(target_os = "linux")]
fn try_load_vulkan() -> bool {
    dlopen_any(&["libvulkan.so.1", "libvulkan.so"])
}

// ── Android ───────────────────────────────────────────────────────────────────
//
// On Android the Vulkan loader is a system library named `libvulkan.so`
// (no version suffix) baked into the OS since API level 24 (Android 7.0).
// The NDK exposes it at link time, but we probe at runtime via dlopen to
// match the pattern used for every other backend.
//
// Architectures: aarch64-linux-android, x86_64-linux-android,
//                armv7-linux-androideabi, i686-linux-android.

#[cfg(target_os = "android")]
fn try_load_vulkan() -> bool {
    dlopen_any(&["libvulkan.so"])
}

// ── macOS ─────────────────────────────────────────────────────────────────────
//
// On macOS, Vulkan is available through:
//   1. A full Vulkan SDK installation → `libvulkan.1.dylib` / `libvulkan.dylib`
//   2. MoltenVK installed stand-alone or via Homebrew → `libMoltenVK.dylib`
//
// MoltenVK implements Vulkan 1.3 on top of Metal.  Homebrew installs it at
// `/opt/homebrew/lib/libMoltenVK.dylib` (Apple Silicon) or
// `/usr/local/lib/libMoltenVK.dylib` (Intel), which are on the default
// `DYLD_LIBRARY_PATH` for Homebrew-aware shells.
//
// The portability-enumeration extension (VK_KHR_portability_enumeration) is
// required for MoltenVK devices to appear during instance enumeration; the
// actual inference runtime must pass VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
// when creating its VkInstance (as llama.cpp does in ggml-vulkan.cpp).
//
// Architectures: x86_64-apple-darwin, aarch64-apple-darwin.

#[cfg(target_os = "macos")]
fn try_load_vulkan() -> bool {
    dlopen_any(&["libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"])
}

// ── Windows ───────────────────────────────────────────────────────────────────
//
// On Windows the Vulkan loader DLL is always named `vulkan-1.dll` for both
// x86_64 and aarch64.  We use `LoadLibraryW` (wide-string variant) to avoid
// ANSI code-page issues.  The Windows loader searches:
//   1. The directory containing the application executable
//   2. The current working directory
//   3. System32 (where GPU driver installers place `vulkan-1.dll`)
//   4. Entries in the PATH environment variable
//
// Architectures: x86_64-pc-windows-msvc, aarch64-pc-windows-msvc,
//                x86_64-pc-windows-gnu, aarch64-pc-windows-gnu.

#[cfg(target_os = "windows")]
fn try_load_vulkan() -> bool {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    // Encode the DLL name as a null-terminated UTF-16 string.
    let name_wide: Vec<u16> = OsStr::new("vulkan-1.dll")
        .encode_wide()
        .chain(std::iter::once(0u16))
        .collect();

    // SAFETY: `LoadLibraryW` is safe to call with a valid null-terminated
    // UTF-16 string.  We immediately free the handle if loading succeeds.
    // HMODULE is `*mut c_void` in windows-sys 0.59; null check via
    // `!= std::ptr::null_mut()`.  FreeLibrary lives in Win32::Foundation.
    unsafe {
        let handle = windows_sys::Win32::System::LibraryLoader::LoadLibraryW(name_wide.as_ptr());
        if !handle.is_null() {
            windows_sys::Win32::Foundation::FreeLibrary(handle);
            return true;
        }
    }
    false
}

// ── Fallback (unsupported platforms) ─────────────────────────────────────────

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "windows",
)))]
fn try_load_vulkan() -> bool {
    false
}

// ── Shared POSIX helper ───────────────────────────────────────────────────────

/// Try each library name in `names` via `dlopen(RTLD_LAZY | RTLD_LOCAL)`.
/// Returns `true` and immediately closes the handle if any candidate succeeds.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
fn dlopen_any(names: &[&str]) -> bool {
    use std::ffi::CString;

    for name in names {
        let Ok(cname) = CString::new(*name) else {
            continue;
        };
        // SAFETY: `dlopen` is safe to call with a valid C string and flags.
        // We immediately `dlclose` the handle — we only need to know whether
        // the library is present and loadable.
        let handle = unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
        if !handle.is_null() {
            unsafe { libc::dlclose(handle) };
            return true;
        }
    }
    false
}
