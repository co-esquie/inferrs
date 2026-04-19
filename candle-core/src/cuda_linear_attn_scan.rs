/// CUDA dispatch for the parallelized GatedDeltaNet chunked scan.
///
/// Kernels run sequentially on the same CUDA stream:
///   K1  linear_attn_intra   grid(B*NH*C) — KKT + fwd-subst + WY per chunk
///   K2a linear_attn_ops     grid(B*NH*C) — compute (A_i, b_i) per chunk
///   K2b linear_attn_scan    variable     — Blelloch prefix scan over chunks
///   K2c linear_attn_apply   grid(B*NH*C_padded) — reconstruct state, compute inter/vnew
///   K3  linear_attn_output  grid(B*NH*C) — tiled qk + matmul per chunk
///
/// Supports F32 and BF16 inputs for q/k/v.  log_g, beta, state are always F32.
/// Output tensors (out, new_state) are always F32.
///
/// All input tensors must be contiguous and shaped as `[B*NH, C, S, dim]`
/// (caller is responsible for reshaping before calling this function).
/// State is `[B*NH, HK, HV]`.
///
/// Returns `(out [B*NH, C, S, HV], new_state [B*NH, HK, HV])` — both F32.
use crate::{op::BackpropOp, DType, Result, Storage, Tensor};
use cudarc::driver::sys as csys;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn next_power_of_2(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

// ── CUDA Graph infrastructure ────────────────────────────────────────────────

struct OwnedGraphExec(csys::CUgraphExec);
// SAFETY: CUgraphExec is a CUDA handle tied to a fixed device context.
// We only ever access it while holding the GRAPH_CACHE mutex, serialising access.
unsafe impl Send for OwnedGraphExec {}

impl Drop for OwnedGraphExec {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { csys::cuGraphExecDestroy(self.0) };
        }
    }
}

/// Typed stable GPU buffer for q/k/v — one dtype variant per cache entry.
enum TypedBuf {
    F32(cudarc::driver::CudaSlice<f32>),
    Bf16(cudarc::driver::CudaSlice<half::bf16>),
}
// SAFETY: CudaSlice<T> is Send; same reasoning as OwnedGraphExec.
unsafe impl Send for TypedBuf {}

/// All GPU buffers for one (b_nh, c, hk, hv, dtype) shape, reused across calls.
///
/// # Hot-path design
///
/// Every call, per-call tensors (q/k/v/log_g/beta/state) are DtoD-copied into the
/// stable input bufs on the capture_stream.  The graph exec (captured once, cold
/// path only) is then re-launched with `cuGraphLaunch` — no re-capture, no
/// `cuGraphExecUpdate_v2`.  Outputs (out_buf / state_out_buf) are DtoD-copied to
/// fresh per-call tensors after the graph.
///
/// Hot-path cost per call: ~6 DtoD copies (inputs) + cuGraphLaunch + ~2 DtoD
/// copies (outputs) + 2 cross-stream joins ≈ 20-40 µs (vs ~80 µs for 16 direct
/// launches).
struct GraphEntry {
    // ── stable intermediate bufs ──────────────────────────────────────────────
    w_buf: cudarc::driver::CudaSlice<f32>,
    u_buf: cudarc::driver::CudaSlice<f32>,
    gc_buf: cudarc::driver::CudaSlice<f32>,
    inter_buf: cudarc::driver::CudaSlice<f32>,
    vnew_buf: cudarc::driver::CudaSlice<f32>,
    p_buf: cudarc::driver::CudaSlice<f32>,
    q_prefix_buf: cudarc::driver::CudaSlice<f32>,
    a_buf: cudarc::driver::CudaSlice<f32>,
    b_buf: cudarc::driver::CudaSlice<f32>,
    // ── stable input bufs (typed; DtoD copied from per-call tensors each call) ─
    q_buf: TypedBuf,
    k_buf: TypedBuf,
    v_buf: TypedBuf,
    lg_buf: cudarc::driver::CudaSlice<f32>,
    bt_buf: cudarc::driver::CudaSlice<f32>,
    state_0_buf: cudarc::driver::CudaSlice<f32>,
    // ── stable output bufs (graph writes here; DtoD copied to fresh tensors) ──
    out_buf: cudarc::driver::CudaSlice<f32>,
    state_out_buf: cudarc::driver::CudaSlice<f32>,
    // ── infra ────────────────────────────────────────────────────────────────
    /// Dedicated non-blocking stream.  The NULL / default stream cannot be
    /// passed to cuStreamBeginCapture_v2 (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
    capture_stream: std::sync::Arc<cudarc::driver::CudaStream>,
    /// Null until the first call (cold path); reused every call thereafter.
    exec: OwnedGraphExec,
}

/// Key: (b_nh, c, hk, hv, is_bf16).
type GraphKey = (usize, usize, usize, usize, bool);

static GRAPH_CACHE: OnceLock<Mutex<HashMap<GraphKey, GraphEntry>>> = OnceLock::new();

// ── Error helper ─────────────────────────────────────────────────────────────

fn cuda_err(r: csys::CUresult, ctx: &'static str) -> crate::Error {
    crate::Error::msg(format!("cuda_linear_attn_scan: {ctx} failed ({r:?})"))
}

// ── Main dispatch ─────────────────────────────────────────────────────────────

pub fn cuda_linear_attn_scan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    use candle_kernels as kernels;
    use cudarc::driver::PushKernelArg;

    let cuda_dev = match q.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("cuda_linear_attn_scan: requires CUDA device"),
    };

    let (b_nh, c, s, hk) = q.dims4()?;
    let hv = v.dim(3)?;

    if s != 64 {
        crate::bail!(
            "cuda_linear_attn_scan: chunk_size={s} != 64 (only S=64 is supported)"
        );
    }

    let is_bf16 = match q.dtype() {
        DType::F32 => false,
        DType::BF16 => true,
        dt => crate::bail!(
            "cuda_linear_attn_scan: unsupported dtype {dt:?} — only F32 or BF16"
        ),
    };
    let dtype_tag = if is_bf16 { "bf16" } else { "f32" };

    let (hk_tag, hv_tag) = match (hk, hv) {
        (64, 64) => ("64", "64"),
        (128, 128) => ("128", "128"),
        _ => crate::bail!(
            "cuda_linear_attn_scan: unsupported (hk={hk}, hv={hv}) — \
             only (64,64) and (128,128)"
        ),
    };

    let c_padded = next_power_of_2(c);
    let c_padded_i = c_padded as i32;
    let c_real_i = c as i32;

    let k1_name = format!("linear_attn_intra_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");
    let k2a_name = format!("linear_attn_ops_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");
    let k2b_up_name = format!("linear_attn_scan_up_hk{hk_tag}_hv{hv_tag}");
    let k2b_down_name = format!("linear_attn_scan_down_hk{hk_tag}_hv{hv_tag}");
    let k2b_clear_name = format!("linear_attn_scan_clear_root_hk{hk_tag}_hv{hv_tag}");
    let k2c_name = format!("linear_attn_apply_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");
    let k3_name = format!("linear_attn_output_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");

    let k1_smem = ((s * s + 2 * s + 2 * s * 64) * std::mem::size_of::<f32>()) as u32;
    let k2a_smem = ((32 * s + s * 32 + s) * std::mem::size_of::<f32>()) as u32;
    // Up-sweep needs an extra s_pr[BK*HK] buffer (BK=32) to cache P_right row-blocks
    // and avoid the in-place read-write conflict in the P composition tiling.
    let k2b_up_smem = ((2 * 32 * 32 + 32 * hk) * std::mem::size_of::<f32>()) as u32;
    let k2b_down_smem = (2 * 32 * 32 * std::mem::size_of::<f32>()) as u32;
    let k2c_smem = ((hk * 16 + 16 * hv + s + hk + 256) * std::mem::size_of::<f32>()) as u32;
    let k3_smem = ((s * s + 2 * s * 64 + s) * std::mem::size_of::<f32>()) as u32;

    let load_fn = |name: &str, smem: u32| -> Result<_> {
        let func = cuda_dev
            .get_or_load_func(name, &kernels::LINEAR_ATTN_SCAN)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        if smem > 48 * 1024 {
            func.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                96 * 1024,
            )
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        }
        Ok((func, smem))
    };
    let (f_k1, smem_k1) = load_fn(&k1_name, k1_smem)?;
    let (f_k2a, smem_k2a) = load_fn(&k2a_name, k2a_smem)?;
    let (f_k2b_up, smem_k2b_up) = load_fn(&k2b_up_name, k2b_up_smem)?;
    let (f_k2b_down, smem_k2b_down) = load_fn(&k2b_down_name, k2b_down_smem)?;
    let (f_k2b_clear, _smem_k2b_clear) = load_fn(&k2b_clear_name, 0)?;
    let (f_k2c, smem_k2c) = load_fn(&k2c_name, k2c_smem)?;
    let (f_k3, smem_k3) = load_fn(&k3_name, k3_smem)?;

    // ── Extract shared F32 input slices ────────────────────────────────────────
    let (lg_stor, lg_lay) = log_g.storage_and_layout();
    let (lg_o1, lg_o2) = lg_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("log_g not contiguous"))?;
    let lg_sl = match &*lg_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(lg_o1..lg_o2),
        _ => crate::bail!("expected Cuda storage for log_g"),
    };

    let (bt_stor, bt_lay) = beta.storage_and_layout();
    let (bt_o1, bt_o2) = bt_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("beta not contiguous"))?;
    let bt_sl = match &*bt_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(bt_o1..bt_o2),
        _ => crate::bail!("expected Cuda storage for beta"),
    };

    let (st_stor, st_lay) = state.storage_and_layout();
    let (st_o1, st_o2) = st_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("state not contiguous"))?;
    let st_src = match &*st_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(st_o1..st_o2),
        _ => crate::bail!("expected Cuda storage for state"),
    };

    // ── Ensure cache entry exists (allocate all stable bufs) ──────────────────
    let graph_key: GraphKey = (b_nh, c, hk, hv, is_bf16);
    let cache_mutex = GRAPH_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache_mutex.lock().unwrap();

    if !cache.contains_key(&graph_key) {
        let alloc_f32 = |n: usize| -> Result<cudarc::driver::CudaSlice<f32>> {
            unsafe {
                cuda_dev
                    .alloc::<f32>(n)
                    .map_err(|e| crate::Error::Cuda(Box::new(e)))
            }
        };
        let capture_stream = cuda_dev
            .new_stream()
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        let (q_buf, k_buf, v_buf) = if is_bf16 {
            let alloc_bf16 = |n: usize| -> Result<_> {
                unsafe {
                    cuda_dev
                        .alloc::<half::bf16>(n)
                        .map_err(|e| crate::Error::Cuda(Box::new(e)))
                }
            };
            (
                TypedBuf::Bf16(alloc_bf16(b_nh * c * s * hk)?),
                TypedBuf::Bf16(alloc_bf16(b_nh * c * s * hk)?),
                TypedBuf::Bf16(alloc_bf16(b_nh * c * s * hv)?),
            )
        } else {
            (
                TypedBuf::F32(alloc_f32(b_nh * c * s * hk)?),
                TypedBuf::F32(alloc_f32(b_nh * c * s * hk)?),
                TypedBuf::F32(alloc_f32(b_nh * c * s * hv)?),
            )
        };
        let entry = GraphEntry {
            w_buf: alloc_f32(b_nh * c * s * hk)?,
            u_buf: alloc_f32(b_nh * c * s * hv)?,
            gc_buf: alloc_f32(b_nh * c * s)?,
            inter_buf: alloc_f32(b_nh * c * s * hv)?,
            vnew_buf: alloc_f32(b_nh * c * s * hv)?,
            p_buf: alloc_f32(b_nh * c_padded * hk * hk)?,
            q_prefix_buf: alloc_f32(b_nh * c_padded * hk * hv)?,
            a_buf: alloc_f32(b_nh * c_padded * hk * hk)?,
            b_buf: alloc_f32(b_nh * c_padded * hk * hv)?,
            q_buf,
            k_buf,
            v_buf,
            lg_buf: alloc_f32(b_nh * c * s)?,
            bt_buf: alloc_f32(b_nh * c * s)?,
            state_0_buf: alloc_f32(b_nh * hk * hv)?,
            out_buf: alloc_f32(b_nh * c * s * hv)?,
            state_out_buf: alloc_f32(b_nh * hk * hv)?,
            capture_stream,
            exec: OwnedGraphExec(std::ptr::null_mut()),
        };
        cache.insert(graph_key, entry);
    }

    // ── Inner block: all entry access under the cache lock ────────────────────
    // Yields (out_fresh, state_fresh) — freshly allocated per-call output bufs.
    let (out_fresh, state_fresh) = {
        let entry = cache.get_mut(&graph_key).unwrap();
        let device_stream = cuda_dev.cuda_stream();
        let capture_cu_stream = entry.capture_stream.cu_stream();

        // ── Sync: capture_stream waits for device_stream ──────────────────────
        // Inputs (q/k/v/state) were produced on the device stream; the capture
        // stream must not start reading them before that work is done.
        entry
            .capture_stream
            .join(&device_stream)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

        // ── DtoD input copies → stable bufs (on capture_stream) ───────────────
        // These happen before any capture or graph launch, so the kernels always
        // see fresh data regardless of which path is taken below.
        match is_bf16 {
            false => {
                let (q_stor, q_lay) = q.storage_and_layout();
                let (q_o1, q_o2) = q_lay
                    .contiguous_offsets()
                    .ok_or_else(|| crate::Error::msg("q not contiguous"))?;
                let q_sl = match &*q_stor {
                    Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(q_o1..q_o2),
                    _ => crate::bail!("expected Cuda storage for q"),
                };
                let (k_stor, k_lay) = k.storage_and_layout();
                let (k_o1, k_o2) = k_lay
                    .contiguous_offsets()
                    .ok_or_else(|| crate::Error::msg("k not contiguous"))?;
                let k_sl = match &*k_stor {
                    Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(k_o1..k_o2),
                    _ => crate::bail!("expected Cuda storage for k"),
                };
                let (v_stor, v_lay) = v.storage_and_layout();
                let (v_o1, v_o2) = v_lay
                    .contiguous_offsets()
                    .ok_or_else(|| crate::Error::msg("v not contiguous"))?;
                let v_sl = match &*v_stor {
                    Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(v_o1..v_o2),
                    _ => crate::bail!("expected Cuda storage for v"),
                };
                let TypedBuf::F32(ref mut q_buf) = entry.q_buf else {
                    unreachable!("cache key encodes dtype")
                };
                let TypedBuf::F32(ref mut k_buf) = entry.k_buf else {
                    unreachable!("cache key encodes dtype")
                };
                let TypedBuf::F32(ref mut v_buf) = entry.v_buf else {
                    unreachable!("cache key encodes dtype")
                };
                entry
                    .capture_stream
                    .memcpy_dtod(&q_sl, q_buf)
                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                entry
                    .capture_stream
                    .memcpy_dtod(&k_sl, k_buf)
                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                entry
                    .capture_stream
                    .memcpy_dtod(&v_sl, v_buf)
                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                drop(q_stor);
                drop(k_stor);
                drop(v_stor);
            }
            true => {
                let (q_stor, q_lay) = q.storage_and_layout();
                let (q_o1, q_o2) = q_lay
                    .contiguous_offsets()
                    .ok_or_else(|| crate::Error::msg("q not contiguous"))?;
                let q_sl = match &*q_stor {
                    Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(q_o1..q_o2),
                    _ => crate::bail!("expected Cuda storage for q"),
                };
                let (k_stor, k_lay) = k.storage_and_layout();
                let (k_o1, k_o2) = k_lay
                    .contiguous_offsets()
                    .ok_or_else(|| crate::Error::msg("k not contiguous"))?;
                let k_sl = match &*k_stor {
                    Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(k_o1..k_o2),
                    _ => crate::bail!("expected Cuda storage for k"),
                };
                let (v_stor, v_lay) = v.storage_and_layout();
                let (v_o1, v_o2) = v_lay
                    .contiguous_offsets()
                    .ok_or_else(|| crate::Error::msg("v not contiguous"))?;
                let v_sl = match &*v_stor {
                    Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(v_o1..v_o2),
                    _ => crate::bail!("expected Cuda storage for v"),
                };
                let TypedBuf::Bf16(ref mut q_buf) = entry.q_buf else {
                    unreachable!("cache key encodes dtype")
                };
                let TypedBuf::Bf16(ref mut k_buf) = entry.k_buf else {
                    unreachable!("cache key encodes dtype")
                };
                let TypedBuf::Bf16(ref mut v_buf) = entry.v_buf else {
                    unreachable!("cache key encodes dtype")
                };
                entry
                    .capture_stream
                    .memcpy_dtod(&q_sl, q_buf)
                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                entry
                    .capture_stream
                    .memcpy_dtod(&k_sl, k_buf)
                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                entry
                    .capture_stream
                    .memcpy_dtod(&v_sl, v_buf)
                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                drop(q_stor);
                drop(k_stor);
                drop(v_stor);
            }
        }
        entry
            .capture_stream
            .memcpy_dtod(&lg_sl, &mut entry.lg_buf)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        entry
            .capture_stream
            .memcpy_dtod(&bt_sl, &mut entry.bt_buf)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        entry
            .capture_stream
            .memcpy_dtod(&st_src, &mut entry.state_0_buf)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

        // ── Cold path: capture all 16 kernels once, then instantiate ──────────
        // Hot path: exec is non-null → skip directly to cuGraphLaunch below.
        if entry.exec.0.is_null() {
            let capture_result = unsafe {
                csys::cuStreamBeginCapture_v2(
                    capture_cu_stream,
                    csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
            };
            if capture_result != csys::CUresult::CUDA_SUCCESS {
                return Err(cuda_err(capture_result, "cuStreamBeginCapture_v2"));
            }

            // Kernel launches — recorded into the graph.
            // All args reference stable entry.* buffers so the captured exec
            // can be replayed on every subsequent call without modification.
            let launch_result = (|| -> Result<()> {
                match is_bf16 {
                    false => {
                        let TypedBuf::F32(ref q_buf) = entry.q_buf else {
                            unreachable!()
                        };
                        let TypedBuf::F32(ref k_buf) = entry.k_buf else {
                            unreachable!()
                        };
                        let TypedBuf::F32(ref v_buf) = entry.v_buf else {
                            unreachable!()
                        };

                        // K1
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: ((b_nh * c) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: smem_k1,
                            };
                            let mut b = f_k1.builder_on_stream(&entry.capture_stream);
                            b.arg(q_buf);
                            b.arg(k_buf);
                            b.arg(v_buf);
                            b.arg(&entry.lg_buf);
                            b.arg(&entry.bt_buf);
                            b.arg(&mut entry.w_buf);
                            b.arg(&mut entry.u_buf);
                            b.arg(&mut entry.gc_buf);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }

                        // K2a
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: ((b_nh * c) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: smem_k2a,
                            };
                            let mut b = f_k2a.builder_on_stream(&entry.capture_stream);
                            b.arg(&mut entry.w_buf);
                            b.arg(&mut entry.u_buf);
                            b.arg(&mut entry.gc_buf);
                            b.arg(k_buf);
                            b.arg(&mut entry.p_buf);
                            b.arg(&mut entry.q_prefix_buf);
                            b.arg(&c_real_i);
                            b.arg(&c_padded_i);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }

                        // K2b up-sweep
                        {
                            let mut stride = 2usize;
                            while stride <= c_padded {
                                let n_pairs = b_nh * (c_padded / stride);
                                let cfg = cudarc::driver::LaunchConfig {
                                    grid_dim: (n_pairs as u32, 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: smem_k2b_up,
                                };
                                let mut b = f_k2b_up.builder_on_stream(&entry.capture_stream);
                                b.arg(&mut entry.p_buf);
                                b.arg(&mut entry.q_prefix_buf);
                                b.arg(&mut entry.a_buf);
                                b.arg(&mut entry.b_buf);
                                let stride_i = stride as i32;
                                b.arg(&stride_i);
                                b.arg(&c_padded_i);
                                unsafe { b.launch(cfg) }
                                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                                stride <<= 1;
                            }
                        }

                        // K2b clear root
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: (b_nh as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut b = f_k2b_clear.builder_on_stream(&entry.capture_stream);
                            b.arg(&mut entry.p_buf);
                            b.arg(&mut entry.q_prefix_buf);
                            b.arg(&c_real_i);
                            b.arg(&c_padded_i);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }

                        // K2b down-sweep
                        {
                            let mut stride = c_padded;
                            while stride >= 2 {
                                let n_pairs = b_nh * (c_padded / stride);
                                let cfg = cudarc::driver::LaunchConfig {
                                    grid_dim: (n_pairs as u32, 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: smem_k2b_down,
                                };
                                let mut b = f_k2b_down.builder_on_stream(&entry.capture_stream);
                                b.arg(&mut entry.p_buf);
                                b.arg(&mut entry.q_prefix_buf);
                                b.arg(&entry.a_buf);
                                b.arg(&entry.b_buf);
                                let stride_i = stride as i32;
                                b.arg(&stride_i);
                                b.arg(&c_padded_i);
                                unsafe { b.launch(cfg) }
                                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                                stride >>= 1;
                            }
                        }

                        // K2c
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: ((b_nh * c_padded) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: smem_k2c,
                            };
                            let mut b = f_k2c.builder_on_stream(&entry.capture_stream);
                            b.arg(&mut entry.w_buf);
                            b.arg(&mut entry.u_buf);
                            b.arg(&mut entry.gc_buf);
                            b.arg(q_buf);
                            b.arg(k_buf);
                            b.arg(&entry.state_0_buf);
                            b.arg(&mut entry.state_out_buf);
                            b.arg(&entry.p_buf);
                            b.arg(&entry.q_prefix_buf);
                            b.arg(&mut entry.inter_buf);
                            b.arg(&mut entry.vnew_buf);
                            b.arg(&c_real_i);
                            b.arg(&c_padded_i);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }

                        // K3
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: ((b_nh * c) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: smem_k3,
                            };
                            let mut b = f_k3.builder_on_stream(&entry.capture_stream);
                            b.arg(q_buf);
                            b.arg(k_buf);
                            b.arg(&mut entry.vnew_buf);
                            b.arg(&mut entry.inter_buf);
                            b.arg(&mut entry.gc_buf);
                            b.arg(&entry.out_buf);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }
                    }

                    true => {
                        let TypedBuf::Bf16(ref q_buf) = entry.q_buf else {
                            unreachable!()
                        };
                        let TypedBuf::Bf16(ref k_buf) = entry.k_buf else {
                            unreachable!()
                        };
                        let TypedBuf::Bf16(ref v_buf) = entry.v_buf else {
                            unreachable!()
                        };

                        // K1
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: ((b_nh * c) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: smem_k1,
                            };
                            let mut b = f_k1.builder_on_stream(&entry.capture_stream);
                            b.arg(q_buf);
                            b.arg(k_buf);
                            b.arg(v_buf);
                            b.arg(&entry.lg_buf);
                            b.arg(&entry.bt_buf);
                            b.arg(&mut entry.w_buf);
                            b.arg(&mut entry.u_buf);
                            b.arg(&mut entry.gc_buf);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }

                        // K2a
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: ((b_nh * c) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: smem_k2a,
                            };
                            let mut b = f_k2a.builder_on_stream(&entry.capture_stream);
                            b.arg(&mut entry.w_buf);
                            b.arg(&mut entry.u_buf);
                            b.arg(&mut entry.gc_buf);
                            b.arg(k_buf);
                            b.arg(&mut entry.p_buf);
                            b.arg(&mut entry.q_prefix_buf);
                            b.arg(&c_real_i);
                            b.arg(&c_padded_i);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }

                        // K2b up-sweep
                        {
                            let mut stride = 2usize;
                            while stride <= c_padded {
                                let n_pairs = b_nh * (c_padded / stride);
                                let cfg = cudarc::driver::LaunchConfig {
                                    grid_dim: (n_pairs as u32, 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: smem_k2b_up,
                                };
                                let mut b = f_k2b_up.builder_on_stream(&entry.capture_stream);
                                b.arg(&mut entry.p_buf);
                                b.arg(&mut entry.q_prefix_buf);
                                b.arg(&mut entry.a_buf);
                                b.arg(&mut entry.b_buf);
                                let stride_i = stride as i32;
                                b.arg(&stride_i);
                                b.arg(&c_padded_i);
                                unsafe { b.launch(cfg) }
                                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                                stride <<= 1;
                            }
                        }

                        // K2b clear root
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: (b_nh as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut b = f_k2b_clear.builder_on_stream(&entry.capture_stream);
                            b.arg(&mut entry.p_buf);
                            b.arg(&mut entry.q_prefix_buf);
                            b.arg(&c_real_i);
                            b.arg(&c_padded_i);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }

                        // K2b down-sweep
                        {
                            let mut stride = c_padded;
                            while stride >= 2 {
                                let n_pairs = b_nh * (c_padded / stride);
                                let cfg = cudarc::driver::LaunchConfig {
                                    grid_dim: (n_pairs as u32, 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: smem_k2b_down,
                                };
                                let mut b = f_k2b_down.builder_on_stream(&entry.capture_stream);
                                b.arg(&mut entry.p_buf);
                                b.arg(&mut entry.q_prefix_buf);
                                b.arg(&entry.a_buf);
                                b.arg(&entry.b_buf);
                                let stride_i = stride as i32;
                                b.arg(&stride_i);
                                b.arg(&c_padded_i);
                                unsafe { b.launch(cfg) }
                                    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                                stride >>= 1;
                            }
                        }

                        // K2c
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: ((b_nh * c_padded) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: smem_k2c,
                            };
                            let mut b = f_k2c.builder_on_stream(&entry.capture_stream);
                            b.arg(&mut entry.w_buf);
                            b.arg(&mut entry.u_buf);
                            b.arg(&mut entry.gc_buf);
                            b.arg(q_buf);
                            b.arg(k_buf);
                            b.arg(&entry.state_0_buf);
                            b.arg(&mut entry.state_out_buf);
                            b.arg(&entry.p_buf);
                            b.arg(&entry.q_prefix_buf);
                            b.arg(&mut entry.inter_buf);
                            b.arg(&mut entry.vnew_buf);
                            b.arg(&c_real_i);
                            b.arg(&c_padded_i);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }

                        // K3
                        {
                            let cfg = cudarc::driver::LaunchConfig {
                                grid_dim: ((b_nh * c) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: smem_k3,
                            };
                            let mut b = f_k3.builder_on_stream(&entry.capture_stream);
                            b.arg(q_buf);
                            b.arg(k_buf);
                            b.arg(&mut entry.vnew_buf);
                            b.arg(&mut entry.inter_buf);
                            b.arg(&mut entry.gc_buf);
                            b.arg(&entry.out_buf);
                            unsafe { b.launch(cfg) }
                                .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                        }
                    }
                }
                Ok(())
            })();

            // End capture — must happen even if kernel launches failed.
            let mut new_graph: csys::CUgraph = std::ptr::null_mut();
            let end_result =
                unsafe { csys::cuStreamEndCapture(capture_cu_stream, &mut new_graph) };

            launch_result?;

            if end_result != csys::CUresult::CUDA_SUCCESS {
                if !new_graph.is_null() {
                    unsafe { csys::cuGraphDestroy(new_graph) };
                }
                return Err(cuda_err(end_result, "cuStreamEndCapture"));
            }

            let r = unsafe {
                csys::cuGraphInstantiateWithFlags(&mut entry.exec.0, new_graph, 0)
            };
            unsafe { csys::cuGraphDestroy(new_graph) };
            if r != csys::CUresult::CUDA_SUCCESS {
                entry.exec.0 = std::ptr::null_mut();
                return Err(cuda_err(r, "cuGraphInstantiateWithFlags"));
            }
        }

        // ── Launch cached graph exec (hot + cold path) ────────────────────────
        let r = unsafe { csys::cuGraphLaunch(entry.exec.0, capture_cu_stream) };
        if r != csys::CUresult::CUDA_SUCCESS {
            return Err(cuda_err(r, "cuGraphLaunch"));
        }

        // ── DtoD output copies → fresh per-call tensors (on capture_stream) ───
        // Queued after cuGraphLaunch on the same stream, so stream ordering
        // guarantees the graph writes are visible before the copies start.
        let mut out_fresh = unsafe {
            cuda_dev
                .alloc::<f32>(b_nh * c * s * hv)
                .map_err(|e| crate::Error::Cuda(Box::new(e)))?
        };
        let mut state_fresh = unsafe {
            cuda_dev
                .alloc::<f32>(b_nh * hk * hv)
                .map_err(|e| crate::Error::Cuda(Box::new(e)))?
        };
        entry
            .capture_stream
            .memcpy_dtod(&entry.out_buf, &mut out_fresh)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        entry
            .capture_stream
            .memcpy_dtod(&entry.state_out_buf, &mut state_fresh)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

        // ── Sync: device stream waits for capture_stream ──────────────────────
        // All subsequent candle ops (on the device stream) that read out_fresh /
        // state_fresh must see the completed graph + output copies.
        device_stream
            .join(&entry.capture_stream)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

        (out_fresh, state_fresh)
    };
    // entry and cache dropped here, releasing the mutex.

    drop(lg_stor);
    drop(bt_stor);
    drop(st_stor);

    // ── Wrap fresh output bufs as tensors ─────────────────────────────────────
    let out_tensor = {
        let cs = crate::CudaStorage::wrap_cuda_slice(out_fresh, cuda_dev.clone());
        let shape = crate::Shape::from_dims(&[b_nh, c, s, hv]);
        Tensor::from_storage(Storage::Cuda(cs), shape, BackpropOp::none(), false)
    };

    let state_tensor = {
        let cs = crate::CudaStorage::wrap_cuda_slice(state_fresh, cuda_dev);
        let shape = crate::Shape::from_dims(&[b_nh, hk, hv]);
        Tensor::from_storage(Storage::Cuda(cs), shape, BackpropOp::none(), false)
    };

    Ok((out_tensor, state_tensor))
}
