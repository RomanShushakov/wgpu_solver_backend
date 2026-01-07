use clap::{Parser, Subcommand};
use futures::executor;
use serde::Serialize;
use serde_json::to_string_pretty;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::process;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use wgpu::{BufferUsages, CommandEncoderDescriptor};
use wgpu_solver_backend::compute::block_jacobi_exec::BlockJacobiExecutor;
use wgpu_solver_backend::compute::dot_scalar_exec::DotScalarExecutor;
use wgpu_solver_backend::compute::pcg_update_scalars_exec::PcgUpdateScalarsExecutor;
use wgpu_solver_backend::compute::spmv_exec::SpmvExecutor;
use wgpu_solver_backend::compute::vec_ops_exec::VecOpsExecutor;
use wgpu_solver_backend::compute::{
    build_lu_blocks_from_csr_block_starts_6, pcg_block_jacobi_csr_wgpu,
};
use wgpu_solver_backend::gpu::context::{GpuBackend, GpuContext};
use wgpu_solver_backend::gpu::readback::readback_to_vec;
use wgpu_solver_backend::io::loaders::load_case_dir;

#[derive(Parser, Debug)]
#[command(
    name = "wgpu-solver-backend",
    version,
    about = "Compute-first wgpu backend for iterative solvers"
)]
struct Cli {
    #[arg(long, default_value = "auto")]
    backend: String,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Print GPU adapter info and emit a metrics JSON blob (stdout).
    Info,
    /// Sanity test for vec ops (AXPY): y = y + alpha * x
    VecTest,
    DotTest,
    SpmvTest,
    BlockJacobiTest,
    PcgUpdateScalarsTest,
    /// Run PCG(Block-Jacobi) on a case directory (matrix.csr.bin, rhs.bin, x0.bin, block_starts.bin)
    RunPcgCase {
        /// Case directory containing matrix.csr.bin, rhs.bin, x0.bin, block_starts.bin
        #[arg(long)]
        case_dir: String,

        /// Max PCG iterations
        #[arg(long, default_value_t = 2000)]
        max_iters: usize,

        /// Relative tolerance
        #[arg(long, default_value_t = 1e-8)]
        rel_tol: f32,

        /// Absolute tolerance
        #[arg(long, default_value_t = 0.0)]
        abs_tol: f32,

        /// Where to write x.bin
        #[arg(long)]
        out_x: String,

        /// Where to write metrics.json
        #[arg(long)]
        out_metrics: String,
    },
    /// Compare two solution vectors stored in .bin format (u32 len + f32[len])
    CompareX {
        /// Path to reference x_ref.bin
        #[arg(long)]
        x_ref: String,

        /// Path to produced x.bin (backend output)
        #[arg(long)]
        x: String,

        /// Relative tolerance
        #[arg(long, default_value_t = 1e-3)]
        rel_tol: f32,

        /// Absolute tolerance
        #[arg(long, default_value_t = 1e-4)]
        abs_tol: f32,

        /// Show top-k worst indices
        #[arg(long, default_value_t = 10)]
        top_k: usize,
    },
}

#[derive(Serialize)]
struct Metrics {
    run_id: String,
    command: String,
    gpu: GpuMetrics,
    build: BuildMetrics,
}

#[derive(Serialize)]
struct GpuMetrics {
    adapter_name: String,
    backend: String,
    device_type: String,
    vendor: u32,
    device: u32,
}

#[derive(Serialize)]
struct BuildMetrics {
    crate_version: String,
    git_rev: Option<String>,
}

#[derive(Serialize)]
struct SolveMetrics {
    run_id: String,
    command: String,

    case_dir: String,
    n: u32,
    nnz: u32,
    max_iters: usize,
    rel_tol: f32,
    abs_tol: f32,

    iterations: Option<usize>,
    converged: bool,
    error: Option<String>,

    timings_ms: TimingsMs,

    gpu: GpuMetrics,
    build: BuildMetrics,
}

#[derive(Serialize)]
struct TimingsMs {
    total: u128,
    load_io: u128,
    gpu_init: u128,
    gpu_setup: u128,
    solve: u128,
    write_out: u128,
}

fn parse_backend(s: &str) -> GpuBackend {
    match s.to_lowercase().as_str() {
        "auto" => GpuBackend::Auto,
        "vulkan" => GpuBackend::Vulkan,
        "dx12" => GpuBackend::Dx12,
        "metal" => GpuBackend::Metal,
        other => {
            eprintln!("Unknown backend '{other}', using auto");
            GpuBackend::Auto
        }
    }
}

fn now_utc_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown-time".to_string())
}

fn run_vec_test(ctx: &GpuContext) {
    let n: usize = 1024;
    let alpha: f32 = 3.0;

    let x_host = vec![2.0f32; n];
    let y_host = vec![1.0f32; n];

    // Create GPU buffers.
    // We want STORAGE for compute + COPY_SRC for readback + COPY_DST for init/updates.
    let x = ctx.create_storage_buffer("x", &x_host, BufferUsages::empty());
    let y = ctx.create_storage_buffer("y", &y_host, BufferUsages::empty());

    // Create executor (pipelines + uniform pool).
    let vec_exec = VecOpsExecutor::create(ctx);
    vec_exec.reset_params_cursor();

    // Encode.
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("vec-test encoder"),
        });

    vec_exec.encode_axpy_inplace(ctx, &mut encoder, &x.buffer, &y.buffer, n as u32, alpha);

    // Submit exactly once (matches your model).
    ctx.queue.submit(Some(encoder.finish()));

    // Read back and assert.
    let y_out = executor::block_on(ctx.readback(&y));

    let expected = 1.0f32 + alpha * 2.0f32; // 7.0
    for (i, v) in y_out.iter().enumerate() {
        assert!(
            (*v - expected).abs() < 1e-6,
            "vec-test failed at i={i}: got {v}, expected {expected}"
        );
    }

    println!("VecTest OK: all y[i] == {expected}");
}

fn run_dot_test(ctx: &GpuContext) {
    // dot([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 32
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    let n = a.len() as u32;

    let a_buf = ctx.create_storage_buffer("dot a", &a, BufferUsages::empty());
    let b_buf = ctx.create_storage_buffer("dot b", &b, BufferUsages::empty());

    // Allocate executor for up to n elements, and 4 scalar slots.
    let exec = DotScalarExecutor::create(ctx, a.len(), 4);
    exec.reset_params_cursor();

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("dot-test encoder"),
        });

    exec.encode_dot_scalar_into(ctx, &mut encoder, &a_buf.buffer, &b_buf.buffer, n, 0);
    exec.encode_copy_scalar_results_to_readback(&mut encoder);

    ctx.queue.submit(Some(encoder.finish()));

    let scalars = futures::executor::block_on(exec.readback_scalar_results(ctx));

    let got = scalars[0];
    let expected = 32.0f32;

    assert!(
        (got - expected).abs() < 1e-5,
        "dot-test failed: got {got}, expected {expected}"
    );

    println!("DotTest OK: got {got}");
}

fn run_spmv_test(ctx: &GpuContext) {
    // 3x3 matrix:
    // [ 10 0  2 ]
    // [ 3  9  0 ]
    // [ 0  7  8 ]
    //
    // CSR:
    // row_ptr = [0, 2, 4, 6]
    // col_idx = [0,2, 0,1, 1,2]
    // vals    = [10,2, 3,9, 7,8]
    //
    // x = [1,2,3]
    // y = [16, 21, 38]

    let n_rows: u32 = 3;

    let row_ptr: Vec<u32> = vec![0, 2, 4, 6];
    let col_idx: Vec<u32> = vec![0, 2, 0, 1, 1, 2];
    let values: Vec<f32> = vec![10.0, 2.0, 3.0, 9.0, 7.0, 8.0];

    let x_host: Vec<f32> = vec![1.0, 2.0, 3.0];
    let expected: Vec<f32> = vec![16.0, 21.0, 38.0];

    // Upload x as a standalone GPU buffer (like your iteration vectors).
    let x_gpu = ctx.create_storage_buffer("spmv-test x_src", &x_host, BufferUsages::empty());

    // Create executor (owns CSR + internal x/y buffers).
    let spmv = SpmvExecutor::create(ctx, n_rows, &row_ptr, &col_idx, &values);

    // Encode: copy x into internal x_buffer, then SpMV into internal y_buffer.
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("spmv-test encoder"),
        });

    // n_bytes = n_rows * sizeof(f32)
    spmv.encode_copy_x_from(&mut encoder, &x_gpu.buffer, (n_rows as u64) * 4);
    spmv.encode_spmv(&mut encoder);

    // Single submit (matches your model).
    ctx.queue.submit(Some(encoder.finish()));

    // Read back y.
    // y_buffer length is n_rows f32.
    let y_out: Vec<f32> = executor::block_on(readback_to_vec::<f32>(
        &ctx.device,
        &ctx.queue,
        spmv.y_buffer(),
        n_rows as usize,
        Some("spmv-test y readback"),
    ));

    assert_eq!(y_out.len(), expected.len());

    for i in 0..expected.len() {
        let got = y_out[i];
        let exp = expected[i];
        assert!(
            (got - exp).abs() < 1e-6,
            "spmv-test failed at i={i}: got {got}, expected {exp}"
        );
    }

    println!("SpmvTest OK: y == {:?}", y_out);
}

fn run_block_jacobi_test(ctx: &GpuContext) {
    // Two 6x6 blocks => n = 12
    let n: u32 = 12;

    // block_starts: [0, 6, 12]
    let block_starts: Vec<u32> = vec![0, 6, 12];

    // Identity LU blocks: 2 blocks * 36 floats each
    let mut lu_blocks = vec![0.0f32; 2 * 36];
    for b in 0..2 {
        for i in 0..6 {
            lu_blocks[b * 36 + i * 6 + i] = 1.0;
        }
    }

    // r = [1..12]
    let r_host: Vec<f32> = (1..=12).map(|v| v as f32).collect();

    // Upload r and allocate z
    let r_gpu = ctx.create_storage_buffer("bj-test r", &r_host, BufferUsages::empty());
    let z_gpu =
        ctx.create_storage_buffer_uninit::<f32>("bj-test z", n as usize, BufferUsages::COPY_SRC);

    // Create executor
    let bj = BlockJacobiExecutor::create(ctx, n, &lu_blocks, &block_starts);

    // Encode apply and submit once
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("block-jacobi-test encoder"),
        });

    bj.encode_apply(ctx, &mut encoder, &r_gpu.buffer, &z_gpu.buffer);

    ctx.queue.submit(Some(encoder.finish()));

    // Read back z
    let z_out = executor::block_on(ctx.readback(&z_gpu));

    // Assert z == r
    for i in 0..(n as usize) {
        let got = z_out[i];
        let exp = r_host[i];
        assert!(
            (got - exp).abs() < 1e-6,
            "block-jacobi-test failed at i={i}: got {got}, expected {exp}"
        );
    }

    println!("BlockJacobiTest OK: z == r (identity blocks)");
}

const PAP: u32 = 0;
const RZ_NEW: u32 = 1;
const RZ_OLD: u32 = 2;
const ALPHA: u32 = 3;
const MINUS_ALPHA: u32 = 4;
const BETA: u32 = 5;

fn run_pcg_update_scalars_test(ctx: &GpuContext) {
    // Scalar buffer with a few slots.
    let scalar_len = 8usize;

    // Host init: we set only the inputs; outputs can start at 0.
    let mut host = vec![0.0f32; scalar_len];

    // pAp = 2, rz_old = 10, rz_new = 5
    host[PAP as usize] = 2.0;
    host[RZ_OLD as usize] = 10.0;
    host[RZ_NEW as usize] = 5.0;

    let scalar_gpu = ctx.create_storage_buffer(
        "pcg_update_scalars_test scalar_results",
        &host,
        BufferUsages::COPY_SRC, // for readback
    );

    let exec = PcgUpdateScalarsExecutor::create(ctx);
    exec.reset_params_cursor();

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("pcg_update_scalars_test encoder"),
        });

    exec.encode_update_scalars(
        ctx,
        &mut encoder,
        &scalar_gpu.buffer,
        PAP,
        RZ_NEW,
        RZ_OLD,
        ALPHA,
        MINUS_ALPHA,
        BETA,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let out = executor::block_on(ctx.readback(&scalar_gpu));

    let alpha = out[ALPHA as usize];
    let minus_alpha = out[MINUS_ALPHA as usize];
    let beta = out[BETA as usize];

    let exp_alpha = 10.0 / 2.0; // 5
    let exp_minus_alpha = -exp_alpha; // -5
    let exp_beta = 5.0 / 10.0; // 0.5

    let eps = 1e-6;
    assert!(
        (alpha - exp_alpha).abs() < eps,
        "alpha got {alpha}, expected {exp_alpha}"
    );
    assert!(
        (minus_alpha - exp_minus_alpha).abs() < eps,
        "minus_alpha got {minus_alpha}, expected {exp_minus_alpha}"
    );
    assert!(
        (beta - exp_beta).abs() < eps,
        "beta got {beta}, expected {exp_beta}"
    );

    println!("PcgUpdateScalarsTest OK: alpha={alpha}, minus_alpha={minus_alpha}, beta={beta}");
}

fn write_x_bin(path: &str, x: &[f32]) -> Result<(), String> {
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::Path;

    let p = Path::new(path);
    if let Some(parent) = p.parent()
        && !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
        }

    let mut f = File::create(p).map_err(|e| format!("create {}: {e}", p.display()))?;
    let n = x.len() as u32;
    f.write_all(&n.to_le_bytes()).map_err(|e| e.to_string())?;
    for &v in x {
        f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn write_json(path: &str, json: &str) -> Result<(), String> {
    let p = Path::new(path);
    if let Some(parent) = p.parent()
        && !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
        }

    fs::write(p, json).map_err(|e| format!("write {}: {e}", p.display()))
}

fn run_pcg_case(
    ctx: &GpuContext,
    case_dir: &str,
    max_iters: usize,
    rel_tol: f32,
    abs_tol: f32,
) -> Result<(usize, Vec<f32>, u32, u32), String> {
    // Load bin inputs (using your backend io module)
    let case = load_case_dir(Path::new(case_dir))?;

    let n = case.a.n_rows as usize;
    let nnz = case.a.nnz;

    // Create executors (once)
    // SpMV
    let spmv_exec = SpmvExecutor::create(
        ctx,
        case.a.n_rows,
        &case.a.row_ptr,
        &case.a.col_idx,
        &case.a.values,
    );

    // Vec ops
    let vec_ops_exec = VecOpsExecutor::create(ctx);

    // Dot scalar (needs >= 7 slots for PCG)
    let dot_scalar_exec = DotScalarExecutor::create(ctx, n, 7);

    // Build GPU block-Jacobi from LU blocks.
    //
    // IMPORTANT:
    // Your current BlockJacobiExecutor::create signature in CLI tests is:
    //   create(ctx, n, &lu_blocks, &block_starts)
    //
    // That means: you need LU blocks on input.
    //
    // But our BIN case currently contains: CSR + block_starts, NOT lu_blocks.
    //
    // So we must build lu_blocks from CSR here (like you did earlier in webgpu).
    //
    // If you already have a helper in backend compute.rs to build lu_blocks:
    //   build_lu_blocks_from_csr_block_starts_6(...)
    // use it here.
    //
    // For now, we assume you have (or will add) a function:
    //   wgpu_solver_backend::compute::block_jacobi_exec::build_lu_blocks_from_csr(...)
    //
    // I’ll show you exactly what to add below.

    let lu_blocks = build_lu_blocks_from_csr_block_starts_6(
        case.a.n_rows as usize,
        &case.a.row_ptr,
        &case.a.col_idx,
        &case.a.values,
        &case.block_starts.starts,
    )?;

    let block_jacobi_exec =
        BlockJacobiExecutor::create(ctx, case.a.n_rows, &lu_blocks, &case.block_starts.starts);

    let pcg_update_scalars_exec = PcgUpdateScalarsExecutor::create(ctx);

    // Solve
    let mut x = case.x0.values.clone();
    let iters = pcg_block_jacobi_csr_wgpu(
        n,
        &case.b.values,
        &mut x,
        max_iters,
        rel_tol,
        abs_tol,
        ctx,
        &spmv_exec,
        &vec_ops_exec,
        &dot_scalar_exec,
        &block_jacobi_exec,
        &pcg_update_scalars_exec,
    )?;

    Ok((iters, x, case.a.n_rows, nnz))
}

fn read_f32_vec_bin(path: &str) -> Result<Vec<f32>, String> {
    let mut f = File::open(path).map_err(|e| format!("open {path}: {e}"))?;

    let mut n_bytes = [0u8; 4];
    f.read_exact(&mut n_bytes)
        .map_err(|e| format!("read len header {path}: {e}"))?;
    let n = u32::from_le_bytes(n_bytes) as usize;

    let mut out = vec![0.0f32; n];
    let mut buf = [0u8; 4];

    for i in 0..n {
        f.read_exact(&mut buf)
            .map_err(|e| format!("read f32[{i}] {path}: {e}"))?;
        out[i] = f32::from_le_bytes(buf);
    }

    Ok(out)
}

#[derive(Debug)]
struct CompareStats {
    n: usize,

    // norms
    l2_ref: f64,
    l2_err: f64,
    rel_l2: f64,
    rmse: f64,

    // diagnostics only
    max_abs_err: f32,
    worst: Vec<WorstEntry>,
}

#[derive(Debug, Clone)]
struct WorstEntry {
    i: usize,
    x_ref: f32,
    x: f32,
    abs_err: f32,
}

fn compare_x_vectors(x_ref: &[f32], x: &[f32], top_k: usize) -> CompareStats {
    let n = x_ref.len().min(x.len());

    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;

    let mut max_abs_err = 0.0f32;

    let mut worst: Vec<WorstEntry> = Vec::with_capacity(top_k);

    for i in 0..n {
        let a = x_ref[i];
        let b = x[i];
        let abs_err = (b - a).abs();

        max_abs_err = max_abs_err.max(abs_err);

        sum_sq_err += (abs_err as f64) * (abs_err as f64);
        sum_sq_ref += (a as f64) * (a as f64);

        if top_k > 0 {
            if worst.len() < top_k {
                worst.push(WorstEntry {
                    i,
                    x_ref: a,
                    x: b,
                    abs_err,
                });
                worst.sort_by(|p, q| q.abs_err.partial_cmp(&p.abs_err).unwrap());
            } else if abs_err > worst.last().unwrap().abs_err {
                worst.pop();
                worst.push(WorstEntry {
                    i,
                    x_ref: a,
                    x: b,
                    abs_err,
                });
                worst.sort_by(|p, q| q.abs_err.partial_cmp(&p.abs_err).unwrap());
            }
        }
    }

    let l2_err = sum_sq_err.sqrt();
    let l2_ref = sum_sq_ref.sqrt();

    let eps = 1e-30f64;
    let rel_l2 = l2_err / l2_ref.max(eps);

    let rmse = (sum_sq_err / (n.max(1) as f64)).sqrt();

    CompareStats {
        n,
        l2_ref,
        l2_err,
        rel_l2,
        rmse,
        max_abs_err,
        worst,
    }
}

fn run_compare_x(
    x_ref_path: &str,
    x_path: &str,
    rel_tol: f32,
    abs_tol: f32,
    top_k: usize,
) -> Result<(), String> {
    let x_ref = read_f32_vec_bin(x_ref_path)?;
    let x = read_f32_vec_bin(x_path)?;

    if x_ref.len() != x.len() {
        return Err(format!(
            "Length mismatch: x_ref len {} vs x len {}",
            x_ref.len(),
            x.len()
        ));
    }

    let stats = compare_x_vectors(&x_ref, &x, top_k);

    // PASS criteria:
    // - relative L2 error small OR
    // - RMSE small
    let pass_rel = stats.rel_l2 <= (rel_tol as f64);
    let pass_rmse = stats.rmse <= (abs_tol as f64);
    let pass = pass_rel || pass_rmse;

    println!("CompareX:");
    println!("  n                 : {}", stats.n);
    println!("  rtol (rel L2)      : {:.3e}", rel_tol);
    println!("  atol (RMSE)        : {:.3e}", abs_tol);
    println!("  ||x_ref||_2        : {:.9e}", stats.l2_ref);
    println!("  ||x - x_ref||_2    : {:.9e}", stats.l2_err);
    println!("  rel_l2             : {:.9e}", stats.rel_l2);
    println!("  rmse               : {:.9e}", stats.rmse);
    println!("  max_abs_err (diag) : {:.9e}", stats.max_abs_err);
    println!("  pass_rel_l2        : {}", pass_rel);
    println!("  pass_rmse          : {}", pass_rmse);
    println!("  pass               : {}", pass);

    if !stats.worst.is_empty() {
        println!("\nTop {} worst entries (by abs error):", stats.worst.len());
        for w in &stats.worst {
            println!(
                "  i={:<8} x_ref={:.9e}  x={:.9e}  abs={:.9e}",
                w.i, w.x_ref, w.x, w.abs_err
            );
        }
    }

    if pass {
        Ok(())
    } else {
        Err("CompareX failed tolerances (rel_l2 and rmse)".into())
    }
}

fn main() {
    let cli = Cli::parse();
    let gpu_backend = parse_backend(&cli.backend);

    match cli.cmd {
        Cmd::Info => {
            let ctx = executor::block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                process::exit(2);
            });

            // Human-readable (nice in logs)
            println!("{}", ctx.describe());

            // Machine-readable (Slurm-friendly)
            let metrics = Metrics {
                run_id: now_utc_rfc3339(),
                command: "info".to_string(),
                gpu: GpuMetrics {
                    adapter_name: ctx.adapter_info.name.clone(),
                    backend: format!("{:?}", ctx.adapter_info.backend),
                    device_type: format!("{:?}", ctx.adapter_info.device_type),
                    vendor: ctx.adapter_info.vendor,
                    device: ctx.adapter_info.device,
                },
                build: BuildMetrics {
                    crate_version: env!("CARGO_PKG_VERSION").to_string(),
                    git_rev: option_env!("GIT_REV").map(|s| s.to_string()),
                },
            };

            println!("{}", to_string_pretty(&metrics).unwrap());
        }
        Cmd::VecTest => {
            let ctx: GpuContext = executor::block_on(GpuContext::create(gpu_backend))
                .unwrap_or_else(|e| {
                    eprintln!("Failed to init GPU context: {e}");
                    process::exit(2);
                });

            run_vec_test(&ctx);
        }
        Cmd::DotTest => {
            let ctx = executor::block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                process::exit(2);
            });

            run_dot_test(&ctx);
        }
        Cmd::SpmvTest => {
            let ctx = executor::block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                process::exit(2);
            });

            run_spmv_test(&ctx);
        }
        Cmd::BlockJacobiTest => {
            let ctx = executor::block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                process::exit(2);
            });

            run_block_jacobi_test(&ctx);
        }
        Cmd::PcgUpdateScalarsTest => {
            let ctx = executor::block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                process::exit(2);
            });

            run_pcg_update_scalars_test(&ctx);
        }
        Cmd::RunPcgCase {
            case_dir,
            max_iters,
            rel_tol,
            abs_tol,
            out_x,
            out_metrics,
        } => {
            use std::time::Instant;

            let t0 = Instant::now();
            let t_load0 = Instant::now();

            // Create ctx
            let t_gpu0 = Instant::now();
            let ctx = executor::block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                process::exit(2);
            });
            let t_gpu = t_gpu0.elapsed();

            // Solve (includes load inside run_pcg_case for now)
            let t_solve0 = Instant::now();
            let result = run_pcg_case(&ctx, &case_dir, max_iters, rel_tol, abs_tol);
            let t_solve = t_solve0.elapsed();

            let (iterations, x, n, nnz, converged, err) = match result {
                Ok((iters, x, n, nnz)) => (Some(iters), x, n, nnz, true, None),
                Err(e) => {
                    eprintln!("Solve failed: {e}");
                    (None, Vec::new(), 0, 0, false, Some(e))
                }
            };

            let t_write0 = Instant::now();
            if converged {
                write_x_bin(&out_x, &x).unwrap_or_else(|e| {
                    eprintln!("Failed to write x.bin: {e}");
                    process::exit(2);
                });
            }
            let t_write = t_write0.elapsed();

            let metrics = SolveMetrics {
                run_id: now_utc_rfc3339(),
                command: "run_pcg_case".to_string(),
                case_dir: case_dir.clone(),
                n,
                nnz,
                max_iters,
                rel_tol,
                abs_tol,
                iterations,
                converged,
                error: err,
                timings_ms: TimingsMs {
                    total: t0.elapsed().as_millis(),
                    load_io: t_load0.elapsed().as_millis(), // (kept simple; we can refine)
                    gpu_init: t_gpu.as_millis(),
                    gpu_setup: 0, // optional: split out later
                    solve: t_solve.as_millis(),
                    write_out: t_write.as_millis(),
                },
                gpu: GpuMetrics {
                    adapter_name: ctx.adapter_info.name.clone(),
                    backend: format!("{:?}", ctx.adapter_info.backend),
                    device_type: format!("{:?}", ctx.adapter_info.device_type),
                    vendor: ctx.adapter_info.vendor,
                    device: ctx.adapter_info.device,
                },
                build: BuildMetrics {
                    crate_version: env!("CARGO_PKG_VERSION").to_string(),
                    git_rev: option_env!("GIT_REV").map(|s| s.to_string()),
                },
            };

            let json = to_string_pretty(&metrics).unwrap();
            write_json(&out_metrics, &json).unwrap_or_else(|e| {
                eprintln!("Failed to write metrics: {e}");
                process::exit(2);
            });

            println!("{json}");
        }
        Cmd::CompareX {
            x_ref,
            x,
            rel_tol,
            abs_tol,
            top_k,
        } => {
            if let Err(e) = run_compare_x(&x_ref, &x, rel_tol, abs_tol, top_k) {
                eprintln!("{e}");
                process::exit(2);
            }
        }
    }
}
