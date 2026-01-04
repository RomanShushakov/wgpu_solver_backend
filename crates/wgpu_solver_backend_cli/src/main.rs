use clap::{Parser, Subcommand};
use futures::executor::block_on;
use serde::Serialize;
use serde_json::to_string_pretty;
use std::process::exit;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use wgpu::{BufferUsages, CommandEncoderDescriptor};
use wgpu_solver_backend::compute::dot_scalar_exec::DotScalarExecutor;
use wgpu_solver_backend::compute::spmv_exec::SpmvExecutor;
use wgpu_solver_backend::compute::vec_ops_exec::VecOpsExecutor;
use wgpu_solver_backend::gpu::context::{GpuBackend, GpuContext};

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
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print GPU adapter info and emit a metrics JSON blob (stdout).
    Info,
    /// Sanity test for vec ops (AXPY): y = y + alpha * x
    VecTest,
    DotTest,
    SpmvTest,
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
    let y_out = block_on(ctx.readback(&y));

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

fn run_spmv_test(ctx: &wgpu_solver_backend::gpu::context::GpuContext) {
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
    let y_out: Vec<f32> = block_on(wgpu_solver_backend::gpu::readback::readback_to_vec::<f32>(
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

fn main() {
    let cli = Cli::parse();
    let gpu_backend = parse_backend(&cli.backend);

    match cli.cmd {
        Command::Info => {
            let ctx = block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                exit(2);
            });

            // Human-readable (nice in logs)
            println!("{}", ctx.describe());

            // Machine-readable (Slurm-friendly)
            let m = Metrics {
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

            println!("{}", to_string_pretty(&m).unwrap());
        }
        Command::VecTest => {
            let ctx: GpuContext = block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                exit(2);
            });

            run_vec_test(&ctx);
        }
        Command::DotTest => {
            let ctx = block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                exit(2);
            });

            run_dot_test(&ctx);
        }
        Command::SpmvTest => {
            let ctx = block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                exit(2);
            });

            run_spmv_test(&ctx);
        }
    }
}
