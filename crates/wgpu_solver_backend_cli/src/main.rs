use clap::{Parser, Subcommand};
use futures::executor::block_on;
use serde::Serialize;
use serde_json::to_string_pretty;
use std::process::exit;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
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
    let x = ctx.create_storage_buffer("x", &x_host, wgpu::BufferUsages::empty());
    let y = ctx.create_storage_buffer("y", &y_host, wgpu::BufferUsages::empty());

    // Create executor (pipelines + uniform pool).
    let vec_exec = VecOpsExecutor::create(ctx);
    vec_exec.reset_params_cursor();

    // Encode.
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
    }
}
