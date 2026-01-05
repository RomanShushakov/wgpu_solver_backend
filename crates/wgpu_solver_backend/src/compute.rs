use futures::executor;
use wgpu::{BufferUsages, CommandEncoderDescriptor};

use crate::{
    compute::{
        block_jacobi_exec::BlockJacobiExecutor,
        buffers::encode_write_f32_into_storage_buffer_at_index, dot_scalar_exec::DotScalarExecutor,
        pcg_update_scalars_exec::PcgUpdateScalarsExecutor, spmv_exec::SpmvExecutor,
        vec_ops_exec::VecOpsExecutor,
    },
    gpu::context::GpuContext,
};

pub mod block_jacobi;
pub mod block_jacobi_exec;
pub mod buffers;
pub mod dot_partials;
pub mod dot_reduce;
pub mod dot_scalar_exec;
pub mod pcg_update_scalars;
pub mod pcg_update_scalars_exec;
pub mod spmv;
pub mod spmv_exec;
pub mod vec_ops;
pub mod vec_ops_exec;

/// In-place LU factorization (no pivoting) for a small dense matrix stored in a fixed 6x6 buffer.
///
/// Storage / layout:
/// - `mat` is a row-major 6x6 buffer (length must be >= 36).
/// - `n` is the active dimension (we factor only the leading n×n block, where n <= 6).
///
/// Output convention (IMPORTANT: must match block_jacobi.wgsl):
/// - Strict lower triangle (i > j): stores L(i,j)
/// - Diagonal + upper (i <= j): stores U(i,j)
/// - L has implicit unit diagonal (L(i,i) == 1.0 not stored)
pub fn lu_factor_inplace_6(mat: &mut [f32], n: usize) -> Result<(), String> {
    let zero = 0.0f32;
    let stride = 6usize;

    if mat.len() < 36 {
        return Err("lu_factor_inplace_6: mat must have len >= 36".into());
    }
    if n > 6 {
        return Err(format!("lu_factor_inplace_6: n must be <= 6, got {n}"));
    }

    for k in 0..n {
        let a_kk = mat[k * stride + k];
        if a_kk == zero {
            return Err(format!("LU zero pivot at k = {}", k));
        }

        // L(i,k) = A(i,k) / pivot
        for i in (k + 1)..n {
            mat[i * stride + k] /= a_kk;
        }

        // trailing update: A(i,j) -= L(i,k) * U(k,j)
        for i in (k + 1)..n {
            let l_ik = mat[i * stride + k];
            if l_ik != zero {
                for j in (k + 1)..n {
                    mat[i * stride + j] -= l_ik * mat[k * stride + j];
                }
            }
        }
    }

    Ok(())
}

/// Build LU blocks (fixed 6x6 storage, active size m<=6) from CSR + block_starts.
///
/// This is the native port of fea_app's builder and MUST match block_jacobi.wgsl.
///
/// Inputs:
/// - CSR arrays are for an n×n matrix.
/// - block_starts is a partition of [0..n], monotonic increasing, last == n.
/// - Each block can be size 1..6 (or bigger, but then we will only take first 6 rows/cols).
///
/// Output:
/// - concatenated blocks, each block is 36 floats (6x6 row-major packed LU)
pub fn build_lu_blocks_from_csr_block_starts_6(
    n: usize,
    row_ptr: &[u32],
    col_idx: &[u32],
    values: &[f32],
    block_starts: &[u32],
) -> Result<Vec<f32>, String> {
    let block_size = 6usize;

    // Basic CSR sanity
    if row_ptr.len() != n + 1 {
        return Err(format!(
            "build_lu_blocks_from_csr_block_starts_6: row_ptr len must be n+1 ({}), got {}",
            n + 1,
            row_ptr.len()
        ));
    }
    let nnz = *row_ptr.last().unwrap() as usize;
    if col_idx.len() != nnz || values.len() != nnz {
        return Err(format!(
            "build_lu_blocks_from_csr_block_starts_6: nnz mismatch: row_ptr says {}, col_idx {}, values {}",
            nnz,
            col_idx.len(),
            values.len()
        ));
    }

    if block_starts.len() < 2 {
        return Ok(vec![]);
    }
    if block_starts[0] != 0 {
        return Err("build_lu_blocks_from_csr_block_starts_6: block_starts must start at 0".into());
    }
    if *block_starts.last().unwrap() as usize != n {
        return Err(format!(
            "build_lu_blocks_from_csr_block_starts_6: block_starts last must equal n ({}), got {}",
            n,
            block_starts.last().unwrap()
        ));
    }
    for w in block_starts.windows(2) {
        if w[1] <= w[0] {
            return Err(
                "build_lu_blocks_from_csr_block_starts_6: block_starts must be strictly increasing"
                    .into(),
            );
        }
    }

    let num_blocks = block_starts.len() - 1;
    let mut out = vec![0.0f32; num_blocks * 36];

    for block in 0..num_blocks {
        let offset = block_starts[block] as usize;
        let end = block_starts[block + 1] as usize;

        // Invalid/empty: keep zeros (or identity), but we follow fea_app: skip.
        if end > n || offset >= end {
            continue;
        }

        // Active size m <= 6
        let m = (end - offset).min(block_size);

        // Local dense 6x6, row-major
        let mut mat = [0.0f32; 36];

        // Identity fill (critical for missing diagonals / partial blocks)
        for i in 0..block_size {
            mat[i * block_size + i] = 1.0;
        }

        // Fill leading m×m from CSR, only entries fully inside this block region.
        for i_local in 0..m {
            let i = offset + i_local;

            let row_start = row_ptr[i] as usize;
            let row_end = row_ptr[i + 1] as usize;

            for idx in row_start..row_end {
                let j = col_idx[idx] as usize;

                if j >= offset && j < offset + m {
                    let j_local = j - offset;
                    mat[i_local * block_size + j_local] = values[idx];
                }
            }
        }

        // Factor only the leading m×m
        lu_factor_inplace_6(&mut mat, m)?;

        // Copy full 6x6 slab into out
        out[block * 36..block * 36 + 36].copy_from_slice(&mat);
    }

    Ok(out)
}

/// Native wgpu port of fea_app's `pcg_block_jacobi_csr_webgpu`.
///
/// IMPORTANT: this is a *core loop* only:
/// - assumes all inputs are already prepared (CSR arrays, block preconditioner already built)
/// - executors are created outside and passed in (no hidden allocations)
/// - 1 submit + 1 scalar readback per iteration (same design)
pub fn pcg_block_jacobi_csr_wgpu(
    // sizes
    n: usize,
    // rhs and in/out solution
    b: &[f32],
    x: &mut [f32],
    // stop
    max_iter: usize,
    rel_tol: f32,
    abs_tol: f32,
    // gpu context + executors (created once outside)
    ctx: &GpuContext,
    spmv_exec: &SpmvExecutor,
    vec_ops_exec: &VecOpsExecutor,
    dot_scalar_exec: &DotScalarExecutor,
    block_jacobi_exec: &BlockJacobiExecutor,
    pcg_update_scalars_exec: &PcgUpdateScalarsExecutor,
) -> Result<usize, String> {
    // -------------------------------------------------------------------------
    // 0) Validate dimensions and precompute common constants
    // -------------------------------------------------------------------------
    if b.len() != n || x.len() != n {
        return Err(format!(
            "PCG(BlockJacobiGpu): dimension mismatch: n={}, b len {}, x len {}",
            n,
            b.len(),
            x.len()
        ));
    }

    let zero: f32 = 0.0;
    let n_u32: u32 = n as u32;
    let n_bytes: u64 = (n * 4) as u64;

    // -------------------------------------------------------------------------
    // 1) Compute ||b||^2 once (GPU), same as fea_app
    // -------------------------------------------------------------------------
    let b_gpu = ctx.create_storage_buffer("pcg b", b, wgpu::BufferUsages::COPY_SRC);

    let b_norm2: f32 = {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pcg b_norm2 encoder"),
            });

        dot_scalar_exec.encode_dot_scalar_into(
            ctx,
            &mut encoder,
            &b_gpu.buffer,
            &b_gpu.buffer,
            n_u32,
            0,
        );
        dot_scalar_exec.encode_copy_scalar_results_to_readback(&mut encoder);

        ctx.queue.submit(Some(encoder.finish()));

        let scalar_results = executor::block_on(dot_scalar_exec.readback_scalar_results(ctx));

        scalar_results[0]
    };

    if b_norm2 == zero {
        return Ok(0);
    }

    let rel_tol2: f32 = rel_tol * rel_tol;
    let abs_tol2: f32 = abs_tol * abs_tol;

    // -------------------------------------------------------------------------
    // 2) Upload initial vectors to GPU
    // -------------------------------------------------------------------------
    let x_gpu = ctx.create_storage_buffer("pcg x", x, BufferUsages::COPY_SRC);
    let r_gpu = ctx.create_storage_buffer_uninit::<f32>("pcg r", n, BufferUsages::COPY_SRC);
    let p_gpu = ctx.create_storage_buffer_uninit::<f32>("pcg p", n, BufferUsages::COPY_SRC);
    let z_gpu = ctx.create_storage_buffer_uninit::<f32>("pcg z", n, BufferUsages::COPY_SRC);

    // -------------------------------------------------------------------------
    // 3) Scalar slot layout (local, identical concept to fea_app)
    // DotScalarExecutor must have scalar_results_len >= 7.
    // -------------------------------------------------------------------------
    let scalar_results_index_for_p_ap: u32 = 0; // p^T (A p)
    let scalar_results_index_for_r_norm2: u32 = 1; // r^T r
    let scalar_results_index_for_rz_new: u32 = 2; // r^T z (new)
    let scalar_results_index_for_rz_old: u32 = 3; // r^T z (old) written from CPU
    let scalar_results_index_for_alpha: u32 = 4; // alpha
    let scalar_results_index_for_minus_alpha: u32 = 5; // -alpha
    let scalar_results_index_for_beta: u32 = 6; // beta

    // -------------------------------------------------------------------------
    // 4) Initialize r0 = b - A*x0, z0 = M^-1 r0, p0 = z0, rz_old
    // We do this init on GPU (native core should be self-contained).
    // -------------------------------------------------------------------------
    let mut rz_old: f32 = {
        vec_ops_exec.reset_params_cursor();
        pcg_update_scalars_exec.reset_params_cursor();

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("pcg init encoder"),
            });

        // r <- b
        encoder.copy_buffer_to_buffer(&b_gpu.buffer, 0, &r_gpu.buffer, 0, n_bytes);

        // Ap = A*x (spmv writes to spmv_exec.y_buffer())
        spmv_exec.encode_copy_x_from(&mut encoder, &x_gpu.buffer, n_bytes);
        spmv_exec.encode_spmv(&mut encoder);

        // r = r + (-1)*Ap
        vec_ops_exec.encode_axpy_inplace(
            ctx,
            &mut encoder,
            spmv_exec.y_buffer(),
            &r_gpu.buffer,
            n_u32,
            -1.0,
        );

        // z = M^-1 r
        block_jacobi_exec.encode_apply(ctx, &mut encoder, &r_gpu.buffer, &z_gpu.buffer);

        // p = z
        encoder.copy_buffer_to_buffer(&z_gpu.buffer, 0, &p_gpu.buffer, 0, n_bytes);

        // rz_old = dot(r,z) -> store in slot [rz_old]
        dot_scalar_exec.encode_dot_scalar_into(
            ctx,
            &mut encoder,
            &r_gpu.buffer,
            &z_gpu.buffer,
            n_u32,
            scalar_results_index_for_rz_old,
        );

        dot_scalar_exec.encode_copy_scalar_results_to_readback(&mut encoder);
        ctx.queue.submit(Some(encoder.finish()));

        let scalar_results = executor::block_on(dot_scalar_exec.readback_scalar_results(ctx));
        scalar_results[scalar_results_index_for_rz_old as usize]
    };

    // -------------------------------------------------------------------------
    // 5) Main PCG loop (single submit + scalar readback per iteration)
    // -------------------------------------------------------------------------
    for k in 0..max_iter {
        let iterations = k + 1;

        vec_ops_exec.reset_params_cursor();
        pcg_update_scalars_exec.reset_params_cursor();

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("pcg single-submit iteration encoder"),
            });

        // A) Ap = A * p
        spmv_exec.encode_copy_x_from(&mut encoder, &p_gpu.buffer, n_bytes);
        spmv_exec.encode_spmv(&mut encoder);

        // B) pAp = dot(p, Ap)
        dot_scalar_exec.encode_dot_scalar_into(
            ctx,
            &mut encoder,
            &p_gpu.buffer,
            spmv_exec.y_buffer(),
            n_u32,
            scalar_results_index_for_p_ap,
        );

        // C) Write rz_old into scalar_results[rz_old]
        encode_write_f32_into_storage_buffer_at_index(
            &ctx.device,
            &mut encoder,
            dot_scalar_exec.scalar_results_buffer(),
            scalar_results_index_for_rz_old,
            rz_old,
            "pcg rz_old staging",
        );

        // D) compute alpha / -alpha (early)
        pcg_update_scalars_exec.encode_update_scalars(
            ctx,
            &mut encoder,
            dot_scalar_exec.scalar_results_buffer(),
            scalar_results_index_for_p_ap,
            scalar_results_index_for_rz_new, // placeholder early
            scalar_results_index_for_rz_old,
            scalar_results_index_for_alpha,
            scalar_results_index_for_minus_alpha,
            scalar_results_index_for_beta,
        );

        // E) x = x + alpha*p ; r = r + (-alpha)*Ap
        vec_ops_exec.encode_axpy_inplace_from_scalar_results(
            ctx,
            &mut encoder,
            &p_gpu.buffer,
            &x_gpu.buffer,
            n_u32,
            dot_scalar_exec.scalar_results_buffer(),
            scalar_results_index_for_alpha,
        );
        vec_ops_exec.encode_axpy_inplace_from_scalar_results(
            ctx,
            &mut encoder,
            spmv_exec.y_buffer(),
            &r_gpu.buffer,
            n_u32,
            dot_scalar_exec.scalar_results_buffer(),
            scalar_results_index_for_minus_alpha,
        );

        // F) r_norm2 = dot(r,r)
        dot_scalar_exec.encode_dot_scalar_into(
            ctx,
            &mut encoder,
            &r_gpu.buffer,
            &r_gpu.buffer,
            n_u32,
            scalar_results_index_for_r_norm2,
        );

        // G) z = M^-1 r
        block_jacobi_exec.encode_apply(ctx, &mut encoder, &r_gpu.buffer, &z_gpu.buffer);

        // H) rz_new = dot(r,z)
        dot_scalar_exec.encode_dot_scalar_into(
            ctx,
            &mut encoder,
            &r_gpu.buffer,
            &z_gpu.buffer,
            n_u32,
            scalar_results_index_for_rz_new,
        );

        // I) compute beta (late)
        pcg_update_scalars_exec.encode_update_scalars(
            ctx,
            &mut encoder,
            dot_scalar_exec.scalar_results_buffer(),
            scalar_results_index_for_p_ap,
            scalar_results_index_for_rz_new,
            scalar_results_index_for_rz_old,
            scalar_results_index_for_alpha,
            scalar_results_index_for_minus_alpha,
            scalar_results_index_for_beta,
        );

        // J) p = z + beta*p
        vec_ops_exec.encode_scale_inplace_from_scalar_results(
            ctx,
            &mut encoder,
            &p_gpu.buffer,
            n_u32,
            dot_scalar_exec.scalar_results_buffer(),
            scalar_results_index_for_beta,
        );
        vec_ops_exec.encode_axpy_inplace(
            ctx,
            &mut encoder,
            &z_gpu.buffer,
            &p_gpu.buffer,
            n_u32,
            1.0,
        );

        // K) scalar_results -> readback
        dot_scalar_exec.encode_copy_scalar_results_to_readback(&mut encoder);

        // Submit once
        ctx.queue.submit(Some(encoder.finish()));

        // Read scalars once
        let scalar_results = executor::block_on(dot_scalar_exec.readback_scalar_results(ctx));

        let p_ap = scalar_results[scalar_results_index_for_p_ap as usize];
        let r_norm2 = scalar_results[scalar_results_index_for_r_norm2 as usize];
        let rz_new = scalar_results[scalar_results_index_for_rz_new as usize];

        // breakdown checks
        if p_ap == zero {
            return Err("PCG(BlockJacobiGpu): dot(p,Ap) is zero (breakdown)".into());
        }
        if rz_old == zero {
            return Err("PCG(BlockJacobiGpu): rz_old is zero (breakdown)".into());
        }

        // stopping condition
        if r_norm2 <= abs_tol2 || r_norm2 <= rel_tol2 * b_norm2 {
            let x_out = executor::block_on(ctx.readback(&x_gpu));
            x.copy_from_slice(&x_out);
            return Ok(iterations);
        }

        // update rz_old (CPU) for next iteration
        rz_old = rz_new;
    }

    Err(format!(
        "PCG(BlockJacobiGpu): did not converge in {} iterations",
        max_iter
    ))
}
