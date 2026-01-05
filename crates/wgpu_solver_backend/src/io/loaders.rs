use std::fs;
use std::path::Path;

use super::bin_format::{BlockStartsBin, CaseInputBin, CsrMatrixBin, VectorBin};

const MATRIX_MAGIC_CSR1: u32 = 0x4353_5231;

pub fn load_case_dir(case_dir: &Path) -> Result<CaseInputBin, String> {
    let a = load_csr_matrix_bin(&case_dir.join("matrix.csr.bin"))?;
    let b = load_vector_bin(&case_dir.join("rhs.bin"))?;
    let x0 = load_vector_bin(&case_dir.join("x0.bin"))?;
    let block_starts = load_block_starts_bin(&case_dir.join("block_starts.bin"))?;

    // Cross-validation:
    if a.n_rows != a.n_cols {
        return Err(format!(
            "matrix must be square for PCG: got {}x{}",
            a.n_rows, a.n_cols
        ));
    }
    if b.n != a.n_rows {
        return Err(format!(
            "rhs length mismatch: b.n={} but matrix n_rows={}",
            b.n, a.n_rows
        ));
    }
    if x0.n != a.n_rows {
        return Err(format!(
            "x0 length mismatch: x0.n={} but matrix n_rows={}",
            x0.n, a.n_rows
        ));
    }

    validate_block_starts(&block_starts.starts, a.n_rows)?;

    Ok(CaseInputBin {
        a,
        b,
        x0,
        block_starts,
    })
}

pub fn load_csr_matrix_bin(path: &Path) -> Result<CsrMatrixBin, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let mut cur = Cursor::new(&bytes);

    let magic = cur.read_u32_le("magic")?;
    if magic != MATRIX_MAGIC_CSR1 {
        return Err(format!(
            "Bad matrix magic in {}: expected 0x{:08x}, got 0x{:08x}",
            path.display(),
            MATRIX_MAGIC_CSR1,
            magic
        ));
    }

    let version = cur.read_u32_le("version")?;
    if version != 1 {
        return Err(format!(
            "Unsupported matrix version in {}: expected 1, got {}",
            path.display(),
            version
        ));
    }

    let n_rows = cur.read_u32_le("n_rows")?;
    let n_cols = cur.read_u32_le("n_cols")?;
    let nnz = cur.read_u32_le("nnz")?;

    // reserved (ignored but must exist)
    let _ = cur.read_u32_le("reserved0")?;
    let _ = cur.read_u32_le("reserved1")?;
    let _ = cur.read_u32_le("reserved2")?;

    let row_ptr_len = (n_rows as usize).checked_add(1).ok_or("n_rows too large")?;
    let nnz_usize = nnz as usize;

    let row_ptr = cur.read_u32_slice_le(row_ptr_len, "row_ptr")?;
    let col_idx = cur.read_u32_slice_le(nnz_usize, "col_idx")?;
    let values = cur.read_f32_slice_le(nnz_usize, "values")?;

    // basic validations
    if row_ptr.first().copied() != Some(0) {
        return Err("CSR row_ptr must start with 0".into());
    }
    if row_ptr.last().copied() != Some(nnz) {
        return Err(format!(
            "CSR row_ptr must end with nnz ({}), got {:?}",
            nnz,
            row_ptr.last()
        ));
    }
    for w in row_ptr.windows(2) {
        if w[1] < w[0] {
            return Err("CSR row_ptr must be non-decreasing".into());
        }
    }
    let max_col = col_idx.iter().copied().max().unwrap_or(0);
    if n_cols > 0 && max_col >= n_cols {
        return Err(format!(
            "CSR col_idx contains out-of-range column {} for n_cols={}",
            max_col, n_cols
        ));
    }

    // ensure file has no extra trailing garbage (optional strictness)
    if cur.remaining() != 0 {
        // Not fatal, but nice to detect format mismatches early.
        return Err(format!(
            "matrix.csr.bin has {} trailing bytes after expected payload",
            cur.remaining()
        ));
    }

    Ok(CsrMatrixBin {
        n_rows,
        n_cols,
        nnz,
        row_ptr,
        col_idx,
        values,
    })
}

pub fn load_vector_bin(path: &Path) -> Result<VectorBin, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let mut cur = Cursor::new(&bytes);

    let n = cur.read_u32_le("n")?;
    let n_usize = n as usize;

    let values = cur.read_f32_slice_le(n_usize, "values")?;

    if cur.remaining() != 0 {
        return Err(format!(
            "{} has {} trailing bytes after expected payload",
            path.display(),
            cur.remaining()
        ));
    }

    Ok(VectorBin { n, values })
}

pub fn load_block_starts_bin(path: &Path) -> Result<BlockStartsBin, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let mut cur = Cursor::new(&bytes);

    let len = cur.read_u32_le("len")? as usize;
    let starts = cur.read_u32_slice_le(len, "block_starts")?;

    if cur.remaining() != 0 {
        return Err(format!(
            "{} has {} trailing bytes after expected payload",
            path.display(),
            cur.remaining()
        ));
    }

    Ok(BlockStartsBin { starts })
}

fn validate_block_starts(starts: &[u32], n: u32) -> Result<(), String> {
    if starts.is_empty() {
        return Err("block_starts must not be empty".into());
    }
    if starts[0] != 0 {
        return Err(format!("block_starts must start with 0, got {}", starts[0]));
    }
    if *starts.last().unwrap() != n {
        return Err(format!(
            "block_starts must end with n ({}), got {}",
            n,
            starts.last().unwrap()
        ));
    }
    for w in starts.windows(2) {
        if w[1] <= w[0] {
            return Err("block_starts must be strictly increasing".into());
        }
    }
    Ok(())
}

/// Simple byte cursor over a slice for LE parsing (no allocations).
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize, label: &str) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| format!("Overflow while reading {label}"))?;
        if end > self.bytes.len() {
            return Err(format!(
                "Unexpected EOF while reading {label}: need {} bytes, have {}",
                n,
                self.remaining()
            ));
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn read_u32_le(&mut self, label: &str) -> Result<u32, String> {
        let b = self.take(4, label)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u32_slice_le(&mut self, len: usize, label: &str) -> Result<Vec<u32>, String> {
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            out.push(self.read_u32_le(&format!("{label}[{i}]"))?);
        }
        Ok(out)
    }

    fn read_f32_slice_le(&mut self, len: usize, label: &str) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let b = self.take(4, &format!("{label}[{i}]"))?;
            out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        Ok(out)
    }
}
