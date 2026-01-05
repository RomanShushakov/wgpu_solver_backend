use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "wgpu_solver_backend_convert_cli")]
#[command(about = "Converters for wgpu-solver-backend datasets (text -> bin)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Convert a fea_app-exported dataset.txt bundle into binary files.
    TextToBin {
        /// Input dataset text file exported from fea_app worker.
        #[arg(long)]
        input: PathBuf,

        /// Output directory to create files in.
        #[arg(long)]
        out_dir: PathBuf,

        /// Overwrite output directory if it exists.
        #[arg(long, default_value_t = false)]
        force: bool,
    },
}

#[derive(Debug)]
struct TextDataset {
    n: usize,
    nnz: usize,

    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    values: Vec<f32>,

    b: Vec<f32>,
    x0: Vec<f32>,

    // Per our agreement: keep it always (as input to solver too).
    block_starts: Vec<u32>,
}

#[derive(Serialize)]
struct MetaJson {
    format: String,
    n: usize,
    nnz: usize,
    row_ptr_len: usize,
    col_idx_len: usize,
    values_len: usize,
    rhs_len: usize,
    x0_len: usize,
    block_starts_len: usize,
}

/// v0 binary format for matrix.csr.bin:
///
/// header (all LE):
///   u32 magic = 0x43535231  ("CSR1")
///   u32 version = 1
///   u32 n_rows
///   u32 n_cols
///   u32 nnz
///   u32 reserved0
///   u32 reserved1
///   u32 reserved2
///
/// payload:
///   row_ptr: u32[n_rows+1]
///   col_idx: u32[nnz]
///   values : f32[nnz]
const MATRIX_MAGIC_CSR1: u32 = 0x4353_5231;

fn main() -> Result<(), String> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::TextToBin {
            input,
            out_dir,
            force,
        } => {
            let ds = parse_dataset_text_file(&input)?;
            write_bins(&ds, &out_dir, force)?;
            println!("Wrote binaries into {}", out_dir.display());
            Ok(())
        }
    }
}

// ==============================
// Parsing (streaming, robust)
// ==============================

fn parse_dataset_text_file(path: &Path) -> Result<TextDataset, String> {
    let f = File::open(path).map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let mut r = BufReader::new(f);

    let mut n: Option<usize> = None;
    let mut nnz: Option<usize> = None;

    let mut row_ptr: Option<Vec<u32>> = None;
    let mut col_idx: Option<Vec<u32>> = None;
    let mut values: Option<Vec<f32>> = None;
    let mut b: Option<Vec<f32>> = None;
    let mut x0: Option<Vec<f32>> = None;
    let mut block_starts: Option<Vec<u32>> = None;

    fn next_data_line<R: BufRead>(r: &mut R, buf: &mut String) -> Result<Option<String>, String> {
        loop {
            buf.clear();
            let n = r.read_line(buf).map_err(|e| e.to_string())?;
            if n == 0 {
                return Ok(None);
            }
            let line = buf.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            return Ok(Some(line.to_string()));
        }
    }

    fn parse_len_line(line: &str, section: &str) -> Result<usize, String> {
        let rest = line
            .strip_prefix("len ")
            .ok_or_else(|| format!("Section {section}: expected 'len N', got '{line}'"))?;
        rest.parse::<usize>()
            .map_err(|e| format!("Section {section}: bad len '{rest}': {e}"))
    }

    fn is_header_like(line: &str) -> bool {
        line.starts_with("n ")
            || line.starts_with("nnz ")
            || line.ends_with("_u32")
            || line.ends_with("_f32")
    }

    fn read_tokens_into_vec_u32<R: BufRead>(
        r: &mut R,
        buf: &mut String,
        section: &str,
        need: usize,
    ) -> Result<Vec<u32>, String> {
        let mut out = Vec::with_capacity(need);
        while out.len() < need {
            buf.clear();
            let nread = r.read_line(buf).map_err(|e| e.to_string())?;
            if nread == 0 {
                return Err(format!(
                    "Section {section}: EOF while reading values (got {}, need {need})",
                    out.len()
                ));
            }
            let line = buf.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if is_header_like(line) {
                return Err(format!(
                    "Section {section}: hit unexpected header '{line}' before collecting len={need} values"
                ));
            }
            for t in line.split_whitespace() {
                if out.len() == need {
                    break;
                }
                out.push(t.parse::<u32>().map_err(|e| {
                    format!("Section {section}: failed to parse u32 token '{t}': {e}")
                })?);
            }
        }
        Ok(out)
    }

    fn read_tokens_into_vec_f32<R: BufRead>(
        r: &mut R,
        buf: &mut String,
        section: &str,
        need: usize,
    ) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(need);
        while out.len() < need {
            buf.clear();
            let nread = r.read_line(buf).map_err(|e| e.to_string())?;
            if nread == 0 {
                return Err(format!(
                    "Section {section}: EOF while reading values (got {}, need {need})",
                    out.len()
                ));
            }
            let line = buf.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if is_header_like(line) {
                return Err(format!(
                    "Section {section}: hit unexpected header '{line}' before collecting len={need} values"
                ));
            }
            for t in line.split_whitespace() {
                if out.len() == need {
                    break;
                }
                out.push(t.parse::<f32>().map_err(|e| {
                    format!("Section {section}: failed to parse f32 token '{t}': {e}")
                })?);
            }
        }
        Ok(out)
    }

    let mut buf = String::new();

    while let Some(line) = next_data_line(&mut r, &mut buf)? {
        if let Some(rest) = line.strip_prefix("n ") {
            n = Some(rest.parse::<usize>().map_err(|e| format!("Bad n: {e}"))?);
            continue;
        }
        if let Some(rest) = line.strip_prefix("nnz ") {
            nnz = Some(rest.parse::<usize>().map_err(|e| format!("Bad nnz: {e}"))?);
            continue;
        }

        // section header
        let section = line;

        let len_line = next_data_line(&mut r, &mut buf)?
            .ok_or_else(|| format!("Section {section}: missing len line (EOF)"))?;
        let expected_len = parse_len_line(&len_line, &section)?;

        match section.as_str() {
            "row_ptr_u32" => {
                row_ptr = Some(read_tokens_into_vec_u32(
                    &mut r,
                    &mut buf,
                    &section,
                    expected_len,
                )?)
            }
            "col_idx_u32" => {
                col_idx = Some(read_tokens_into_vec_u32(
                    &mut r,
                    &mut buf,
                    &section,
                    expected_len,
                )?)
            }
            "values_f32" => {
                values = Some(read_tokens_into_vec_f32(
                    &mut r,
                    &mut buf,
                    &section,
                    expected_len,
                )?)
            }
            "b_f32" => {
                b = Some(read_tokens_into_vec_f32(
                    &mut r,
                    &mut buf,
                    &section,
                    expected_len,
                )?)
            }
            "x0_f32" => {
                x0 = Some(read_tokens_into_vec_f32(
                    &mut r,
                    &mut buf,
                    &section,
                    expected_len,
                )?)
            }
            "block_starts_u32" => {
                block_starts = Some(read_tokens_into_vec_u32(
                    &mut r,
                    &mut buf,
                    &section,
                    expected_len,
                )?)
            }
            other => return Err(format!("Unknown section name: {other}")),
        }
    }

    let n = n.ok_or_else(|| "Missing 'n'".to_string())?;
    let nnz = nnz.ok_or_else(|| "Missing 'nnz'".to_string())?;

    let row_ptr = row_ptr.ok_or_else(|| "Missing row_ptr_u32".to_string())?;
    let col_idx = col_idx.ok_or_else(|| "Missing col_idx_u32".to_string())?;
    let values = values.ok_or_else(|| "Missing values_f32".to_string())?;
    let b = b.ok_or_else(|| "Missing b_f32".to_string())?;
    let x0 = x0.ok_or_else(|| "Missing x0_f32".to_string())?;
    let block_starts = block_starts.ok_or_else(|| "Missing block_starts_u32".to_string())?;

    // Validations
    if row_ptr.len() != n + 1 {
        return Err(format!(
            "row_ptr_u32 len must be n+1 ({}), got {}",
            n + 1,
            row_ptr.len()
        ));
    }
    if col_idx.len() != nnz || values.len() != nnz {
        return Err(format!(
            "col_idx/values len must be nnz ({}), got col_idx={}, values={}",
            nnz,
            col_idx.len(),
            values.len()
        ));
    }
    if b.len() != n || x0.len() != n {
        return Err(format!(
            "b/x0 len must be n ({}), got b={}, x0={}",
            n,
            b.len(),
            x0.len()
        ));
    }

    // block_starts sanity: monotonic, first=0, last=n
    if block_starts.is_empty() {
        return Err("block_starts_u32 must not be empty".to_string());
    }
    if block_starts[0] != 0 {
        return Err(format!(
            "block_starts_u32 must start with 0, got {}",
            block_starts[0]
        ));
    }
    if *block_starts.last().unwrap() != n as u32 {
        return Err(format!(
            "block_starts_u32 must end with n ({}), got {}",
            n,
            block_starts.last().unwrap()
        ));
    }
    for w in block_starts.windows(2) {
        if w[1] <= w[0] {
            return Err("block_starts_u32 must be strictly increasing".to_string());
        }
    }

    Ok(TextDataset {
        n,
        nnz,
        row_ptr,
        col_idx,
        values,
        b,
        x0,
        block_starts,
    })
}

// ==============================
// Writing .bin outputs
// ==============================

fn write_bins(ds: &TextDataset, out_dir: &Path, force: bool) -> Result<(), String> {
    if out_dir.exists() {
        if !force {
            return Err(format!(
                "Output dir {} already exists. Use --force to overwrite.",
                out_dir.display()
            ));
        }
        fs::remove_dir_all(out_dir)
            .map_err(|e| format!("Failed to remove {}: {e}", out_dir.display()))?;
    }
    fs::create_dir_all(out_dir)
        .map_err(|e| format!("Failed to create {}: {e}", out_dir.display()))?;

    // matrix.csr.bin
    {
        let mut f = File::create(out_dir.join("matrix.csr.bin"))
            .map_err(|e| format!("Failed to create matrix.csr.bin: {e}"))?;

        write_u32_le(&mut f, MATRIX_MAGIC_CSR1)?;
        write_u32_le(&mut f, 1)?; // version
        write_u32_le(&mut f, ds.n as u32)?; // n_rows
        write_u32_le(&mut f, ds.n as u32)?; // n_cols (square for now)
        write_u32_le(&mut f, ds.nnz as u32)?;
        write_u32_le(&mut f, 0)?;
        write_u32_le(&mut f, 0)?;
        write_u32_le(&mut f, 0)?;

        write_u32_slice_le(&mut f, &ds.row_ptr)?;
        write_u32_slice_le(&mut f, &ds.col_idx)?;
        write_f32_slice_le(&mut f, &ds.values)?;
    }

    // rhs.bin: [u32 n][f32[n]]
    {
        let mut f = File::create(out_dir.join("rhs.bin"))
            .map_err(|e| format!("Failed to create rhs.bin: {e}"))?;
        write_u32_le(&mut f, ds.n as u32)?;
        write_f32_slice_le(&mut f, &ds.b)?;
    }

    // x0.bin: [u32 n][f32[n]]
    {
        let mut f = File::create(out_dir.join("x0.bin"))
            .map_err(|e| format!("Failed to create x0.bin: {e}"))?;
        write_u32_le(&mut f, ds.n as u32)?;
        write_f32_slice_le(&mut f, &ds.x0)?;
    }

    // block_starts.bin: [u32 len][u32[len]]
    {
        let mut f = File::create(out_dir.join("block_starts.bin"))
            .map_err(|e| format!("Failed to create block_starts.bin: {e}"))?;
        write_u32_le(&mut f, ds.block_starts.len() as u32)?;
        write_u32_slice_le(&mut f, &ds.block_starts)?;
    }

    // meta.json
    {
        let meta = MetaJson {
            format: "wgpu-solver-backend-text-v0".to_string(),
            n: ds.n,
            nnz: ds.nnz,
            row_ptr_len: ds.row_ptr.len(),
            col_idx_len: ds.col_idx.len(),
            values_len: ds.values.len(),
            rhs_len: ds.b.len(),
            x0_len: ds.x0.len(),
            block_starts_len: ds.block_starts.len(),
        };

        let meta_text = serde_json::to_string_pretty(&meta)
            .map_err(|e| format!("Failed to serialize meta.json: {e}"))?;
        fs::write(out_dir.join("meta.json"), meta_text)
            .map_err(|e| format!("Failed to write meta.json: {e}"))?;
    }

    Ok(())
}

fn write_u32_le<W: Write>(w: &mut W, v: u32) -> Result<(), String> {
    w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
}
fn write_u32_slice_le<W: Write>(w: &mut W, data: &[u32]) -> Result<(), String> {
    for &v in data {
        write_u32_le(w, v)?;
    }
    Ok(())
}
fn write_f32_slice_le<W: Write>(w: &mut W, data: &[f32]) -> Result<(), String> {
    for &v in data {
        w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}
