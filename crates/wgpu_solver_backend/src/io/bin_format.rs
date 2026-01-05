#[derive(Debug, Clone)]
pub struct CsrMatrixBin {
    pub n_rows: u32,
    pub n_cols: u32,
    pub nnz: u32,
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct VectorBin {
    pub n: u32,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct BlockStartsBin {
    pub starts: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct CaseInputBin {
    pub a: CsrMatrixBin,
    pub b: VectorBin,
    pub x0: VectorBin,
    pub block_starts: BlockStartsBin,
}
