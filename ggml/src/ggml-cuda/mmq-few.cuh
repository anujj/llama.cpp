#pragma once

#include "common.cuh"

// Min. number of src1 columns from which the few-row MMQ replaces MMVQ, 0 if it is not used for this type, GPU and row length.
int ggml_cuda_mmq_few_min_ncols(ggml_type type, int cc, int64_t ne00);

// Whether to use the few-row MMQ kernel: int8 tensor core MMA for 2 to 24 src1 columns.
bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, int cc);

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
