#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// CPY + CONCAT: src0 of the concat is an F32 tensor cast to BF16, converted while concatenating
void ggml_cuda_op_concat_cast(ggml_backend_cuda_context & ctx, const ggml_tensor * cpy, ggml_tensor * concat);
