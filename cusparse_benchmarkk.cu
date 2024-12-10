#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <vector>
#include <iostream>
#include <chrono>

// Function to initialize matrices with random values
void initializeMatrix(std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Sparse Matrix Multiplication
void runSparseMatmul(int m, int n, int k) {
    // Host matrices
    std::vector<float> h_A(m * k); // Dense matrix A
    std::vector<float> h_B(k * n); // Dense matrix B
    std::vector<float> h_C(m * n); // Result matrix C

    // Initialize matrices with random values
    initializeMatrix(h_A, m, k);
    initializeMatrix(h_B, k, n);

    // Device matrices
    float *d_A, *d_B, *d_C, *d_A_compressed;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(float));

    // cuSPARSELt setup
    cusparseLtHandle_t handle;
    cusparseLtInit(&handle);

    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t compressed_size, compress_buffer_size;
    void* compress_buffer = nullptr;

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Descriptors - Changed to CUDA_R_32F
    cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, k, 16, CUDA_R_32F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT);
    cusparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Matmul descriptor
    cusparseLtMatmulDescriptorInit(&handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &matA, &matB, &matC, &matC, CUSPARSE_COMPUTE_32F);
    cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);

    float elapsed_time_ms = 0.0f;

    // Prune and compress
    cusparseLtSpMMAPrune(&handle, &matmul, d_A, d_A, CUSPARSELT_PRUNE_SPMMA_TILE, stream);
    cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compress_buffer_size);

    cudaMalloc(&d_A_compressed, compressed_size);
    cudaMalloc(&compress_buffer, compress_buffer_size);
    
    cusparseStatus_t compress_status = cusparseLtSpMMACompress(
        &handle,
        &plan,
        d_A,
        d_A_compressed,
        compress_buffer,
        stream
    );

    // Workspace
    size_t workspace_size;
    void* d_workspace = nullptr;
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
    if (workspace_size > 0) {
        cudaMalloc(&d_workspace, workspace_size);
    }
    float alpha = 1.0f, beta = 0.0f;

    int num_iterations = 1000;
    for (int i = 0; i < 10; ++i) {
        cusparseLtMatmul(&handle, &plan, &alpha, d_A_compressed, d_B, &beta, d_C, d_C, d_workspace, &stream, 0);
    }

    cudaEventRecord(start, stream);

    for (int i = 0; i < num_iterations; ++i) {
        cusparseLtMatmul(&handle, &plan, &alpha, d_A_compressed, d_B, &beta, d_C, d_C, d_workspace, &stream, 0);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    std::cout << "Elapsed time: " << elapsed_time_ms << " ms" << std::endl;
    double avg_time_per_iteration = elapsed_time_ms / num_iterations;

    std::cout << "Sparse matrix multiplication (m=" << m << ", n=" << n << ", k=" << k
          << ") average runtime over " << num_iterations << " iterations: "
          << avg_time_per_iteration << " ms" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_compressed);
    cudaFree(compress_buffer);
    if (workspace_size > 0) {
        cudaFree(d_workspace);
    }

    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtDestroy(&handle);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

int main() {
    // int m = 1024, n = 1024, k = 1024; 
    // int m = 128, n = 128, k = 128; 
    // int m = 256, n = 256, k = 256; 
    // int m = 512, n = 512, k = 512; 
    int m = 1024, n = 1024, k = 1024; 

    std::cout << "Starting cuSPARSELt sparse matrix multiplication benchmark..." << std::endl;
    runSparseMatmul(m, n, k);

    return 0;
} 
