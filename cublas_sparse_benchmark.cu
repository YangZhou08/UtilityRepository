#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <vector>
#include <iostream>
#include <chrono>

// Function to initialize matrices with random values
void initializeMatrix(std::vector<__half>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Sparse Matrix Multiplication
void runSparseMatmul(int m, int n, int k) {
    // Host matrices
    std::vector<__half> h_A(m * k); // Dense matrix A
    std::vector<__half> h_B(k * n); // Dense matrix B
    std::vector<__half> h_C(m * n); // Result matrix C

    // Initialize matrices with random values
    initializeMatrix(h_A, m, k);
    initializeMatrix(h_B, k, n);

    // Device matrices
    __half *d_A, *d_B, *d_C, *d_A_compressed;
    cudaMalloc(&d_A, m * k * sizeof(__half));
    cudaMalloc(&d_B, k * n * sizeof(__half));
    cudaMalloc(&d_C, m * n * sizeof(__half));

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(__half));

    // cuSPARSELt setup
    cusparseLtHandle_t handle;
    cusparseLtInit(&handle);

    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;

    cudaStream_t stream = nullptr;
    size_t compressed_size, compress_buffer_size;
    void* compress_buffer = nullptr;

    // Descriptors
    cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, k, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT);
    cusparseLtDenseDescriptorInit(&handle, &matB, k, n, n, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW);
    cusparseLtDenseDescriptorInit(&handle, &matC, m, n, n, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW);

    // Matmul descriptor
    cusparseLtMatmulDescriptorInit(&handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &matA, &matB, &matC, &matC, CUSPARSE_COMPUTE_32F);
    cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);

    // Prune and compress
    cusparseLtSpMMAPrune(&handle, &matmul, d_A, d_A, CUSPARSELT_PRUNE_SPMMA_TILE, stream);
    cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compress_buffer_size);

    cudaMalloc(&d_A_compressed, compressed_size);
    cudaMalloc(&compress_buffer, compress_buffer_size);
    cusparseLtSpMMACompress(&handle, &plan, &matA, d_A, d_A_compressed, compress_buffer, stream);

    // Workspace
    size_t workspace_size;
    void* d_workspace = nullptr;
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
    if (workspace_size > 0) {
        cudaMalloc(&d_workspace, workspace_size);
    } 

    int num_iterations = 1000; 

    // Timer
    auto start = std::chrono::high_resolution_clock::now();

    // Matrix multiplication
    float alpha = 1.0f, beta = 0.0f; 
    for (int i = 0; i < num_iterations; ++i) {
        cusparseLtMatmul(&handle, &plan, &alpha, d_A_compressed, d_B, &beta, d_C, d_C, d_workspace, nullptr, 0); 
        cudaDeviceSynchronize();
    } 

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start; 

    // Calculate average runtime
    double avg_time_per_iteration = elapsed.count() / num_iterations; 

    std::cout << "Sparse matrix multiplication (m=" << m << ", n=" << n << ", k=" << k
          << ") average runtime over " << num_iterations << " iterations: "
          << avg_time_per_iteration << " seconds." << std::endl; 

    // std::cout << "Sparse matrix multiplication (m=" << m << ", n=" << n << ", k=" << k
    //           << ") took " << elapsed.count() << " seconds." << std::endl; 

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
}

int main() {
    // Example: Multiply two 1024 x 1024 matrices
    int m = 1024, n = 1024, k = 1024;

    std::cout << "Starting cuSPARSELt sparse matrix multiplication benchmark..." << std::endl;
    runSparseMatmul(m, n, k);

    return 0;
} 
