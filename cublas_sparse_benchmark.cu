#include <cuda_runtime.h>
#include <cublasLt.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>

// Initialize random values for matrices
void initializeMatrix(std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Convert a dense matrix to 2-out-of-4 sparsity format
void convertTo2OutOf4Sparsity(const std::vector<float>& dense, std::vector<float>& sparse, int rows, int cols) {
    sparse.resize(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j += 4) {
            // Retain the two largest values in every 4-element block
            // std::array<int, 4> indices = {j, j + 1, j + 2, j + 3}; 
            std::vector<int> indices = {j, j + 1, j + 2, j + 3};
            // std::partial_sort(indices.begin(), indices.begin() + 2, indices.end(),
            //                   [&dense, i, cols](int lhs, int rhs) {
            //                       return std::abs(dense[i * cols + lhs]) > std::abs(dense[i * cols + rhs]); 
            //                   }); 
            for (int a = 0; a < 2; ++a) {
                for (int b = a + 1; b < 4; ++b) {
                    if (std::abs(dense[i * cols + indices[b]]) > std::abs(dense[i * cols + indices[a]])) {
                        std::swap(indices[a], indices[b]);
                    }
                }
            } 

            for (int k = 0; k < 4; ++k) {
                if (k == indices[0] % 4 || k == indices[1] % 4) {
                    sparse[i * cols + j + k] = dense[i * cols + j + k];
                } else {
                    sparse[i * cols + j + k] = 0.0f;
                }
            }
        }
    }
}

// Perform sparse matrix multiplication using cuBLASLt
void runSparseMatmul(int m, int n, int k) {
    // Host matrices
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C(m * n);
    std::vector<float> h_sparse_A;

    // Initialize matrices
    initializeMatrix(h_A, m, k);
    initializeMatrix(h_B, k, n);

    // Convert A to 2-out-of-4 sparse format
    convertTo2OutOf4Sparsity(h_A, h_sparse_A, m, k);

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, h_sparse_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(float));

    // Create cuBLASLt handle
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    // Define matrix layouts
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, m, k, m); // Leading dimension = m
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, k, n, k); // Leading dimension = k
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, m, n, m); // Leading dimension = m

    // Define matmul operation
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F); 

    // Define algorithm descriptor
    cublasLtMatmulAlgo_t algo;
    // cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, layoutA, layoutB, layoutC, layoutC, &algo); 
    // cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, m, n, k, &algo); 
    cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, layoutA, layoutB, layoutC, layoutC, &algo);

    // Perform matrix multiplication
    const float alpha = 1.0f, beta = 0.0f; 

    size_t workspace_size = 1024 * 1024 * 8; // Example: 8 MB
    void* workspace;
    cudaMalloc(&workspace, workspace_size); 

    auto start = std::chrono::high_resolution_clock::now(); 

    cublasLtMatmul(
        handle, 
        matmulDesc,
        &alpha, 
        d_A, layoutA,  // Sparse A
        d_B, layoutB,          // Dense B
        &beta, 
        d_C, layoutC,   // Output C 
        d_C, layoutC,   // Output C 
        &algo, workspace, workspace_size, nullptr  // Algo, workspace, and stream 
    ); 

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Sparse matrix multiplication (m=" << m << ", n=" << n << ", k=" << k
              << ") took " << elapsed.count() << " seconds." << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); 
    cudaFree(workspace); 
}

int main() {
    int m = 1024, n = 1024, k = 1024;
    std::cout << "Starting cuBLASLt sparse matrix multiplication benchmark..." << std::endl;
    runSparseMatmul(m, n, k);
    return 0;
}
