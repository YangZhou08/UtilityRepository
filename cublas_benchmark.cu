#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>

// Initialize random values for matrices
void initializeMatrix(std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to perform cuBLAS matrix multiplication
void runCuBLAS(int m, int n, int k) {
    // Host matrices
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C(m * n);

    // Initialize matrices with random values
    initializeMatrix(h_A, m, k);
    initializeMatrix(h_B, k, n);

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(float));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication
    const float alpha = 1.0f, beta = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,             // Matrix dimensions
                &alpha,              // Alpha
                d_A, m,              // Matrix A and leading dimension
                d_B, k,              // Matrix B and leading dimension
                &beta,               // Beta
                d_C, m);             // Matrix C and leading dimension

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix multiplication (m=" << m << ", n=" << n << ", k=" << k
              << ") took " << elapsed.count() << " seconds." << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

int main() {
    // Example: Multiply two 1024 x 1024 matrices
    int m = 1024, n = 1024, k = 1024;

    std::cout << "Starting cuBLAS matrix multiplication..." << std::endl;
    runCuBLAS(m, n, k);

    return 0;
} 