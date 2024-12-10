#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h> // For __half type
#include <cusparseLt.h>

// Define matrix dimensions and parameters
#define NUM_A_ROWS 128 // Number of rows in matrix A
#define NUM_A_COLS 128 // Number of columns in matrix A
#define NUM_B_ROWS 128 // Number of rows in matrix B
#define NUM_B_COLS 128 // Number of columns in matrix B
#define NUM_C_ROWS NUM_A_ROWS // Number of rows in matrix C
#define NUM_C_COLS NUM_B_COLS // Number of columns in matrix C
#define LDA NUM_A_COLS        // Leading dimension of matrix A
#define LDB NUM_B_COLS        // Leading dimension of matrix B
#define LDC NUM_C_COLS        // Leading dimension of matrix C
#define ALIGNMENT 16          // Memory alignment

int main() {
    // Scalars for multiplication
    float alpha = 1.0f;
    float beta = 0.0f;

    // CUDA stream
    cudaStream_t stream = nullptr;

    // Create cuSPARSELt handle
    cusparseLtHandle_t handle;
    cusparseLtInit(&handle);

    // Define matrix data types
    cudaDataType type = CUDA_R_16F; // Half precision (__half)
    cusparseOrder_t order = CUSPARSE_ORDER_ROW; // Row-major order
    cusparseComputeType compute_type = CUSPARSE_COMPUTE_32F; // Compute in single precision

    // Allocate memory for matrices on the device
    __half *dA, *dB, *dC, *dD;
    cudaMalloc(&dA, NUM_A_ROWS * NUM_A_COLS * sizeof(__half));
    cudaMalloc(&dB, NUM_B_ROWS * NUM_B_COLS * sizeof(__half));
    cudaMalloc(&dC, NUM_C_ROWS * NUM_C_COLS * sizeof(__half));
    cudaMalloc(&dD, NUM_C_ROWS * NUM_C_COLS * sizeof(__half));

    // Fill matrices with random data (for simplicity, using memset here)
    cudaMemset(dA, 1, NUM_A_ROWS * NUM_A_COLS * sizeof(__half));
    cudaMemset(dB, 1, NUM_B_ROWS * NUM_B_COLS * sizeof(__half));
    cudaMemset(dC, 0, NUM_C_ROWS * NUM_C_COLS * sizeof(__half));

    // Initialize matrix descriptors
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtStructuredDescriptorInit(&handle, &matA, NUM_A_ROWS, NUM_A_COLS, LDA, ALIGNMENT, type, order, CUSPARSELT_SPARSITY_50_PERCENT);
    cusparseLtDenseDescriptorInit(&handle, &matB, NUM_B_ROWS, NUM_B_COLS, LDB, ALIGNMENT, type, order);
    cusparseLtDenseDescriptorInit(&handle, &matC, NUM_C_ROWS, NUM_C_COLS, LDC, ALIGNMENT, type, order);

    // Initialize matmul descriptor and plan
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cusparseLtMatmulDescriptorInit(&handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &matA, &matB, &matC, &matC, compute_type);
    cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);

    // Prune matrix A
    cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream);

    // Check pruning correctness
    int *d_valid;
    cudaMalloc((void **)&d_valid, sizeof(int));
    cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream);

    int is_valid;
    cudaMemcpyAsync(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (is_valid != 0) {
        std::cerr << "Pruned matrix A is invalid!" << std::endl;
        return EXIT_FAILURE;
    }

    // Compress matrix A
    size_t compressed_size;
    cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size);
    void *dA_compressed;
    cudaMalloc(&dA_compressed, compressed_size);
    cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream);

    // Allocate workspace
    size_t workspace_size;
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
    void *d_workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&d_workspace, workspace_size);
    }

    // Perform matrix multiplication
    cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, &stream, 0);

    // Cleanup
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtDestroy(&handle);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);
    cudaFree(dA_compressed);
    cudaFree(d_workspace);
    cudaFree(d_valid);

    std::cout << "Matrix multiplication completed successfully!" << std::endl;
    return EXIT_SUCCESS;
} 
