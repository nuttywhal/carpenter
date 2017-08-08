#include <iostream>

#include "AI.h"

bool checkForError(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::cout << cudaGetErrorString(error) << std::endl;
    }

    return error != cudaSuccess;
}

bool checkForKernelError(const char *errorMessage)
{
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cout << errorMessage << cudaGetErrorString(status) << std::endl;
    }

    return status != cudaSuccess;
}
