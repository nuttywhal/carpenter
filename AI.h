#ifndef __AI_H__
#define __AI_H__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "Board.h"
#include "Parameters.h"

/**
 * Check for a CUDA error.
 *
 * error : The error to check.
 */
bool checkForError(cudaError_t error);

/**
 * Check for a kernel error.
 *
 * errorMessage : The error message to print.
 */
bool checkForKernelError(const char *errorMessage);

#endif // __AI_H__
