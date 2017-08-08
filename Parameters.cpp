#include <chrono>
#include <iostream>
#include <functional>
#include <random>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "AI.h"
#include "Parameters.h"

// ------------------------------------------------------------------------- //
//  CPU FUNCTIONS
// ------------------------------------------------------------------------- //

void initialize_population(Parameters *population, int populationSize)
{
    // Seed random number generator.
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto rgen = std::bind(std::uniform_real_distribution<>(-1, 1), std::mt19937((unsigned)seed));

    // Initialize the population.
    for (int i = 0; i < populationSize; i++) {
        population[i].jaggedness = rgen();
        population[i].neglection = rgen();
        population[i].holeCount  = rgen();
    }
}

void free_population(Parameters *population, int populationSize)
{
    delete[] population;
}

// ------------------------------------------------------------------------- //
//  GPU FUNCTIONS
// ------------------------------------------------------------------------- //

bool g_initialize_population(Parameters *d_population, Parameters *h_population, int populationSize)
{
    // For error checking.
    bool error;

    // Allocate memory on the device for the initial population.
    error = checkForError(cudaMalloc((void **)&d_population,
                                     sizeof(Parameters) * populationSize));

    if (error) {
        std::cout << "DEVICE ERROR - Allocating memory on device during "
                  << "population initialization" << std::endl;

        return error;
    }

    // Copy the initial population from host memory to device memory.
    error = checkForError(cudaMemcpy(d_population,
                                     h_population,
                                     sizeof(Parameters) * populationSize,
                                     cudaMemcpyHostToDevice));

    if (error) {
        std::cout << "DEVICE ERROR - Copying population to device during "
                  << "population initialization" << std::endl;

        return error;
    }

    return false;
}
