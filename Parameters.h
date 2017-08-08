#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

// ------------------------------------------------------------------------- //
//  SHARED STRUCTS
// ------------------------------------------------------------------------- //

typedef struct Parameters
{
    double jaggedness; // How many vertices do the holes have?
    double neglection; // How long has it been since the holes have been attended to?
    double holeCount;  // How many holes have the holes been split into?
} Parameters;

// ------------------------------------------------------------------------- //
//  CPU FUNCTIONS
// ------------------------------------------------------------------------- //

/**
 * Initialize a population of parameters in host memory.
 *
 * population     : The population of parameters to initialize.
 * populationSize : The size of the population.
 */
void initialize_population(Parameters *population, int populationSize);

/**
 * Free a population of parameters from host memory.
 *
 * population     : The population of parameters to delete.
 * populationSize : The size of the population.
 */
void free_population(Parameters *population, int populationSize);

// ------------------------------------------------------------------------- //
//  GPU FUNCTIONS
// ------------------------------------------------------------------------- //

/**
 * Initialize a population of parameters in device memory.
 *
 * d_population   : The population to initialize on the device.
 * h_population   : The host population to copy from.
 * populationSize : The size of the population.
 */
bool g_initialize_population(Parameters *d_population,
                             Parameters *h_population,
                             int populationSize);

#endif // __PARAMETERS_H__
