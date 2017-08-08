#include <iostream>

#include "AI.h"
#include "Board.h"
#include "Parameters.h"

int main()
{
    std::cout << "Nuttywhal's Carpentry Bot" << std::endl;

    // 1. Initialize board and copy it to the device.

    Board *h_board = new Board;
    initialize_board(h_board);

    Board *d_board = nullptr;
    g_initialize_board(d_board, h_board);

    // 2. Initialize initial population of parameters and copy
    //    it to the device.

    const unsigned int populationSize = 100;
    Parameters *h_population = new Parameters[populationSize];
    initialize_population(h_population, populationSize);

    Parameters *d_population = nullptr;
    g_initialize_population(d_population, h_population, populationSize);

    // 3. Execute Genetic Algorithm.

        // -----------------------
        //  TODO: Implement.
        // -----------------------

    // 4. Free allocated memory.

    free_population(h_population, populationSize);
    free_board(h_board);
    cudaFree(d_board);
    
    return 0;
}
