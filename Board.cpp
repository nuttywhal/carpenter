#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "AI.h"
#include "Board.h"

// ------------------------------------------------------------------------- //
//  CPU FUNCTIONS
// ------------------------------------------------------------------------- //

void initialize_board(Board *board)
{
    for (int i = 0; i < NUM_HOLES; i++) {
        initialize_hole(&board->holes[i]);
    }
}

void initialize_hole(Hole *hole, bool blocks[HOLE_SIZE][HOLE_SIZE])
{
    if (blocks) {
        // Initialize hole from 2D boolean array.
        for (int i = 0; i < HOLE_SIZE; i++) {
            for (int j = 0; j < HOLE_SIZE; j++) {
                hole->blocks[i][j] = blocks[i][j];
            }
        }
    }

    // Compute the number of pieces needed for a perfect hole.
    int emptyBlocks = 0;
    for (int i = 0; i < HOLE_SIZE; i++) {
        for (int j = 0; j < HOLE_SIZE; j++) {
            if (hole->blocks[i][j]) {
                emptyBlocks++;
            }
        }
    }
    hole->perfectHole = emptyBlocks / 5;

    // Initialize the rest of the struct members.
    hole->isActive = true;
    hole->piecesPlaced = 0;
    hole->lastMove = 0;
}

void free_board(Board *board)
{
    delete board;
}

// ------------------------------------------------------------------------- //
//  GPU FUNCTIONS
// ------------------------------------------------------------------------- //

bool g_initialize_board(Board *d_board, Board *h_board)
{
    // For error checking.
    bool error;

    // Allocate memory on the device for the board.
    error = checkForError(cudaMalloc((void **)&d_board, sizeof(Board)));

    if (error) {
        std::cout << "DEVICE ERROR - Allocating memory on device during "
                  << "board initialization" << std::endl;

        return error;
    }

    // Copy the board from host memory to device memory.
    error = checkForError(cudaMemcpy(d_board,
                                     h_board,
                                     sizeof(Board),
                                     cudaMemcpyHostToDevice));

    if (error) {
        std::cout << "DEVICE ERROR - Copying board to device during "
                  << "board initialization" << std::endl;

        return error;
    }

    return false;
}
