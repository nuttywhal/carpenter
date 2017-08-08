#ifndef __BOARD_H__
#define __BOARD_H__

// At any given time, there can be up to four active holes.
#define NUM_HOLES (4)

// Length of a hole. All holes can fit in a 10 x 10 block grid.
#define HOLE_SIZE (10)

// ------------------------------------------------------------------------- //
//  SHARED STRUCTS
// ------------------------------------------------------------------------- //

typedef struct Hole
{
    bool blocks[HOLE_SIZE][HOLE_SIZE]; // A block in the hole.
    bool isActive;                     // Is this hole being worked on?
    int piecesPlaced;                  // Number of pieces placed so far.
    int perfectHole;                   // Number of pieces needed for "A Masterpiece!"
    int lastMove;                      // Number of moves since last worked on.
} Hole;

typedef struct Board
{
    Hole holes[NUM_HOLES]; // Array of holes.
} Board;

// ------------------------------------------------------------------------- //
//  CPU FUNCTIONS
// ------------------------------------------------------------------------- //

/**
 * Initialize a Board struct in host memory.
 *
 * board : The board to initialize.
 */
void initialize_board(Board *board);

/**
 * Initialize a Hole struct in host memory.
 *
 * board  : The hole to initialize.
 * blocks : 2D boolean array to initialize the hole to.
 */
void initialize_hole(Hole *hole, bool blocks[HOLE_SIZE][HOLE_SIZE] = nullptr);

/**
 * Free a board from host memory.
 *
 * board : The board to delete.
 */
void free_board(Board *board);

// ------------------------------------------------------------------------- //
//  GPU FUNCTIONS
// ------------------------------------------------------------------------- //

/**
 * Initialize a Board struct in device memory.
 *
 * d_board : The board to initialize on the device.
 * h_board : The host board to copy from.
 */
bool g_initialize_board(Board *d_board, Board *h_board);

#endif // __BOARD_H__
