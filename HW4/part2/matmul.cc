#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <string.h>


void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    /* TODO: The data is stored in a_mat and b_mat.
     * You need to allocate memory for a_mat_ptr and b_mat_ptr,
     * and copy the data from a_mat and b_mat to a_mat_ptr and b_mat_ptr, respectively.
     * You can use any size and layout you want if they provide better performance.
     * Unambitiously copying the data is also acceptable.
     *
     * The matrix multiplication will be performed on a_mat_ptr and b_mat_ptr.
     */
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank != 0)
    {
        // Non-root ranks can't construct matrices since *a_mat, *b_mat are only available in rank 0
        *a_mat_ptr = nullptr;
        *b_mat_ptr = nullptr;
        return;
    }
    int *A = new int[n * m];
    memcpy(A, a_mat, n * m * sizeof(int));

    // there is a trap here, in main.cc b_mat is loaded incorrectly as a m x l matrix, which is already transposed
    int *BT = new int[l * m];
    memcpy(BT, b_mat, m * l * sizeof(int));

    *a_mat_ptr = A;
    *b_mat_ptr = BT;
}


#define BK 64
#define BJ 64

static void local_block_gemm_tiled(const int *__restrict Ablk, // local_n x m
                                   const int *__restrict BT,            // l x m  (B transposed)
                                   int *__restrict Cblk,                // local_n x l
                                   int local_n,
                                   int m,
                                   int l)
{
    // Cblk[i, j] = sum_k Ablk[i, k] * B[k, j]
    // Using BT so we access BT[j, k] contiguous along k.
    // Tile on j (columns of C) and k (reduction) to keep cache-friendly.

    for (int j0 = 0; j0 < l; j0 += BJ)
    {
        int jmax = (j0 + BJ < l) ? (j0 + BJ) : l;
        for (int k0 = 0; k0 < m; k0 += BK)
        {
            int kmax = (k0 + BK < m) ? (k0 + BK) : m;
            for (int i = 0; i < local_n; ++i)
            {
                const int *Ai = Ablk + (size_t)i * (size_t)m + (size_t)k0;
                int *Ci = Cblk + (size_t)i * (size_t)l + (size_t)j0;
                for (int j = j0; j < jmax; ++j)
                {
                    const int *BTj = BT + (size_t)j * (size_t)m + (size_t)k0; // row j of BT
                    int acc = 0;
                    for (int k = kmax - k0 - 1; k >= 0; --k)
                    {
                        acc += Ai[k] * BTj[k];
                    }
                    Ci[j - j0] += acc;
                }
            }
        }
    }
}
void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *bt_mat, int *out_mat)
{
    /* TODO: Perform matrix multiplication on a_mat and b_mat. Which are the matrices you've
     * constructed. The result should be stored in out_mat, which is a continuous memory placing n *
     * l elements of int. You need to make sure rank 0 receives the result.
     */
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int base = n / world_size;
    int rem = n % world_size;

    // numbers and offsets of A and C for each process
    int *rows = new int[world_size];
    int *numbers_A = new int[world_size];
    int *offsets_A = new int[world_size];
    int *numbers_C = new int[world_size];
    int *offsets_C = new int[world_size];

    int offset_A_tmp = 0, offset_C_tmp = 0;
    for (int r = 0; r < world_size; ++r)
    {
        rows[r] = base + (r < rem ? 1 : 0);

        numbers_A[r] = rows[r] * m;
        offsets_A[r] = offset_A_tmp;
        offset_A_tmp += numbers_A[r];

        numbers_C[r] = rows[r] * l;
        offsets_C[r] = offset_C_tmp;
        offset_C_tmp += numbers_C[r];
        // printf("rank %d: rows=%d, numbers_A=%d, offset_A=%d, numbers_C=%d, offset_C=%d\n",
        //        r, rows[r], numbers_A[r], offsets_A[r], numbers_C[r], offsets_C[r]);
    }

    const int local_rows = rows[world_rank];

    int *local_BT = NULL;
    // Allocate local_BT for non-root ranks since *bt_mat is const and cannot be used directly
    if (world_rank == 0)
    {
        local_BT = (int *)bt_mat;
    }
    else
    {
        local_BT = new int[l * m];
    }

    int *local_A = new int[local_rows * m];
    int *local_C = new int[local_rows * l](); // initialize to zero

    // send A's rows and broadcast B^T to all processes since they dont have them
    MPI_Scatterv(a_mat, numbers_A, offsets_A, MPI_INT, local_A, numbers_A[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    // broadcast BT since all processes need entire B table regardless the number of rows assigned
    MPI_Bcast(local_BT, l * m, MPI_INT, 0, MPI_COMM_WORLD);

    local_block_gemm_tiled(local_A, local_BT, local_C, local_rows, m, l);

    MPI_Gatherv(local_C, numbers_C[world_rank], MPI_INT, out_mat, numbers_C, offsets_C, MPI_INT, 0, MPI_COMM_WORLD);

    delete[] local_A;
    delete[] local_C;
    delete[] rows;
    delete[] numbers_A;
    delete[] offsets_A;
    delete[] numbers_C;
    delete[] offsets_C;
    if (world_rank != 0)
    {
        delete[] local_BT;
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    /* TODO */
    delete[] a_mat;
    delete[] b_mat;
}
