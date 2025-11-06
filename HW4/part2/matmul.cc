#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

int *rows;
int *numbers_A;
int *offsets_A;
int *numbers_C;
int *offsets_C;

int *local_A;
int *local_BT;
int *local_C;

int local_rows;

int world_rank, world_size;

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

    // dont use them at all
    *a_mat_ptr = nullptr;
    *b_mat_ptr = nullptr;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int base = n / world_size;
    int rem = n % world_size;

    // calculate numbers and offsets of A and C for each process
    rows = new int[world_size];
    numbers_A = new int[world_size];
    offsets_A = new int[world_size];
    numbers_C = new int[world_size];
    offsets_C = new int[world_size];

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
    }

    local_rows = rows[world_rank];

    // Allocate local_BT for non-root ranks since *bt_mat is const and cannot be used directly
    if (world_rank == 0)
    {
        // there is a trap here, in main.cc b_mat is loaded incorrectly as a m x l matrix, which is already transposed
        local_BT = (int *)b_mat;
    }
    else
    {
        local_BT = new int[l * m];
    }

    local_A = new int[local_rows * m];
    local_C = new int[local_rows * l];

    // send A's rows and broadcast B^T to all processes since they dont have them
    MPI_Scatterv(a_mat, numbers_A, offsets_A, MPI_INT, local_A, numbers_A[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    // broadcast BT since all processes need entire B table regardless the number of rows assigned
    MPI_Bcast(local_BT, l * m, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *bt_mat, int *out_mat)
{
    /* TODO: Perform matrix multiplication on a_mat and b_mat. Which are the matrices you've
     * constructed. The result should be stored in out_mat, which is a continuous memory placing n *
     * l elements of int. You need to make sure rank 0 receives the result.
     */

    for (int i = 0; i < local_rows; ++i)
    {
        for (int j = 0; j < l; ++j)
        {
            int sum = 0;
            for (int k = 0; k < m; ++k)
            {
                sum += local_A[i * m + k] * local_BT[j * m + k];
            }
            local_C[i * l + j] = sum;
        }
    }

    MPI_Gatherv(local_C, numbers_C[world_rank], MPI_INT, out_mat, numbers_C, offsets_C, MPI_INT, 0, MPI_COMM_WORLD);
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    /* TODO */
    delete[] a_mat;
    delete[] b_mat;

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
