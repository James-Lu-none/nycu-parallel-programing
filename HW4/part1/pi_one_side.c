#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long long int local_tosses = tosses / world_size;
    unsigned int seed = 314581029 * world_rank;
    double x, y;
    int count = 0;
    for (long long int i = 0; i < local_tosses; i++)
    {
        x = rand_r(&seed) / (double)RAND_MAX;
        y = rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0)
        {
            count++;
        }
    }
    double local_pi = 4.0 * count / local_tosses;
    
    double *recv_pis = NULL;

    if (world_rank == 0)
    {
        // Main
        recv_pis = malloc(world_size * sizeof(double));
        // exposes recv_pis as remotely accessible memory in the window win
        MPI_Win_create(recv_pis, world_size * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
        // obtain the window object without exposing any memory
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    MPI_Win_fence(0, win);

    if (world_rank == 0)
    {
        recv_pis[0] = local_pi;
    }
    else
    {
        MPI_Put(&local_pi, 1, MPI_DOUBLE, 0, world_rank, 1, MPI_DOUBLE, win);
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        for (int i = 0; i < world_size; i++)
            pi_result += recv_pis[i];
        pi_result /= world_size;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
