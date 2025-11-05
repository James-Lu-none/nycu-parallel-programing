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

    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long long int local_tosses = tosses / world_size;
    unsigned int seed = time(NULL) * world_rank;
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
    if (world_rank > 0)
    {
        // TODO: handle workers
        MPI_Send(&local_pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: main
        pi_result = local_pi;
        for (int i = 1; i < world_size; i++)
        {
            double local_pi;
            MPI_Recv(&local_pi, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pi_result += local_pi;    
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
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
