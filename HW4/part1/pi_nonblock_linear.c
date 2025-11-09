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

    if (world_rank == 0)
    {
        double *recv_pis = malloc(sizeof(double) * world_size);
        MPI_Request *requests = malloc(sizeof(MPI_Request) * world_size);
        
        for (int i = 1; i < world_size; i++)
        {
            MPI_Irecv(&recv_pis[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requests[i]);
        }

        MPI_Waitall(world_size - 1, &requests[1], MPI_STATUSES_IGNORE);

        for (int i = 1; i < world_size; i++)
        {
            local_pi += recv_pis[i];
        }

        pi_result = local_pi / world_size;

        free(recv_pis);
        free(requests);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    else
    {
        MPI_Send(&local_pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
