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

    // TODO: binary tree redunction
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
    double recv_buf;
    int partner;

    // tree reduction
    // log2(world_size) steps
    for (int step = 1; step < world_size; step <<= 1)
    {
        if ((world_rank % (step << 1)) == 0)
        {
            // this rank receives local_pi from partner = rank + step, if exists
            partner = world_rank + step;
            if (partner < world_size)
            {
                MPI_Recv(&recv_buf, 1, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_pi += recv_buf;
            }
        }
        else
        {
            // send local_pi to rank - step and exit loop since has nothing more to do
            partner = world_rank - step;
            MPI_Send(&local_pi, 1, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);
            break;
        }
    }
    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = local_pi / world_size;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
