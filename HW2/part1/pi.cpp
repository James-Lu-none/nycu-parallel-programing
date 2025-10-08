#include <cmath>
#include <cstdio>
#include <getopt.h>


void usage(char *progname) {
    fprintf(stderr, "Usage: %s <threads> <n>\n", progname);
    fprintf(stderr, "  <threads> : number of threads to use (positive integer)\n");
    fprintf(stderr, "  <n>       : number of intervals (positive integer)\n");
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    char *endptr;
    int threads = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || threads <= 0) {
        fprintf(stderr, "Invalid value for threads: %s\n", argv[1]);
        usage(argv[0]);
        return 1;
    }

    long long int n = strtoll(argv[2], &endptr, 10);
    if (*endptr != '\0') {
        fprintf(stderr, "Invalid value for n: %s\n", argv[2]);
        usage(argv[0]);
        return 1;
    }

    long long int hits = 0;
    

    for (long long int i = 0; i < n; ++i) {
        float x = rand() / (float)RAND_MAX;
        float y = rand() / (float)RAND_MAX;
        if (x * x + y * y <= 1.0f) {
            hits++;
        }
    }

    double pi = hits * 4.0 / n;
    printf("%lf\n", pi);
    return 0;
}