#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;
    }

    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/nnodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / nnodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / nnodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }


     */
    /*
        Another basic page rank pseudocode that is equivalent but more efficient:
        1. For each vj, distribute score_old[vj]/outdegree(vj) to its outgoing neighbors
        2. After distribution, score_new[*i] += damping * contrib; base_score = (1-d)/N + d*dangling/N
        3. Precompute total dangling mass, then apply uniformly
     */
    double *score_old = solution;
    double *score_new = new double[nnodes];
    double global_diff = 1;
    while (global_diff > convergence)
    {
        for (int i = 0; i < nnodes; ++i)
            score_new[i] = 0.0;
        
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int v = 0; v < nnodes; ++v)
            if (outgoing_size(g, v) == 0)
                dangling_sum += score_old[v];

        for (int vj = 0; vj < nnodes; ++vj)
        {
            int out_size = outgoing_size(g, vj);
            if (out_size == 0)
                continue;
            double contrib = score_old[vj] / out_size;
            for (const Vertex *i = outgoing_begin(g, vj); i != outgoing_end(g, vj); ++i)
                score_new[*i] += damping * contrib;
        }

        double base_score = ((1.0 - damping) + damping * dangling_sum) / nnodes;
        #pragma omp parallel for
        for (int vi = 0; vi < nnodes; ++vi)
            score_new[vi] += base_score;
        
        global_diff = 0.0;
        for (int vi = 0; vi < nnodes; ++vi){
            global_diff += fabs(score_new[vi] - score_old[vi]);
            score_old[vi] = score_new[vi];
        }
    }
    delete[] score_new;
}