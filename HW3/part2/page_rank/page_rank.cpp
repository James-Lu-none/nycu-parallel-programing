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

    double *score_old = solution;
    double *score_new = new double[nnodes];
    double *outgoing_sizes = new double[nnodes];
    const Vertex **incoming_begins = new const Vertex *[nnodes];
    const Vertex **incoming_ends = new const Vertex *[nnodes];
    double *out_contrib = new double[nnodes];

    #pragma omp parallel for
    for (int v = 0; v < nnodes; ++v)
    {
        outgoing_sizes[v] = outgoing_size(g, v);
        incoming_begins[v] = incoming_begin(g, v);
        incoming_ends[v] = incoming_end(g, v);
    }

    double global_diff = 1;
    while (global_diff > convergence)
    {        
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int v = 0; v < nnodes; ++v){
            if (outgoing_sizes[v] == 0){
                dangling_sum += score_old[v];
            }
        }
        const double base_score = (1.0 - damping) / nnodes + damping * dangling_sum / nnodes;

        
        #pragma omp parallel for
        for (int v = 0; v < nnodes; ++v)
            out_contrib[v] = (outgoing_sizes[v] > 0) ? score_old[v] / outgoing_sizes[v] : 0.0;
        #pragma omp parallel for
        for (int v = 0; v < nnodes; ++v)
        {
            double sum = 0.0;
            for (const Vertex *u = incoming_begins[v]; u != incoming_ends[v]; ++u)
                sum += out_contrib[*u];

            score_new[v] = base_score + damping * sum;
        }

        global_diff = 0.0;
        #pragma omp parallel for reduction(+:global_diff)
        for (int vi = 0; vi < nnodes; ++vi)
        {
            global_diff += fabs(score_new[vi] - score_old[vi]);
            score_old[vi] = score_new[vi];
        }
    }
    delete[] score_new;
    delete[] outgoing_sizes;
    delete[] incoming_begins;
    delete[] incoming_ends;
    delete[] out_contrib;
}