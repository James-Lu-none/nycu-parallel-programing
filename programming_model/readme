# programming models

## programming model 1: shared memory model

segments that is shared:
text segment (code)
data segment (global variable)
heap segment (dynamic allocate memory)
stack segment (local variable)

### case that local variable is shared

if local variable address is passed to other thread, it is shared

### race condition / data race

two processors access the same variable, and at least one does a write

how to prevent? use lock which OS knowns

### POSIX (portable operating system interface for Unix) threads

1. pthread provide many api like
pthread_create()
pthread_join()
2. pthread is a library, so developer need to do thread management
3. pthread is low level, so developer need to do more work

### openMP (open multi-processing)

1. compiler does the job of parallelization
2. developer hint the compiler with compiler directive to tell which part to parallelize
3. developer need to tell the compiler which variable is shared or private and provide enough synchronization information
4. compiler generate the thread management code

### openMP execution model

Fork-Join parallelism
1. master thread starts
2. master thread encounter parallel directive
3. master thread creates a team of threads
4. each thread executes a portion of the code
5. master thread waits for all threads to complete

parallel region can be nested: nested parallel region
but nested parallel region isn't used often because of overhead of creating threads, we often would like to have threads matches cpu core number.

## programming model 2: distributed memory model

### MPI (message passing interface)

1. process number usually fixed on create
2. communication between processes is done via message passing (send/receive message must be explicit written by developer)
3. coordination is implicit (no lock, no semaphore)
4. each process has its own memory space
5. no shared memory (no shared data)

if sending and receiving communication operation is blocking, deadlock may occur

ex: process 0 send to process 1, process 1 send to process 0
if both process send first, both process will wait forever

possible solution: 
1. process 0 send first, process 1 receive first
2. use non-blocking send/receive

pros and cons of mpi standard
pros
1. portable
cons
1. no other communication mechanism

### when to use shared memory model or distributed memory model

hardware
1. shared memory model: multi-core cpu
2. distributed memory model: cluster

but cluster can also use shared memory model by using memory virtualization technology


## programming model 3: data parallel

1. sigle thread of control consisting of parallel operations on data
2. not all problem can be expressed in data parallel model

### GPU and cloud

CUDA (compute unified device architecture)
1. nvidia's parallel computing architecture
2. use c/c++ as programming language
3. use s/w memory model (shared memory model)
4. data to be processed is copied from main memory to gpu memory
5. each thread performs the same operation on different data (single instruction multiple thread)
6. one of the first to support heterogeneous architecture (system with two or more unique types of processing cores ex: cpu + gpu)
OpenCL (open computing language)
1. open standard for parallel programming of heterogeneous systems
2. use c as programming language
3. use s/w memory model (shared memory model)
MapReduce

1. programming model for processing large data sets with a parallel, distributed algorithm on a cluster
2. use key-value pair as data structure
3. use functional programming model (map and reduce functions)
4. use distributed memory model

mapper and reducer number can be configured by developer or determined by the system, and how the data is distributed to mapper is determined by the system
map: process key-value pair to generate intermediate key-value pair
reduce: merge all intermediate values associated with the same intermediate key
example applications: distributed grep, distributed sort, word count
1. word count
map: input (document name, document content) -> output (word, 1))
reduce: input (word, list(1,1,1,...)) -> output (word, count))
2. distributed grep
map: input (document name, document content) -> output (line number, line content)
reduce: input (line number, list(line content)) -> output (line number, list(line content)))


Hadoop (High-availability Distributed Object Oriented Platform)

hadoop was derived from google's mapreduce and google file system (GFS)
1. open source implementation of MapReduce
2. use java as programming language
3. use distributed file system (HDFS) to store data and communicate
4. use distributed memory models

## speedup and efficiency

in practice speedup < ideal speedup
1. overhead of creating and managing threads/processes
2. overhead of communication and synchronization
3. load imbalance

we define the speedup as
s = T serial / T parallel
for linear speedup, s = p

Amdahl's law
the speedup of a program using multiple processors in parallel computing is limited by the sequential fraction of the program

in reality, the parallel portion of a program may not be perfectly parallelizable, and there may be some overhead associated with managing the parallel tasks. Therefore, the actual speedup achieved may be less than the theoretical maximum predicted by Amdahl's law.

efficiency 
p = number of processors
S = speedup
E = S / p = T serial / (p * T parallel) 

ex: for 4 processors, if speedup is 4, efficiency is 4/4 = 1.0, for 8 processors, if speedup is 6, efficiency is 6/8 = 0.75

## scalability

1. a parallel system is scalable if the efficiency remains constant as the number of processors increases
2. a parallel system is scalable if the speedup increases as the number of processors increases
3. we want the efficiency to remain constant as the number of processors increases

## A word of warning

1. every parallel program contains at least one serial program