# parallel hardware and software

before 2002: single core
before 2002: multi core
why?

1. Demand for higher performance
2. Power efficiency
3. Improved throughput
4. cmos physical limits frequency scaling

how do uniprocessor computer architectures extract parallelism?

# Instruction-level parallelism (ILP)

## Pipelining

Pipelining helps to increase instruction throughput but not reduce latency
throughput limits by slowest pipeline stage
potential speedup = number of pipeline stages

in MIPS, each instruction takes 5 cycles
but with pipelining, one instruction completes every cycle
            v  v
t1 t2 t3 t4 t5
   t1 t2 t3 t4 t5
      t1 t2 t3 t4 t5
         t1 t2 t3 t4 t5
            t1 t2 t3 t4 t5
               t1 t2 t3 t4 t5

limits to pipelining

can't switch instructions around

1. structural hazards
attempts to use the same hardware resource at the same time

2. data hazards

read after write (RAW) {true dependency, can't not be eliminated
write after read (WAR) {anti-dependency, we can just write to a different register
write after write (WAW) {output dependency, same with WAR, we can just write to a different register
solution:
bobbling: stall the pipeline when register is not ready to read (in other word, it is a sync action)

3. control hazards
branch instructions

what is a branch? a branch is a type of instruction that can cause the program to deviate from its sequential flow. it typically involves a comparison and can result in different instructions being executed based on the outcome of that comparison.

## Out-of-order execution

DIVD R1, R2, R3
ADD R4, R1, R5 <- needs to wait for R1
MUL R6, R7, R8 <- can be executed first because it does not depend on R1

deal with hazards: by guessing (branch prediction)
if guessing is wrong, flush the pipeline and undo the wrong work

## Superscalar execution

try to find independent instructions that can put in to different functional units to execute instruction simultaneously

sequential source code
-> superscalar compiler
-> sequential machine code
-> superscalar processor (scheduler)
-> multiple instructions executed in parallel

VLIW (Very Long Instruction Word)
-> superscalar compiler
-> sequential machine code
-> VLIW processor (no scheduler)
-> multiple instructions executed in parallel

ISA view: HW binder
VLIW: SW binder


## SIMD (Single Instruction, Multiple Data)
vector processing: too flexible, too expensive
SIMD: less flexible, less expensive

you either write the asm code by yourself to call the SIMD instructions
or compiler auto vectorization your code

multimedia extensions (SIMD extensions)
MMX (MultiMedia eXtensions) Intel 1997
motivations: speed up multimedia applications
sub word parallelism: multiple data in one register
treat a 32-bit register as a vector of 4 8-bit integers, 2 16-bit integers, or 1 32-bit integer

how to explicitly tell compiler to use SIMD instructions if compiler doesn't parallelize the code?
1. inline assembly (not portable)
2. intrinsic functions (higher portability, compiler-specific)

# Thread-level parallelism (TLP)

common notions of thread creation

if hardware does the context switch, it requires
1. separate PC
2. separate register set
3. separate stack (separate page table)

# simultaneous multithreading (SMT) (hyper-threading from Intel)
combine of hardware thread and multithreading
multiple threads share the same physical core
ex: two thread doesn't use the same functional unit at the same time, so they can be executed in parallel in the same core
so it will be seems like two cores

so 4 core 8 T means 4 physical core, each core can run 2 threads simultaneously, and in reality, 2 threads in a core is ideal since most of the time the functional unit that a thread require is common, so with more than 2 threads in a core, it will be just overhead

Flynn's taxonomy
SISD: single instruction single data (uniprocessor)
SIMD: single instruction multiple data (vector processing, multimedia extensions)
MISD: multiple instruction single data (not exist in reality)
MIMD: multiple instruction multiple data (multi-core, multi-processor, multi-computers)

# types of shared memory architectures
1. UMA (Uniform Memory Access)
all processors share the same physical memory
memory access time is uniform
2. NUMA (Non-Uniform Memory Access)
each processor has its own local memory and can also access other processor's memory
memory access time is non-uniform

before, memory controller is in the north bridge chipset, so all cpu memory access time is uniform (UMA)
now, memory controller is in the cpu, so each cpu has its own local memory, and can also access other cpu's memory (NUMA)

# UMA 

## cache coherence problem
write back vs write through
write back: write to cache, and write to memory when evicted
write through: write to cache and memory at the same time

how to fix with with a bus: coherence protocol
write through is not enough for cache coherence since other processor's cache may already have cache line
invalidate other processors' caches when a cache line is written
example: 8 core AMD Opteron

# NUMA

coherence is not enough
## memory consistency model
specification of the order in which memory operations (read and write) appear to execute to the programmer
ex: sequential consistency: only one processor can access the memory at a time, and the result is the same as if the operations of all the processors were executed in some sequential order, and the operations of each individual processor appear in this sequence in the order issued by that processor

example:
CPU0         CPU1
A=0          A=0
A=1          A=1
print A       print A

## summerize

coherence defines what values can be returned by a read
consistency defines when a written value by a read from another processor