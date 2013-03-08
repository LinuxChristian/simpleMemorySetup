=================
simpleMemorySetup
=================

A simple test setup to run different memory alignments on CUDA devices. Currently it is a implementation of the CUDA offsetCopy kernel from the programming guide. For more infomation on CUDA memory read [3].

** REQUIRES CUDA 5.0 OR GREATER **

Running
-----------

To run the program do:
    
|    make
|    make test

This will run the RunTest.sh script and use nvprof to get global load/store transactions with no offset.

Implemented tests
___________________

Test 1:  - Constant offset
"""""""""""""""""""""""""""

This tests run the copy with a constant offset that is user defined. The test can be run with the command

|    make test1

or 

|    ./RunTest.sh 1 5

where the first value (1) is the test and the second value (5) is the offset to use. This will profile the run and return the following events from nvprof,

* L1_global_load_hit
* L1_global_load_miss
* gld_requested
* gst_requested
* global_store_requests

Run **nvprof --query-events** to get a list of available events on the system. Infomation on the CUDA profiler can be found here [1].
When running the default large setup (i.e. 1000 blocks pr. grid and 128 threads pr. block) and a zero offset the resulting ratio should be 2 when using doubles (which is also standart). This means that the access is coalesced. Running with a non-zero offset will result in a ratio greater then 2.

Test 2 - Changing offset
"""""""""""""""""""""""""

Test 2 is almost the same as test 1 but the profiler now returns kernel runtime and not the above profiler events. The offset is run for the offset sequence 0:2:32. The output is printed to stdout and saved into a textfile. This file will then be plottet by gnuplot. The result should have the same shape as the one from [3] (Figure 6).


Test 3 - Skipping threads in the copy
""""""""""""""""""""""""""""""""""""""

Same setup as test 1 but not all threads will copy memory. By default threads 36 -> 45 will skip the copy. The result is the same as in test 1 if similar parmeters where used. This agrees with the theory.

Test 4 - Finite difference using global memory
""""""""""""""""""""""""""""""""""""""""""""""

The setup is a quite simple finite difference stencil that computes the difference between the value above and below the current node. All the nodes are independent therefore the kernel should be able to scale nicely. 
The problem with the kernel is that the memory access is now no longer coalecsed. Nodes will access memory at quite different addresses. There will also be a overlap where four threads need to access the same infomation. The test should therefore show a cache hit rate greatere then 0% and a ratio greater then 2 because the GPU now has to transfer much more data.

Test 5 - Finite difference using shared memory
""""""""""""""""""""""""""""""""""""""""""""""

The same stencil as in test 4 but now the vaules are extracted from shared memory. This means that the uncoalesced memory access is now to shared memory and the overhead should be reduced a lot.

Requirements
____________________

To use the software you need to install getoptpp [2] and have the file getopt_pp.cpp in the root src directory. You will also need gnuplot if you want to produce .png files.
If the system does not have a awk version installed I suggest installing gawk.

References
_____________________

* [1] < http://docs.nvidia.com/cuda/profiler-users-guide/index.html >
* [2] < http://code.google.com/p/getoptpp >
* [3] < http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-global-memory >
