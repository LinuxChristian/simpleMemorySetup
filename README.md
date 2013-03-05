simpleMemorySetup
=================

A simple test setup to run different memory alignments on CUDA devices. Currently it is a implementation of the CUDA offsetCopy kernel from the programming guide.

Running
-----------

First do a
    
  make

and then
    
  make test

This will run the RunTest.sh script and use nvprof to get global load/store transactions with no offset.