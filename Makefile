NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_20,code=sm_20 -lineinfo
OUTPUT = main
INCLUDE = -I/home/christian/cuda_5.0_samples/0_Simple/simplePrintf/	
GETOPTPP = getopt_pp.cpp

main:
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) $(GETOPTPP) main.cu -o $(OUTPUT)

test: main test1 test2 test3

test1: main
	./RunTest.sh 1 0

test2:
	./RunTest.sh 2
	gnuplot GPUMemoryThroughput.gp > GPUMemoryThroughput.png	

test3:
	./RunTest.sh 3

test4:
	./RunTest.sh 4

test5:
	./RunTest.sh 5

test6:
	./RunTest.sh 6

test7:
	./RunTest.sh 7

test8: test7
	./RunTest.sh 8

plot7: test7 
	gnuplot PaddingEfficiency.gp > PaddingEfficiency.png

plot8: test8
	gnuplot GlobalVsSharedStencil.gp > GlobalVsSharedEfficiency.png

clean:
	rm $(OUTPUT)