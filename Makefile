NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_20,code=sm_20
OUTPUT = main

main:
	$(NVCC) $(NVCCFLAGS) main.cu -o $(OUTPUT)

test: main test1

test1:
	./RunTest.sh