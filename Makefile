NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_20,code=sm_20 -lineinfo
OUTPUT = main
INCLUDE = -I/home/christian/cuda_5.0_samples/0_Simple/simplePrintf/	

main:
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) main.cu -o $(OUTPUT)

test: main test1

test1:
	./RunTest.sh 0

clean:
	rm $(OUTPUT)