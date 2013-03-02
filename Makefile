NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_20,code=sm_20

all:
	$(NVCC) $(NVCCFLAGS) main.cu -o main