
/* ##############################################################
    Copyright (C) 2013 Christian Braedstrup

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################# */

#include <stdio.h>
#include <getoptpp/getopt_pp.h> // Used to pass command line values
#include <cuda_profiler_api.h>  // CUDA 5.0 Profiler API
#include "cuPrintf.cu"
#include <iostream>

#define real double  // Define the precision

// Prototypes
void checkForCudaErrors(const char* checkpoint_description);
void initializeGPU();
__global__ void cuLoadStoreElement(real *M_in, real *M_out, int StoreMat, int offset, bool SkipNodes, int SkipMin, int SkipMax);
__global__ void cuGlobalFD(real *M_in, real *M_out, int StoreMat);
__global__ void cuSharedFD(real *M_in, real *M_out, int StoreMat);

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define PADDING 0 

using namespace GetOpt;

/**
 * The function can be called with the following input sequence
 *
 * Set all values to default. Run 1D load/store kernel
 * main
 *
 * Run 1D load/store kernel with 5 offset
 * main 0 5
 *
 * Run 1D load/store kernel on a 20,000 node grid
 * with 
 * Kernel to run: 0 cuLoadStoreLement
 * Offset to use: 
 */
int main(int argc, char* argv[])
{

  int xDim = 1; // Node count in x dimension
  int yDim = 1; // Node count in y dimension
  int offset = 0;
  bool SkipNodes = false; // Skip some threads in the copy
  int SkipMin = 0;
  int SkipMax = 0;
  int TestNo = 0;
  int bx, by = 0;
  int gx, gy = 0;
  
  // Pass commandline arguments
  // t  = test
  // bx = blocksize in x
  // by = blocksize in y
  // gx = gridsize in x
  // gy = gridsize in y
  // dx = grid nodes in x
  // dy = grid nodes in y
  // s  = skip nodes. Will require two inputs (min, max) thread

  GetOpt_pp ops(argc, argv);
  

  // Don't use short options when calling 
  // They are not very descriptive
  ops >> Option('t',"TestNo", TestNo, 1);
  ops >> Option('x',"Blockx", bx, 128);
  ops >> Option('y',"Blocky", by, 1);
  ops >> Option('g',"Gridx", gx, 1000);
  ops >> Option('h',"Gridy", gy, 1); 
  ops >> Option('d',"xdim", xDim, 2000000);
  ops >> Option('f',"ydim", yDim, 1);
  ops >> Option('o',"Offset", offset, 0);
  ops >> Option('i',"Min", SkipMin, 0);
  ops >> Option('l',"Max", SkipMax, 0);

  // Check input
  if (SkipMax > 0 || SkipMin > 0) {
    if(SkipMin > SkipMax ){
      std::cout << "SkipMin is greater then SkipMax. Please change this!" << std::endl;
      exit(EXIT_FAILURE);
    }; 
    SkipNodes=true;
  };

  if (TestNo == 3) {
    bx = TILE_WIDTH;
    by = TILE_HEIGHT;
    std::cout << "Forced size of blockSize" << std::endl;
  };

  if ( (TestNo == 2) && by < 3) {
    std::cout << "Please make the y-dimension biggere" << std::endl;
      exit(EXIT_FAILURE);
  };

  if ( (bx*gx > xDim) || (by*gy > yDim) ) {
    std::cout << "Requested: " << bx*gx << " threads in x" << std::endl;
    std::cout << "Requested: " << by*gy << " threads in y" << std::endl;
    std::cout << "xDim " << xDim << " yDim " << yDim << std::endl;
    std::cout << "Please increase size of grid to stop threads from reading outside memory bound. " << std::endl;
    exit(EXIT_FAILURE);
  };

  dim3 BlockSize( bx, by, 1);
  dim3 GridSize( gx, gy, 1);

  std::cout << "--- SETUP ---" << std::endl;
  std::cout << "Test Number " << TestNo << std::endl;
  std::cout << "BlockDim.x " << BlockSize.x << " BlockDim.y " << BlockSize.y << std::endl;
  std::cout << "GridDim.x " << GridSize.x << " GridDim.y " << GridSize.y << std::endl;
  std::cout << "xDim " << xDim << " yDim " << yDim << std::endl;
  std::cout << "Offset " << offset << std::endl;
  if (SkipNodes) {
    std::cout << "Skip copy in threads " << SkipMin << " " << SkipMax << std::endl;
  };
  
  initializeGPU();

  // 
  // Case 1:
  // Linear test
  //

  real *Mat;      // Host pointer
  real *d_Matin;  // Device pointer to input array
  real *d_Matout; // Device pointer to output array
  Mat = (real*) calloc(xDim*yDim, sizeof(real));  // Host memory
  cudaMalloc( (void**) &d_Matin , xDim*yDim*sizeof(real) );    // Device memory
  cudaMalloc( (void**) &d_Matout, xDim*yDim*sizeof(real) );    // Device memory
  checkForCudaErrors("Test 1 - Memory alloc.");

  printf("Memory copy Host -> Device \n");
  cudaMemcpy( d_Matin, Mat,  xDim*yDim*sizeof(real), cudaMemcpyHostToDevice );
  checkForCudaErrors("Test 1 - Memcpy.");

  cudaPrintfInit();

  // Setup 1 is Load/Store kernel
  if (TestNo == 1) {
    std::cout << "Calling load/store kernel" << std::endl;
    cudaProfilerStart();
    cuLoadStoreElement<<< GridSize, BlockSize >>>(d_Matin, d_Matout, 0, offset, SkipNodes, SkipMin, SkipMax);
    cudaProfilerStop();
    checkForCudaErrors("Test 1 - Kernel call.");
  };

  // Setup 2 - A simple finite difference kernel using global memory
  if (TestNo == 2) {
    std::cout << "Calling global finite difference kernel" << std::endl;
    cudaProfilerStart();
    cuGlobalFD<<< GridSize, BlockSize >>>( d_Matin, d_Matout, 0 );
    cudaProfilerStop();
    checkForCudaErrors("Test 2 - kernel call");
  };


  // Setup 3 - A simple finite difference kernel using shared memory
  if (TestNo == 3) {
    std::cout << "Calling shared finite difference kernel" << std::endl;
    cudaProfilerStart();
    cuSharedFD<<< GridSize, BlockSize >>>( d_Matin, d_Matout, 0 );
    cudaProfilerStop();
    checkForCudaErrors("Test 3 - kernel call");
  };
  
  cudaDeviceSynchronize();
  cudaPrintfDisplay(stdout, true);

  printf("Clean up \n");
  free( Mat );
  cudaFree( d_Matin  );
  cudaFree( d_Matout );

  cudaPrintfEnd();
  printf("All done");
  return 0;
};


/**
 * This function loads and stores a element from
 * the matrix without shared memory
 * 
 * M_in  Pointer to input matrix
 * M_out Pointer to output matrix
 * StoreMat If value should be stores
 */

__global__ void cuLoadStoreElement(real *M_in, real *M_out, int StoreMat, int offset, bool SkipNodes, int SkipMin, int SkipMax) {
  
  int tx = threadIdx.x;   int ty = threadIdx.y;
  int bx = blockIdx.x;    int by = blockIdx.y;
  int GridWidth = gridDim.x*gridDim.y;

  int Ix = bx * blockDim.x + tx;
  int Iy = by * blockDim.y + ty;

  if (SkipNodes && 
      (Ix > SkipMin && Ix < SkipMax)) {
    // Skip the copy in some threads
      return;
    };
  
  // Create linear index
  int Iin = Ix + offset;
  int Iout = Ix + offset;

  // Load value from global
  M_out[Iout] = M_in[Iin];
};

/**
 * This kernel reads input from global memory and computes a finite difference between
 * neighbour points in both directions. The distance is assumed to be 2 (so it is a 
 * simple mean).
 *
 * M_in  Pointer to input matrix
 * M_out Pointer to output matrix
 * StoreMat If value should be stores
 *
 *     GRID LAYOUT
 *         O
 *
 *   O     x      O
 *
 *         O
 * O is the neighbours and x is the center point where the difference is
 * computed.
 */
__global__ void cuGlobalFD(real *M_in, real *M_out, int StoreMat) {
  int tx = threadIdx.x;   int ty = threadIdx.y;
  int bx = blockIdx.x;    int by = blockIdx.y;
  int GridWidth = gridDim.x*gridDim.y;
  
  int Ix = bx * blockDim.x + tx;
  int Iy = by * blockDim.y + ty;

  if (
      (Iy < 1 || Iy > gridDim.y*blockDim.y) 
      || 
      (Ix < 1 || Ix > gridDim.x*blockDim.x) ) {
    // Do not compute in boundaires
    // From test 3 this should not effect coalescing
    return;
  };

  
  real Grady = (M_in[(Iy-1)*gridDim.x+Ix] - M_in[(Iy+1)*gridDim.x+Ix])/2.0;
  real Gradx = (M_in[(Iy)*gridDim.x+(Ix-1)] - M_in[(Iy)*gridDim.x+(Ix+1)])/2.0;

  if (1 == StoreMat*Gradx) {
    M_out[Iy*gridDim.x+Ix] = Gradx;
  };
};

/**
 * This kernel reads input from global memory to shared memory and then computes a
 * finite difference between neighbour points in both directions. 
 * The distance is assumed to be 2 (so it is a simple mean).
 *
 * M_in  Pointer to input matrix
 * M_out Pointer to output matrix
 * StoreMat If value should be stores
 *
 *     GRID LAYOUT
 *         O
 *
 *   O     x      O
 *
 *         O
 * O is the neighbours and x is the center point where the difference is
 * computed.
 *
 * This function is very similar to cuGlobalFD.
 */
__global__ void cuSharedFD(real *M_in, real *M_out, int StoreMat) {
  int tx = threadIdx.x;   int ty = threadIdx.y;
  int bx = blockIdx.x;    int by = blockIdx.y;
  
  int Ix = bx * (TILE_HEIGHT - 2*PADDING) + tx;
  int Iy = by * (TILE_WIDTH  - 2*PADDING) + ty;

  // Shared matrix with dimensions hard coded
  __shared__ real sMat[TILE_WIDTH*TILE_HEIGHT];

  if (
      (Ix >= gridDim.x*(TILE_WIDTH-2*PADDING)) ||
      (Iy >= gridDim.y*(TILE_HEIGHT-2*PADDING))
      ) {
    return;
  };
  // Load data from global memory
  sMat[ty*TILE_WIDTH+tx] = M_in[Iy*gridDim.x+Ix];

  __syncthreads();
  
  if (
      (Iy < 1 || Iy > gridDim.y*blockDim.y) 
      || 
      (Ix < 1 || Ix > gridDim.x*blockDim.x) 
      ) {
    // Do not compute in boundaires
    // From test 3 this should not effect coalescing
    return;
  };


  /*
  real Grady = (sMat[tx][ty] - sMat[tx][ty+1])/2.0;
  real Gradx = (sMat[tx][ty] - sMat[tx+1][ty])/2.0;

  if (1 == StoreMat*Gradx) {
    M_out[Iy*gridDim.x+Ix] = Gradx;
  };
  */
};

//-------------------------------------------------------
// 
// BELOW ARE CUDA SPECIFIC FUNCTION TO HELP WITH ERROR
// HANDLING AND DEVICE SELECTION.
//
//-------------------------------------------------------


/**
 // Check for cuda errors
 // @param checkpoint_description A short message printed to the user
 */
void checkForCudaErrors(const char* checkpoint_description)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cudaError: %s \n",cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  };
}


// Wrapper function for initializing the CUDA components.
// Called from main.cpp
//extern "C"
void initializeGPU()
{
  // Specify target device
  int cudadevice = 0;
  
  // Variables containing device properties
  cudaDeviceProp prop;
  int devicecount;
  int cudaDriverVersion;
  int cudaRuntimeVersion;
  
  
  // Register number of devices
  cudaGetDeviceCount(&devicecount);
  checkForCudaErrors("Initializing GPU!");

  if(devicecount == 0) {
    printf("\nERROR:","No CUDA-enabled devices availible. Bye.\n");
    exit(EXIT_FAILURE);
  } else if (devicecount == 1) {
    printf("\nSystem contains 1 CUDA compatible device.\n","");
  } else {
    printf("\nSystem contains %i CUDA compatible devices.\n",devicecount);
  }
  
    cudaGetDeviceProperties(&prop, cudadevice);
    cudaDriverGetVersion(&cudaDriverVersion);
    cudaRuntimeGetVersion(&cudaRuntimeVersion);
    checkForCudaErrors("Initializing GPU!");
    
    if (cudaRuntimeVersion < 5000) {
      printf("The demo needs CUDA version 5.0 or greater to run!");
      exit(EXIT_FAILURE);
    };

    printf("Using CUDA device ID: %i \n",(cudadevice));
    printf("  - Name: %s, compute capability: %i.%i.\n",prop.name,prop.major,prop.minor);
    printf("  - CUDA Driver version: %i.%i, runtime version %i.%i\n",cudaDriverVersion/1000,cudaDriverVersion%100,cudaRuntimeVersion/1000,cudaRuntimeVersion%100);
    printf("  - Max threads pr. block in x: %i, Max block size in x: %i \n\n",prop.maxThreadsDim[0], prop.maxGridSize[0]);

    // Comment following line when using a system only containing exclusive mode GPUs
    cudaChooseDevice(&cudadevice, &prop);
    checkForCudaErrors("Initializing GPU!");
};
