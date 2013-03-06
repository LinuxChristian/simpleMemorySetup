
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

  if (SkipMax > 0 || SkipMin > 0) {
    if(SkipMin > SkipMax ){
      std::cout << "SkipMin is greater then SkipMax. Please change this!" << std::endl;
      exit(EXIT_FAILURE);
    }; 
    SkipNodes=true;
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
  real *d_Matout; // Device pointer to input array
  Mat = (real*) calloc(xDim, sizeof(real));  // Host memory
  cudaMalloc( (void**) &d_Matin , xDim*yDim*sizeof(real) );    // Device memory
  cudaMalloc( (void**) &d_Matout, xDim*yDim*sizeof(real) );    // Device memory
  checkForCudaErrors("Test 1 - Memory alloc.");

  printf("Memory copy Host -> Device \n");
  cudaMemcpy( d_Matin, Mat,  xDim, cudaMemcpyHostToDevice );
  checkForCudaErrors("Test 1 - Memcpy.");

  cudaPrintfInit();

  // Test 1 is Load/Store kernel
  if (TestNo == 1) {
    cudaProfilerStart();
    cuLoadStoreElement<<<GridSize, BlockSize>>>(d_Matin, d_Matout, 0, offset, SkipNodes, SkipMin, SkipMax);
    cudaProfilerStop();
    checkForCudaErrors("Test 1 - Kernel call.");
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
 * StoreMat Bool if value should be stores
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

  //  cuPrintf("Index %i bx %i bd %i \n",Iin, bx, gridDim.x);
  // Load value from global
  M_out[Iout] = M_in[Iin];

  // Avoid compiler optimization if
  // no store request is given
  /*
  if ( 1 == ValIn*StoreMat ) {
    M_out[Ix] = (double) 5.0;// ValIn;
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
