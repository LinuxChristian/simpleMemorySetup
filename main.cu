
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
#define real double // Define the precision

// Prototypes
void checkForCudaErrors(const char* checkpoint_description);
void initializeGPU();

int main(int argc, char* argv[])
{

  int xDim = 500; // Node count in x dimension
  int yDim = 500; // Node count in y dimension

  initializeGPU();

  // 
  // Case 1:
  // Linear test
  //

  real *Mat;   // Host pointer
  real *d_Mat; // Device pointer
  Mat = (real*) calloc(xDim, sizeof(real));  // Host memory
  cudaMalloc( (void**) &d_Mat, xDim );       // Device memory
  
  printf("Memory copy Host -> Device \n");
  cudaMemcpy( d_Mat,   Mat,  xDim, cudaMemcpyHostToDevice );
  checkForCudaErrors("Post test 1 memcpy.");

  printf("Clean up \n");
  free( Mat );
  cudaFree( d_Mat );

  
  printf("All done");
  return 0;
};


/**
 * This function loads and stores a element from
 * the matrix without shared memory
 * 
 * M_in  Pointer to input matrix
 * M_out Pointer to output matrix
 */
__global__ void cuLoadStoreElement(real *M_in, real *M_out) {

};

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

    printf("Using CUDA device ID: %i \n",(cudadevice));
    printf("  - Name: %s, compute capability: %i.%i.\n",prop.name,prop.major,prop.minor);
    printf("  - CUDA Driver version: %i.%i, runtime version %i.%i\n",cudaDriverVersion/1000,cudaDriverVersion%100,cudaRuntimeVersion/1000,cudaRuntimeVersion%100);
    printf("  - Max threads pr. block in x: %i, Max block size in x: %i \n\n",prop.maxThreadsDim[0], prop.maxGridSize[0]);

    // Comment following line when using a system only containing exclusive mode GPUs
    cudaChooseDevice(&cudadevice, &prop);
    checkForCudaErrors("Initializing GPU!");
};
