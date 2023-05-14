

#ifndef CUDA_LIB_H
#define CUDA_LIB_H

	#include "cpuLib.h"

	#include <cuda.h>
	#include <curand_kernel.h>

	// Uncomment this to suppress console output
	// #define DEBUG_PRINT_DISABLE

	//	Uncomment this to disable error counting
	//	#define CONV_CHECK_DISABLE 

	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
	extern void gpuAssert(cudaError_t code, const char *file, int line, bool abort);

	/**
	 * @brief GPU kernel to perform 2D Pool operation
	 * 
	 * @param input 	float *			pointer to input tensor
	 * @param inShape 	TensorShape		dimensions of input tensor
	 * @param output 	float *			pointer to output tensor
	 * @param outShape 	TensorShape		dimensions of output tensor
	 * @param args 		PoolLayerArgs	parameters of pool operation
	 * @return int 
	 */
	extern __global__ void poolLayer_gpu (float * input, TensorShape iShape,
		float * output, TensorShape oShape, PoolLayerArgs args, uint32_t batchSize);

	/**
	 * @brief 
	 * 
	 * @param input 	float *
	 * @param iShape 	TensorShape
	 * @param filter 	float *
	 * @param fShape 	TensorShape
	 * @param bias 		float *
	 * @param output 	float *
	 * @param oShape 	TensorShape		dimensions of output tensor
	 * @param args 		ConvLayerArgs	parameters for convolution operation
	 * @param batchSize uint32_t		
	 * @return int 
	 */
	extern __global__ void convLayer_gpu ( float * input, TensorShape iShape, 
		float * filter, TensorShape fShape, 
		float * bias, float * output, TensorShape oShape, 
		ConvLayerArgs args, uint32_t batchSize, uint32_t bias_offset);

	extern __global__ void gemmLayer_gpu(float *a, TensorShape aShape, 
                              float *b, TensorShape bShape,
                              float *c, TensorShape cShape,
                              GemmLayerArgs args);

	extern int runAlexnet (int argc, char ** argv);

#endif
