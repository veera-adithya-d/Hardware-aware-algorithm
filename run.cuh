#pragma once

#ifndef RUN_CUH
#define RUN_CUH

	#include "cpuLib.h"
	// #define DEBUG_PRINT_DISABLE
	
	#define VECTOR_SIZE (1 << 15)

	#define MC_SAMPLE_SIZE		1e6
	#define MC_ITER_COUNT		32

	#define WARP_SIZE			32
	#define SAMPLE_SIZE			MC_SAMPLE_SIZE
	#define GENERATE_BLOCKS		1024
	#define REDUCE_SIZE			32
	#define REDUCE_BLOCKS		(GENERATE_BLOCKS / REDUCE_SIZE)

	extern int testLoadBytesImage(std::string filePath);


#endif