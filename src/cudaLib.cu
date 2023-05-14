#include "cudaLib.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__constant__ float bias_d[4096];

__global__ void convLayer_gpu ( float * input, TensorShape iShape, 
	float * filter, TensorShape fShape, float * output, TensorShape oShape, 
	ConvLayerArgs args, uint32_t batchSize=1, uint32_t bias_offset=0) {

	// Define shared memory
	extern __shared__ float ds_i[];
	uint32_t ds_W = blockDim.x;

	// Define convolutional parameters
	uint32_t tx = threadIdx.x;
	uint32_t ty = threadIdx.y;
	uint32_t col_ref = (blockDim.x-fShape.width+args.strideW)/args.strideW;
	uint32_t row_ref = (blockDim.y-fShape.height+args.strideH)/args.strideH;

	// Define indexing paramters
	uint32_t inCol = blockIdx.x*col_ref*args.strideW+tx;
	uint32_t inRow = blockIdx.y*row_ref*args.strideH+ty;
	uint32_t outCol = blockIdx.x*col_ref+tx;
	uint32_t outRow = blockIdx.y*row_ref+ty;
	uint32_t offset_input, offset_shared, offset_output, offset_filter;
	float value_input, value_filter, value_output;

	// Begin convolution
	for (uint32_t n=0; n<batchSize; ++n){
		for(uint32_t m=0; m<oShape.channels; ++m){
			value_output = 0;
			value_filter = 0;
			if(inCol-tx+fShape.width <= iShape.width+2*args.padW && inRow-ty+fShape.height <= iShape.height+2*args.padH){
				for(uint32_t k=0; k<iShape.channels; ++k){
					value_output = 0;
				
					// Load data into shared memory
					offset_shared = ty*ds_W + tx;
					if(inCol>=args.padW && inRow>=args.padH && inCol-args.padW<iShape.width && inRow-args.padH<iShape.height){
						offset_input = inCol - args.padW + iShape.width*(inRow - args.padH + iShape.height*(k + iShape.channels*n));
						ds_i[offset_shared] = input[offset_input];
					}
                    else{
                        ds_i[offset_shared] = 0.0;
                    }
					__syncthreads();

					// Apply filters
					if(tx < col_ref && ty < row_ref && outCol < oShape.width && outRow < oShape.height){
						
						offset_output = outCol + oShape.width*(outRow + oShape.height*(m + oShape.channels*n));
						if(k==0) output[offset_output] = bias_d[m + bias_offset];
						
						for(uint32_t i=0; i<fShape.height; ++i){
							for(uint32_t j=0; j<fShape.width; ++j){
								offset_shared = (args.strideW*tx+j)+ds_W*(args.strideH*ty+i);
								value_input = ds_i[offset_shared];
								offset_filter = j+fShape.width*(i+fShape.height*(k+fShape.channels*m));
								value_filter = filter[offset_filter];

								value_output += value_input*value_filter;
							}
						}
						output[offset_output] += value_output;


					}
					__syncthreads();
				}

				if(tx < col_ref && ty < row_ref && outCol < oShape.width && outRow < oShape.height){
					offset_output = outCol + oShape.width*(outRow + oShape.height*(m + oShape.channels*n));
					// Activation - ReLU
					if(output[offset_output]<0 || output[offset_output]>1000000000) output[offset_output]=0;
				}
				__syncthreads();
			}
		}
	}
}

__global__ void poolLayer_gpu(float *input, TensorShape iShape, float *output, TensorShape oShape, PoolLayerArgs args, uint32_t batchSize){
    
	float poolPick;
    // Static shared memory
    __shared__ float s[16][16];

    uint32_t poolH = args.poolH;
    uint32_t poolW = args.poolW;
    uint32_t outputH = oShape.height;
    uint32_t outputW = oShape.width;
    uint32_t row, col;

    uint32_t outCol = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t outRow = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t c = blockIdx.z;
	
    if (outCol < outputW && outRow < outputH && c < iShape.channels) {
        switch (args.opType){
            case PoolOp::MaxPool: poolPick = -__FLT_MAX__; break;
            case PoolOp::AvgPool: poolPick = 0; break;
            case PoolOp::MinPool: poolPick = __FLT_MAX__; break;
            default: poolPick = -__FLT_MAX__; break;
        }
		for (uint32_t n=0; n<batchSize; ++n){
			for (uint32_t poolRow = 0; poolRow < poolH; ++poolRow) {
				for (uint32_t poolCol = 0; poolCol < poolW; ++poolCol) {
					row = outRow * args.strideH + poolRow;
					col = outCol * args.strideW + poolCol;

					// Load data from global memory into shared memory
					if (row < iShape.height && col < iShape.width){
						s[threadIdx.y][threadIdx.x] = *(input + ((n * iShape.channels + c) * iShape.height + row) * iShape.width + col);
					}
					else s[threadIdx.y][threadIdx.x] = 0;
					__syncthreads();

					switch (args.opType) {
						case PoolOp::MaxPool:
							if (poolPick < s[threadIdx.y][threadIdx.x]) {
								poolPick = s[threadIdx.y][threadIdx.x];
							}
							break;
						case PoolOp::AvgPool:
							poolPick += s[threadIdx.y][threadIdx.x];
							break;
						case PoolOp::MinPool:
							if (poolPick > s[threadIdx.y][threadIdx.x]) {
								poolPick = s[threadIdx.y][threadIdx.x];
							}
							break;
						default:
							if (poolPick < s[threadIdx.y][threadIdx.x]) {
								poolPick = s[threadIdx.y][threadIdx.x];
							}
							break;
					}
					__syncthreads();
				}
			}
			if (args.opType == PoolOp::AvgPool) poolPick = poolPick / (poolH * poolW);
			*(output + ((n * oShape.channels + c) * outputH + outRow) * outputW + outCol) = poolPick;
		}
    }
}

int makeTensorUVM (float ** t, TensorShape & shape) {

	if (shape.count == 0) {
		std::cout << " Shape has invalid count (4th dim) - setting to 1 \n";
		shape.count = 1;
	}

	uint64_t offset;
	float *m = *t;

	std::random_device random_device;
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	//	Implement NCHW layout
	for (uint32_t count = 0; count < shape.count; ++ count) {
		for (uint32_t chIdx = 0; chIdx < shape.channels; ++ chIdx ) {
			for (uint32_t rowIdx = 0; rowIdx < shape.height; ++ rowIdx) {
				for (uint32_t colIdx = 0; colIdx < shape.width; ++ colIdx) {
					offset = count*shape.channels*shape.height*shape.width + chIdx * shape.height * shape.width + rowIdx * shape.width + colIdx;
					m[offset] = dist(random_device);
				}
			}
		}
	}
	return 0;
}

__global__ void gemmLayer_gpu(float *a, TensorShape aShape, 
                              float *b, TensorShape bShape,
                              float *c, TensorShape cShape,
                              GemmLayerArgs args){

    uint64_t row = blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t col = blockIdx.y*blockDim.y+threadIdx.y;
    uint64_t subTilesAlongK = (aShape.width + args.tileH - 1) / args.tileH;
    uint64_t subTile,subTileK;

    // Declare shared memory
	extern __shared__ float sharedMem[];
    float *aTile = &sharedMem[0];
    float *bTile = &sharedMem[args.tileW * args.tileH];

    for (subTile=0; subTile<subTilesAlongK; ++subTile){
		if (row<cShape.height && col<cShape.width && subTile == 0) *(c+IDX2R(row, col, cShape.width)) = 0;

        // Load data from global memory to shared memory
        if (row<aShape.height && subTile*args.tileW+threadIdx.y<aShape.width){
			aTile[threadIdx.x*blockDim.y+threadIdx.y] = a[IDX2R(row, subTile*args.tileW+threadIdx.y, aShape.width)];
		}
		else aTile[threadIdx.x*blockDim.y+threadIdx.y] = 0.0;

        if (subTile*args.tileH+threadIdx.x<bShape.height && col<bShape.width){
			bTile[threadIdx.x*blockDim.y+threadIdx.y] = b[IDX2R(subTile*args.tileH+threadIdx.x, col, bShape.width)];
		}
		else bTile[threadIdx.x*blockDim.y+threadIdx.y] = 0.0;
        __syncthreads();

        // Perform matrix multiplication using data from shared memory
        for (subTileK = 0; subTileK < args.tileW; ++subTileK) {
            if (row<cShape.height && col<cShape.width){
				*(c+IDX2R(row, col, cShape.width)) += aTile[threadIdx.x*blockDim.y+subTileK] * bTile[subTileK*blockDim.y+threadIdx.y];
			}
        }
        __syncthreads();
    }
}

int runAlexnet (int argc, char ** argv){

	uint32_t batchSize = 8;
	uint32_t bias_offset = 0;

	// ------------------------------- ALEX NET CONV 1 LAYER ----------------------------------------
	TensorShape iShape = AlexL1_InShape;
	iShape.count = batchSize;
	TensorShape fShape = AlexL1_FilterShape;
	ConvLayerArgs args = AlexL1_ConvArgs;

	TensorShape oShapeL1;
	oShapeL1.height 	= (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShapeL1.width	= (iShape.width  + 2 * args.padW - fShape.width)  / args.strideW + 1;
	oShapeL1.channels	= fShape.count;
	oShapeL1.count 	= batchSize;		
	
	float * in = nullptr;
	float * filter = nullptr;
	float * bias = nullptr;

	int retVal;
	retVal = makeTensor(&in, iShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return __UINT32_MAX__;
	}
	retVal = makeTensor(&filter, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return __UINT32_MAX__;
	}
	retVal = makeVector(&bias, fShape.count);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n" ;
		return __UINT32_MAX__;
	}

	float *in_d, *filter_d, *out_d;

	gpuErrchk(cudaMalloc(&in_d, tensorSize(iShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(in_d, in, tensorSize(iShape)*sizeof(float), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&filter_d, tensorSize(fShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(filter_d, filter, tensorSize(fShape)*sizeof(float), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpyToSymbol(bias_d, bias, fShape.count*sizeof(float), bias_offset*sizeof(float)));

	gpuErrchk(cudaMalloc(&out_d, tensorSize(oShapeL1)*sizeof(float)));

	float TILE_WIDTH = 16.0;
	uint32_t row_ref = (uint32_t) ((TILE_WIDTH-fShape.height+args.strideH)/args.strideH);
	uint32_t col_ref =  (uint32_t) ((TILE_WIDTH-fShape.width+args.strideW)/args.strideW);
	dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 grid(ceil((float)(iShape.width+2*args.padW)/(float)(col_ref*args.strideW)), ceil((float)(iShape.height+2*args.padH)/(float)(row_ref*args.strideH)), 1);

	// Launch the kernel
	convLayer_gpu <<< grid, block, TILE_WIDTH*TILE_WIDTH*sizeof(float)>>> (in_d, iShape, filter_d, fShape, out_d, oShapeL1, args, batchSize, bias_offset);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	bias_offset += fShape.count;
	// Free variables
	cudaFree(in_d);
	cudaFree(filter_d);
	free(in);
	free(filter);
	free(bias);

	// ----------------------------- MAX POOLING 1 LAYER -----------------------------------------
	iShape = oShapeL1;
	PoolLayerArgs poolArgs = AlexPL1_poolArgs;
	TensorShape oShapePL1;

	oShapePL1 = {batchSize,iShape.channels,((iShape.height-poolArgs.poolH)/poolArgs.strideH)+1,((iShape.width-poolArgs.poolW)/poolArgs.strideW)+1};
	float * inL2 = (float *) malloc(tensorSize(oShapePL1)*sizeof(float));
	float *out_d_pool;
	gpuErrchk(cudaMalloc(&out_d_pool, tensorSize(oShapePL1)*sizeof(float)));
	
	grid.x = ceil((float)oShapePL1.width/TILE_WIDTH);
	grid.y = ceil((float)oShapePL1.height/TILE_WIDTH);
	grid.z = iShape.channels;

	// Launch the kernel
	poolLayer_gpu <<<grid, block>>> (out_d, iShape, out_d_pool, oShapePL1, poolArgs, batchSize);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(inL2, out_d_pool, tensorSize(oShapePL1)*sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(out_d);
	cudaFree(out_d_pool);
	
	
	// ------------------------------- ALEX NET CONV 2 LAYER ----------------------------------------
	iShape = oShapePL1;
	fShape = AlexL2_FilterShape;
	args = AlexL2_ConvArgs;

	TensorShape oShapeL2;
	oShapeL2.height 	= (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShapeL2.width	= (iShape.width  + 2 * args.padW - fShape.width)  / args.strideW + 1;
	oShapeL2.channels	= fShape.count;
	oShapeL2.count 	= batchSize;		
	
	float * in_checkL2 = nullptr;
	float * filterL2 = nullptr;
	float * biasL2 = nullptr;

	retVal = makeTensor(&in_checkL2,iShape);
	retVal = makeTensor(&filterL2, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return __UINT32_MAX__;
	}
	retVal = makeVector(&biasL2, fShape.count);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n" ;
		return __UINT32_MAX__;
	}

	gpuErrchk(cudaMalloc(&in_d, tensorSize(iShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(in_d, in_checkL2, tensorSize(iShape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&filter_d, tensorSize(fShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(filter_d, filterL2, tensorSize(fShape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(bias_d, biasL2, fShape.count*sizeof(float), bias_offset*sizeof(float)));
	gpuErrchk(cudaMalloc(&out_d, tensorSize(oShapeL2)*sizeof(float)));

	row_ref = (uint32_t) ((TILE_WIDTH-fShape.height+args.strideH)/args.strideH);
	col_ref =  (uint32_t) ((TILE_WIDTH-fShape.width+args.strideW)/args.strideW);
	grid.x = ceil((float)(iShape.width+2*args.padW)/(float)(col_ref*args.strideW));
	grid.y = ceil((float)(iShape.height+2*args.padH)/(float)(row_ref*args.strideH));
	grid.z = 1;

	// Launch the kernel
	convLayer_gpu <<< grid, block, TILE_WIDTH*TILE_WIDTH*sizeof(float) >>> (in_d, iShape, filter_d, fShape, out_d, oShapeL2, args, batchSize, bias_offset);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	
	bias_offset += fShape.count;
	// Free variables
	cudaFree(in_d);
	cudaFree(filter_d);
	free(in_checkL2);
	free(inL2);
	free(filterL2);
	free(biasL2);

	
	// ----------------------------- MAX POOLING 2 LAYER -----------------------------------------
	iShape = oShapeL2;
	poolArgs = AlexPL2_poolArgs;
	TensorShape oShapePL2;
	oShapePL2 = {batchSize,iShape.channels,((iShape.height-poolArgs.poolH)/poolArgs.strideH)+1,((iShape.width-poolArgs.poolW)/poolArgs.strideW)+1};
	
	gpuErrchk(cudaMalloc(&out_d_pool, tensorSize(oShapePL2)*sizeof(float)));
	
	grid.x = ceil((float)oShapePL2.width/TILE_WIDTH);
	grid.y = ceil((float)oShapePL2.height/TILE_WIDTH);
	grid.z = iShape.channels;

	float * inL3 = (float *) malloc(tensorSize(oShapePL2)*sizeof(float));
	// Launch the kernel
	poolLayer_gpu <<<grid, block>>> (out_d, iShape, out_d_pool, oShapePL2, poolArgs, batchSize);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(inL3, out_d_pool, tensorSize(oShapePL2)*sizeof(float), cudaMemcpyDeviceToHost));
	
	cudaFree(out_d);
	cudaFree(out_d_pool);

	// ------------------------------- ALEX NET CONV 3 LAYER ----------------------------------------
	iShape = oShapePL2;
	fShape = AlexL3_FilterShape;
	args = AlexL3_ConvArgs;

	TensorShape oShapeL3;
	oShapeL3.height 	= (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShapeL3.width	= (iShape.width  + 2 * args.padW - fShape.width)  / args.strideW + 1;
	oShapeL3.channels	= fShape.count;
	oShapeL3.count 	= batchSize;
		
	float * in_checkL3 = nullptr;
	float * filterL3 = nullptr;
	float * biasL3 = nullptr;

	retVal = makeTensor(&in_checkL3,iShape);
	retVal = makeTensor(&filterL3, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return __UINT32_MAX__;
	}
	retVal = makeVector(&biasL3, fShape.count);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n" ;
		return __UINT32_MAX__;
	}

	gpuErrchk(cudaMalloc(&in_d, tensorSize(iShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(in_d, in_checkL3, tensorSize(iShape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&filter_d, tensorSize(fShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(filter_d, filterL3, tensorSize(fShape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(bias_d, biasL3, fShape.count*sizeof(float), bias_offset*sizeof(float)));
	gpuErrchk(cudaMalloc(&out_d, tensorSize(oShapeL3)*sizeof(float)));

	row_ref = (uint32_t) ((TILE_WIDTH-fShape.height+args.strideH)/args.strideH);
	col_ref =  (uint32_t) ((TILE_WIDTH-fShape.width+args.strideW)/args.strideW);
	grid.x = ceil((float)(iShape.width+2*args.padW)/(float)(col_ref*args.strideW));
	grid.y = ceil((float)(iShape.height+2*args.padH)/(float)(row_ref*args.strideH));
	grid.z = 1;

	float * inL4 = (float *) malloc(tensorSize(oShapeL3)*sizeof(float));
	// Launch the kernel
	convLayer_gpu <<< grid, block, TILE_WIDTH*TILE_WIDTH*sizeof(float) >>> (in_d, iShape, filter_d, fShape, out_d, oShapeL3, args, batchSize, bias_offset);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(inL4, out_d, tensorSize(oShapeL3)*sizeof(float), cudaMemcpyDeviceToHost));
	
	bias_offset += fShape.count;
	// Free variables
	cudaFree(in_d);
	cudaFree(out_d);
	cudaFree(filter_d);
	free(in_checkL3);
	free(inL3);
	free(filterL3);
	free(biasL3);

	// ------------------------------- ALEX NET CONV 4 LAYER ----------------------------------------
	iShape = oShapeL3;
	fShape = AlexL4_FilterShape;
	args = AlexL4_ConvArgs;

	TensorShape oShapeL4;
	oShapeL4.height 	= (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShapeL4.width	= (iShape.width  + 2 * args.padW - fShape.width)  / args.strideW + 1;
	oShapeL4.channels	= fShape.count;
	oShapeL4.count 	= batchSize;
		
	float * in_checkL4 = nullptr;
	float * filterL4 = nullptr;
	float * biasL4 = nullptr;

	retVal = makeTensor(&in_checkL4,iShape);
	retVal = makeTensor(&filterL4, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return __UINT32_MAX__;
	}
	retVal = makeVector(&biasL4, fShape.count);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n" ;
		return __UINT32_MAX__;
	}

	gpuErrchk(cudaMalloc(&in_d, tensorSize(iShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(in_d, in_checkL4, tensorSize(iShape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&filter_d, tensorSize(fShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(filter_d, filterL4, tensorSize(fShape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(bias_d, biasL4, fShape.count*sizeof(float), bias_offset*sizeof(float)));
	gpuErrchk(cudaMalloc(&out_d, tensorSize(oShapeL4)*sizeof(float)));

	row_ref = (uint32_t) ((TILE_WIDTH-fShape.height+args.strideH)/args.strideH);
	col_ref =  (uint32_t) ((TILE_WIDTH-fShape.width+args.strideW)/args.strideW);
	grid.x = ceil((float)(iShape.width+2*args.padW)/(float)(col_ref*args.strideW));
	grid.y = ceil((float)(iShape.height+2*args.padH)/(float)(row_ref*args.strideH));
	grid.z = 1;

	float * inL5 = (float *) malloc(tensorSize(oShapeL4)*sizeof(float));
	// Launch the kernel
	convLayer_gpu <<< grid, block, TILE_WIDTH*TILE_WIDTH*sizeof(float) >>> (in_d, iShape, filter_d, fShape, out_d, oShapeL4, args, batchSize, bias_offset);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(inL5, out_d, tensorSize(oShapeL4)*sizeof(float), cudaMemcpyDeviceToHost));
	
	bias_offset += fShape.count;
	// Free variables
	cudaFree(in_d);
	cudaFree(out_d);
	cudaFree(filter_d);
	free(in_checkL4);
	free(inL4);
	free(filterL4);
	free(biasL4);

	// ------------------------------- ALEX NET CONV 5 LAYER ----------------------------------------
	iShape = oShapeL4;
	fShape = AlexL5_FilterShape;
	args = AlexL5_ConvArgs;

	TensorShape oShapeL5;
	oShapeL5.height 	= (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShapeL5.width	= (iShape.width  + 2 * args.padW - fShape.width)  / args.strideW + 1;
	oShapeL5.channels	= fShape.count;
	oShapeL5.count 	= batchSize;
		
	float * in_checkL5 = nullptr;
	float * filterL5 = nullptr;
	float * biasL5 = nullptr;

	retVal = makeTensor(&in_checkL5,iShape);
	retVal = makeTensor(&filterL5, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return __UINT32_MAX__;
	}
	retVal = makeVector(&biasL5, fShape.count);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n" ;
		return __UINT32_MAX__;
	}

	gpuErrchk(cudaMalloc(&in_d, tensorSize(iShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(in_d, in_checkL5, tensorSize(iShape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&filter_d, tensorSize(fShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(filter_d, filterL5, tensorSize(fShape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(bias_d, biasL5, fShape.count*sizeof(float), bias_offset*sizeof(float)));
	gpuErrchk(cudaMalloc(&out_d, tensorSize(oShapeL5)*sizeof(float)));

	row_ref = (uint32_t) ((TILE_WIDTH-fShape.height+args.strideH)/args.strideH);
	col_ref =  (uint32_t) ((TILE_WIDTH-fShape.width+args.strideW)/args.strideW);
	grid.x = ceil((float)(iShape.width+2*args.padW)/(float)(col_ref*args.strideW));
	grid.y = ceil((float)(iShape.height+2*args.padH)/(float)(row_ref*args.strideH));
	grid.z = 1;

	float * inPL3 = (float *) malloc(tensorSize(oShapeL5)*sizeof(float));
	// Launch the kernel
	convLayer_gpu <<< grid, block, TILE_WIDTH*TILE_WIDTH*sizeof(float) >>> (in_d, iShape, filter_d, fShape, out_d, oShapeL5, args, batchSize, bias_offset);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(inPL3, out_d, tensorSize(oShapeL5)*sizeof(float), cudaMemcpyDeviceToHost));
	
	bias_offset += fShape.count;
	// Free variables
	cudaFree(in_d);
	cudaFree(out_d);
	cudaFree(filter_d);
	free(in_checkL5);
	free(inL5);
	free(filterL5);
	free(biasL5);

	// ----------------------------- MAX POOLING 3 LAYER -----------------------------------------
	
	iShape = oShapeL5;
	poolArgs = AlexPL3_poolArgs;
	TensorShape oShapePL3;
	oShapePL3 = {batchSize,iShape.channels,((iShape.height-poolArgs.poolH)/poolArgs.strideH)+1,((iShape.width-poolArgs.poolW)/poolArgs.strideW)+1};
	
	gpuErrchk(cudaMalloc(&out_d_pool, tensorSize(oShapePL3)*sizeof(float)));
	
	grid.x = ceil((float)oShapePL3.width/TILE_WIDTH);
	grid.y = ceil((float)oShapePL3.height/TILE_WIDTH);
	grid.z = iShape.channels;

	float * in_checkPL3 = nullptr;
	retVal = makeTensor(&in_checkPL3,iShape);
	float * out = (float *) malloc(tensorSize(oShapePL3)*sizeof(float));
	gpuErrchk(cudaMalloc(&in_d, tensorSize(iShape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(in_d, in_checkPL3, tensorSize(iShape)*sizeof(float), cudaMemcpyHostToDevice));
	// Launch the kernel
	poolLayer_gpu <<<grid, block>>> (in_d, iShape, out_d_pool, oShapePL3, poolArgs, batchSize);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(out, out_d_pool, tensorSize(oShapePL3)*sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(in_d);
	cudaFree(out_d_pool);
	free(inPL3);
	free(in_checkPL3);

	// ----------------------------------- FLATTEN ----------------------------------------------
	TensorShape flat;
	flat = {1, 1, batchSize, oShapePL3.channels*oShapePL3.height*oShapePL3.width};

	float * inFC1 = (float *) malloc(tensorSize(flat)*sizeof(float));
	for(uint32_t b=0; b<flat.height; b++){
		for(uint32_t c=0; c<oShapePL3.channels; c++){
			for(uint32_t row=0; row<oShapePL3.height; row++){
				for(uint32_t col=0; col<oShapePL3.width; col++){
					uint32_t flat_offset = b*flat.width + col + oShapePL3.width*(row + oShapePL3.height*c);
					*(inFC1 + flat_offset) = out[col + oShapePL3.width*(row + oShapePL3.height*(c + oShapePL3.channels*b))];
				}
			}
		}
	}

	// -------------------------------- FULLY CONNECTED 1 ---------------------------------------
	GemmLayerArgs argsFC = {32, 32, 1};
	
	TensorShape FC1_shape;
	FC1_shape = {1,1,flat.width,4096};

	TensorShape inFC2_shape;
	inFC2_shape = {flat.count, flat.channels, flat.height, FC1_shape.width};
	
	float * FC1 = nullptr;
	makeTensor(&FC1, FC1_shape);

	float *inFC2 = (float *) malloc(tensorSize(inFC2_shape)*sizeof(float));
	uint64_t sharedMemSize = 2*(argsFC.tileH*argsFC.tileW*sizeof(float));
	
	float *a_d, *b_d, *c_d;
	gpuErrchk(cudaMalloc(&a_d, tensorSize(flat)*sizeof(float)));
	gpuErrchk(cudaMemcpy(a_d, inFC1, tensorSize(flat)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&b_d, tensorSize(FC1_shape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(b_d, FC1, tensorSize(FC1_shape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&c_d, tensorSize(inFC2_shape)*sizeof(float)));

	dim3 block_fc(argsFC.tileH, argsFC.tileW);
	dim3 grid_fc(ceil((float)inFC2_shape.height/(float)argsFC.tileH), ceil((float)inFC2_shape.width/(float)argsFC.tileW));

	gemmLayer_gpu <<< grid_fc, block_fc, sharedMemSize >>> (a_d, flat, b_d, FC1_shape, c_d, inFC2_shape, argsFC);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpy(inFC2, c_d, tensorSize(inFC2_shape)*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	free(inFC1);
	free(FC1);
	
	// -------------------------------- FULLY CONNECTED 2 ---------------------------------------
	TensorShape FC2_shape;
	FC2_shape = {1,1,inFC2_shape.width,4096};

	TensorShape inFC3_shape;
	inFC3_shape = {inFC2_shape.count, inFC2_shape.channels, inFC2_shape.height, FC2_shape.width};
	
	float * FC2 = nullptr;
	makeTensor(&FC2, FC2_shape);

	float *inFC3 = (float *) malloc(tensorSize(inFC3_shape)*sizeof(float));
	
	gpuErrchk(cudaMalloc(&a_d, tensorSize(inFC2_shape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(a_d, inFC2, tensorSize(inFC2_shape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&b_d, tensorSize(FC2_shape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(b_d, FC2, tensorSize(FC2_shape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&c_d, tensorSize(inFC3_shape)*sizeof(float)));

	grid_fc.x = ceil((float)inFC3_shape.height/(float)argsFC.tileH);
	grid_fc.y = ceil((float)inFC3_shape.width/(float)argsFC.tileW);

	gemmLayer_gpu <<< grid_fc, block_fc, sharedMemSize >>> (a_d, inFC2_shape, b_d, FC2_shape, c_d, inFC3_shape, argsFC);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpy(inFC3, c_d, tensorSize(inFC3_shape)*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	free(inFC2);
	free(FC2);

	// ---------------------------------- SOFTMAX ---------------------------------------------
	TensorShape FC3_shape;
	FC3_shape = {1,1,inFC3_shape.width,1000};

	TensorShape inFC4_shape;
	inFC4_shape = {inFC3_shape.count, inFC3_shape.channels, inFC3_shape.height, FC3_shape.width};
	
	float * FC3 = nullptr;
	makeTensor(&FC3, FC3_shape);

	float *softmax = (float *) malloc(tensorSize(inFC4_shape)*sizeof(float));
	
	gpuErrchk(cudaMalloc(&a_d, tensorSize(inFC3_shape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(a_d, inFC3, tensorSize(inFC3_shape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&b_d, tensorSize(FC3_shape)*sizeof(float)));
	gpuErrchk(cudaMemcpy(b_d, FC3, tensorSize(FC3_shape)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&c_d, tensorSize(inFC4_shape)*sizeof(float)));

	grid_fc.x = ceil((float)inFC4_shape.height/(float)argsFC.tileH);
	grid_fc.y = ceil((float)inFC4_shape.width/(float)argsFC.tileW);

	gemmLayer_gpu <<< grid_fc, block_fc, sharedMemSize >>> (a_d, inFC3_shape, b_d, FC3_shape, c_d, inFC4_shape, argsFC);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpy(softmax, c_d, tensorSize(inFC4_shape)*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	free(inFC3);
	free(FC3);
	free(softmax);

	return 0;
}
