#ifndef CPU_LIB_H
#define CPU_LIB_H

	#include <iostream>
	#include <cstdlib>
	#include <ctime>
	#include <random>
    #include <iomanip>
	#include <chrono>
	#include <cstring>
	#include <cstdarg>
	#include <fstream>
	#include <vector>
	#include <algorithm>
	
	// Uncomment this to suppress console output
	// #define DEBUG_PRINT_DISABLE

	extern void dbprintf(const char* fmt...);


	extern void vectorInit(float* v, int size);
	extern int verifyVector(float* a, float* b, float* c, float scale, int size);
	extern void printVector(float* v, int size);
	
	typedef struct ImageDim_t
	{
		uint32_t height;
		uint32_t width;
		uint32_t channels;
		uint32_t pixelSize;
	} ImageDim;

	extern std::ostream& operator<< (std::ostream &o,ImageDim imgDim);
	
	/**
	 * @brief 
	 * 
	 * @param bytesFilePath 
	 * @param imgDim 
	 * @param imgData 
	 * @return int 
	 */
	extern int loadBytesImage(std::string bytesFilePath, ImageDim &imgDim, uint8_t ** imgData);

	extern int writeBytesImage(std::string outPath, ImageDim &imgDim, uint8_t * outData);

	typedef struct MedianFilterArgs_t {
		uint32_t filterH;
		uint32_t filterW;
	} MedianFilterArgs;

	enum class PoolOp{MaxPool, AvgPool, MinPool};

	extern std::ostream& operator<< (std::ostream &o,PoolOp op);

	typedef struct TensorShape_t {
		uint32_t count;		//	4th dimension	-	Quite unimaginative .. I know
		uint32_t channels;	//	3rd dimension
		uint32_t height;	//	Height = # rows	-	2nd dimension
		uint32_t width;		//	Width = # cols	-	1st dimension
	} TensorShape;

	extern std::ostream& operator << (std::ostream &o, const TensorShape & t);

	extern uint64_t tensorSize (const TensorShape & t);
	extern uint64_t tensorSize1D (const TensorShape & t);
	extern uint64_t tensorSize2D (const TensorShape & t);
	extern uint64_t tensorSize3D (const TensorShape & t);

	typedef struct PoolLayerArgs_t {
		PoolOp opType;
		uint32_t poolH;		//	pooling rows
		uint32_t poolW;		//	pooling cols
		uint32_t strideH;
		uint32_t strideW;
	} PoolLayerArgs;

	typedef struct ConvLayerArgs_t {
		uint32_t padH;
		uint32_t padW;
		uint32_t strideH;
		uint32_t strideW;
		bool activation;
	} ConvLayerArgs;

	typedef struct GemmLayerArgs_t {
		uint32_t tileH;
		uint32_t tileW;
		uint32_t subTileCount;
	} GemmLayerArgs;

	extern int makeTensor (float ** t, TensorShape & shape);
	extern int makeVector (float ** v, uint64_t size);

	const TensorShape AlexL1_InShape 		= {1, 3, 227, 227};
	const TensorShape AlexL1_FilterShape	= {96, 3, 11, 11};
	const ConvLayerArgs AlexL1_ConvArgs 	= {0, 0, 4, 4, false};

	const PoolLayerArgs AlexPL1_poolArgs = {PoolOp::MaxPool, 3, 3, 2, 2};

	const TensorShape AlexL2_InShape 		= {1, 96, 27, 27};
	const TensorShape AlexL2_FilterShape	= {256, 96, 5, 5};
	const ConvLayerArgs AlexL2_ConvArgs 	= {2, 2, 1, 1, false};

	const PoolLayerArgs AlexPL2_poolArgs = {PoolOp::MaxPool, 3, 3, 2, 2};

	const TensorShape AlexL3_InShape 		= {1, 256, 27, 27};
	const TensorShape AlexL3_FilterShape	= {384, 256, 3, 3};
	const ConvLayerArgs AlexL3_ConvArgs 	= {1, 1, 1, 1, false};

	const TensorShape AlexL4_InShape 		= {1, 384, 13, 13};
	const TensorShape AlexL4_FilterShape	= {384, 384, 3, 3};
	const ConvLayerArgs AlexL4_ConvArgs 	= {1, 1, 1, 1, false};

	const TensorShape AlexL5_InShape 		= {1, 384, 13, 13};
	const TensorShape AlexL5_FilterShape	= {256, 384, 3, 3};
	const ConvLayerArgs AlexL5_ConvArgs 	= {1, 1, 1, 1, false};

	const PoolLayerArgs AlexPL3_poolArgs = {PoolOp::MaxPool, 3, 3, 2, 2};

	#define IDX2R(r, c, cDim) ((r) * (cDim) + (c))

	void printTensor (float * t, TensorShape shape);

#endif
