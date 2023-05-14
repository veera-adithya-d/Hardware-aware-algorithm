#include "cpuLib.h"

void dbprintf(const char* fmt...) {
	#ifndef DEBUG_PRINT_DISABLE
		va_list args;

		va_start(args, fmt);
		int result = vprintf(fmt, args);
		// printf(fmt, ...);
		va_end(args);
	#endif
	return;
}

void vectorInit(float* v, int size) {
	for (int idx = 0; idx < size; ++idx) {
		v[idx] = (float)(rand() % 100);
	}
}

int verifyVector(float* a, float* b, float* c, float scale, int size) {
	int errorCount = 0;
	for (int idx = 0; idx < size; ++idx) {
		if (c[idx] != scale * a[idx] + b[idx]) {
			++errorCount;
			#ifndef DEBUG_PRINT_DISABLE
				std::cout << "Idx " << idx << " expected " << scale * a[idx] + b[idx] 
					<< " found " << c[idx] << " = " << a[idx] << " + " << b[idx] << "\n";
			#endif
		}
	}
	return errorCount;
}

void printVector(float* v, int size) {
	int MAX_PRINT_ELEMS = 5;
	std::cout << "Printing Vector : \n"; 
	for (int idx = 0; idx < std::min(size, MAX_PRINT_ELEMS); ++idx){
		std::cout << "v[" << idx << "] : " << v[idx] << "\n";
	}
	std::cout << "\n";
}

std::ostream& operator<< (std::ostream &o,ImageDim imgDim) {
	return (
		o << "Image : " << imgDim.height  << " x " << imgDim.channels << " x "
			<< imgDim.channels << " x " << imgDim.pixelSize << " " 
	);
}

int loadBytesImage(std::string bytesFilePath, ImageDim &imgDim, uint8_t ** imgData ) {
	#ifndef DEBUG_PRINT_DISABLE
		std::cout << "Opening File @ \'" << bytesFilePath << "\' \n";
	#endif

	std::ifstream bytesFile;

	bytesFile.open(bytesFilePath.c_str(), std::ios::in | std::ios::binary);

	if (! bytesFile.is_open()) {
		std::cout << "Unable to open \'" << bytesFilePath << "\' \n";
		return -1;
	}

	ImageDim_t fileDim;
	bytesFile.read((char *) &fileDim, sizeof(fileDim));

	std::cout << "Found " << fileDim.height << " x " << fileDim.width
		<< " x " << fileDim.channels << " x " << fileDim.pixelSize << " \n";
	
	uint64_t numBytes = fileDim.height * fileDim.width * fileDim.channels;
	*imgData = (uint8_t *) malloc(numBytes * sizeof(uint8_t));
	if (imgData == nullptr) {
		std::cout << "Unable to allocate memory for image data \n";
		return -2;
	}

	bytesFile.read((char *) *imgData, numBytes * sizeof(uint8_t));

	std::cout << "Read " << bytesFile.gcount() << " bytes \n" ;

	imgDim.height		= fileDim.height;
	imgDim.width		= fileDim.width;
	imgDim.channels		= fileDim.channels;
	imgDim.pixelSize	= fileDim.pixelSize;

	bytesFile.close();
	
	return bytesFile.gcount();

}

int writeBytesImage (std::string outPath, ImageDim &imgDim, uint8_t * outData) {
	std::ofstream bytesFile;

	bytesFile.open(outPath.c_str(), std::ios::out | std::ios::binary);

	if (! bytesFile.is_open()) {
		std::cout << "Unable to open \'" << outPath << "\' \n";
		return -1;
	}

	uint64_t numBytes = imgDim.height * imgDim.width * imgDim.channels;
	bytesFile.write((char*) &imgDim, sizeof(imgDim));
	bytesFile.write((char *) outData, numBytes * sizeof(uint8_t));

	bytesFile.close();

}

std::ostream& operator << (std::ostream &o, const TensorShape & t) {
	return (
		o << "Tensor : " 
		<< t.count << " x " << t.channels << " x "
		<< t.height << " x " << t.width << " "
	);

}

uint64_t tensorSize (const TensorShape & t) {
	uint64_t size =  ( (uint64_t)t.count * t.channels * t.height * t.width );
	if (size == 0) {
		std::cout << "Invalid shape parameters \n";
	}
	return size;
}

int makeTensor (float ** t, TensorShape & shape) {
	if (*t != nullptr) {
		std::cout << "Pointer already points to memory ! \n";
		return -1;
	}

	if (shape.count == 0) {
		std::cout << " Shape has invalid count (4th dim) - setting to 1 \n";
		shape.count = 1;
	}

	uint64_t tensorSize = shape.height * shape.width * shape.channels * shape.count;
	*t = (float *) malloc (tensorSize * sizeof(float));

	if (*t == nullptr) {
		std::cout << "Malloc failed ! \n";
		return -2;
	}

	float * m = * t;
	uint64_t offset;

	std::random_device random_device;
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	//	Implement NCHW layout
	for (uint32_t count = 0; count < shape.count; ++ count) {
		for (uint32_t chIdx = 0; chIdx < shape.channels; ++ chIdx ) {
			for (uint32_t rowIdx = 0; rowIdx < shape.height; ++ rowIdx) {
				for (uint32_t colIdx = 0; colIdx < shape.width; ++ colIdx) {
					offset = chIdx * shape.height * shape.width + rowIdx * shape.width + colIdx;
					m[offset] = dist(random_device);
				}
			}
		}
	}
	return 0;
}

int makeVector (float ** v, uint64_t size) {
	if (*v != nullptr) {
		std::cout << "Pointer already points to memory ! \n";
		return -1;
	}

	*v = (float *) malloc (size * sizeof(float));

	float * m = * v;

	std::random_device random_device;
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	//	Implement NCHW layout
	for (uint64_t idx = 0; idx < size; ++ idx) {
		m[idx] = dist(random_device);
	}
	return 0;
}