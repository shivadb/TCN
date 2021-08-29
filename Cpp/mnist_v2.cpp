#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>

#include "logger.h"
#include "readerwriterqueue.h"

#define DLACore -1
#define IMGSIZE 784
#define OUTBUFFERLEN 10000
#define ENGINEPATH "/runfa/shivb/TCN/TCN/mnist_pixel/models_trt/aug_k7l6_trt_amax.engine"
#define FP16 false
#define PRINTOUT false

using namespace nvinfer1;
using namespace std;

typedef unsigned char uchar;

struct TRTDestroy {
    template<class T> 
    void operator()(T* obj) const {
        obj->destroy();
    }
};

template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

class MNISTTRTBuffer {
    public:
        void* buffers[2];
        int inputIndex;
        int outputIndex;

        MNISTTRTBuffer(int inTRTIdx, int outTRTIdx, int* outBufferAddr){
            inputIndex = inTRTIdx;
            outputIndex = outTRTIdx;
            cudaMalloc(&buffers[inputIndex], IMGSIZE*sizeof(float));
            buffers[outputIndex] = outBufferAddr;
        }

        void updateOutBufferAddr(int* outBufferAddr){
            buffers[outputIndex] = outBufferAddr;
        }

        void* getInBufferPtr(){
            return buffers[inputIndex];
        }

        ~MNISTTRTBuffer() {
            cudaFree(buffers[inputIndex]);
        }
        
};

uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

uchar* read_mnist_labels(string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

void printDigit(uchar* img)
{
    for (int i = 0; i < IMGSIZE; i++)
    {
        if (i % 28 == 0)
        {
            cout << endl;
        }

        if ((int)img[i] > 0)
        {
            cout << "# ";
        }
        else
        {
            cout << ". ";
        }
        
    }
    cout << endl;
}

void printNormalizedDigit(float* img)
{
    for (int i = 0; i < IMGSIZE; i++)
    {
        if (i % 28 == 0)
        {
            cout << endl;
        }

        if (img[i] > 0.1307)
        {
            cout << "# ";
        }
        else
        {
            cout << ". ";
        }
        
    }
    cout << endl;
}

float** charToFloatArr (uchar** ds, int numImgs)
{
    float** arr = new float*[numImgs];

    for (int i = 0; i < numImgs; i++)
    {
        arr[i] = new float[IMGSIZE];
        for (int j = 0; j < IMGSIZE; j++)
        {
            arr[i][j] = ((int)ds[i][j]/255.0 - 0.1307) / 0.3081;
        }
    }

    return arr;
}

float* charToFloatPinnedArr (uchar** ds, int numImgs)
{
    float* pinnedArr;
    cudaMallocHost((void **) &pinnedArr, IMGSIZE * numImgs * sizeof(float));

    for (int i = 0; i < numImgs; i++)
    {
        for (int j = 0; j < IMGSIZE; j++)
        {
            pinnedArr[i*IMGSIZE + j] = ((int)ds[i][j]/255.0 - 0.1307) / 0.3081;
        }
    }

    return pinnedArr;
}

int main(int argc, char const *argv[])
{
    // Read MNIST
    string test_image_path = "/runfa/shivb/TCN/data/mnist/MNIST/raw/t10k-images-idx3-ubyte";
    string test_labels_path = "/runfa/shivb/TCN/data/mnist/MNIST/raw/t10k-labels-idx1-ubyte";
    int num_images, img_size, num_labels;
    
    uchar** ds = read_mnist_images(test_image_path, num_images, img_size);
    uchar* lbls = read_mnist_labels(test_labels_path, num_labels);

    cout << "Number of Images: " << num_images << ", Image Size: " << img_size << ", Number of Labels: " << num_labels << endl;
    
    // float** normalizedData = charToFloatArr(ds, OUTBUFFERLEN);
    float* pinnedNormData = charToFloatPinnedArr(ds, OUTBUFFERLEN);
    // printDigit(ds[0]);
    // printNormalizedDigit(normalizedData[0]);

    // Load TRT Engine
    cout << ENGINEPATH << endl;

    TRTUniquePtr<IRuntime> runtime {nullptr};
    TRTUniquePtr<ICudaEngine> engine {nullptr};
    TRTUniquePtr<IExecutionContext> context {nullptr};

    ifstream engineFile(ENGINEPATH, ios::binary);
    if(!engineFile)
    {
        cout << "Error opening engine file." << endl;
        return 0;
    }
    
    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    // std::cout << "fsize: " << fsize << std::endl;

    vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    if(!engineFile)
    {
        cout << "Error loading engine file." << endl;
        return 0;
    }

    // TrtUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
    runtime.reset(createInferRuntime(gLogger.getTRTLogger()));

    if (DLACore != -1)
    {
        runtime->setDLACore(DLACore);
    }

    // TRTUniquePtr<ICudaEngine> engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    
    context.reset(engine->createExecutionContext());

    int inputIndex = engine->getBindingIndex("input_0");
    int outputIndex = engine->getBindingIndex("output_0");

    int* outputs;
    cudaHostAlloc((void**) &outputs, OUTBUFFERLEN*sizeof(int), cudaHostAllocMapped);

    // float* pinnedInput;
    // cudaMallocHost((void **) &pinnedInput, IMGSIZE * sizeof(float));

    // void* buffers[2];
    // cudaMalloc(&buffers[inputIndex], IMGSIZE*sizeof(float));
    MNISTTRTBuffer trtBuffer(inputIndex, outputIndex, nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cout << "Started Running Samples" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < OUTBUFFERLEN; i++)
    {
        // cout << "True Label: " << (int)lbls[i] << endl;
        trtBuffer.updateOutBufferAddr(outputs + i);

        // cudaMemcpyAsync(buffers[inputIndex], pinnedNormData + IMGSIZE*i, IMGSIZE*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(trtBuffer.getInBufferPtr(), pinnedNormData + IMGSIZE*i, IMGSIZE*sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueue(1, trtBuffer.buffers, stream, nullptr);

    }
    cudaStreamSynchronize(stream);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> elapsedTime = end - start;

    if (PRINTOUT)
    {
        cout << "Output: " << endl;
        for (int i = 0; i < OUTBUFFERLEN; i++) cout << outputs[i] << " ";
        cout << endl;
    }
    
    std::cout << "Average inference time per image: " << elapsedTime.count()/OUTBUFFERLEN << "ms" << std::endl;

    cudaStreamDestroy(stream);
    // cudaFreeHost(pinnedInput);
    cudaFreeHost(pinnedNormData);
    // cudaFree(buffers[inputIndex]);
    cudaFreeHost(outputs);

    return 0;
}
