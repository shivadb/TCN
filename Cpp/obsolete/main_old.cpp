#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>

// #include "sampleUtils.h"
#include "logger.h"

#define ENGINEPATH "/runfa/shivb/TCN/TCN/mnist_pixel/models_trt/aug_k7l6_trt_onnx_amax_fp16.engine"
#define DLACore -1
#define NUMSAMPLES 10000

using namespace nvinfer1;
// using namespace sample;


struct TRTDestroy {
    template<class T> 
    void operator()(T* obj) const {
        obj->destroy();
    }
};

// struct gen_rand { 
//     float range;
// public:
//     gen_rand(float r=1.0) : range(r) {}
//     float operator()() { 
//         return (rand()/(float)RAND_MAX) * range;
//     }
// };

struct gen_rand { 
    __half range;
public:
    gen_rand(__half r= __float2half(1.0)) : range(r) {}
    __half operator()() { 
        return (__float2half(rand())/__double2half(RAND_MAX)) * range;
    }
};


template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

template<typename T>
void printVec(std::vector <T> const &a) {
   for(int i=0; i < a.size(); i++)
   std::cout << a.at(i) << ' ';
   std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
    std::cout << ENGINEPATH << std::endl;

    TRTUniquePtr<IRuntime> runtime {nullptr};
    TRTUniquePtr<ICudaEngine> engine {nullptr};
    TRTUniquePtr<IExecutionContext> context {nullptr};

    std::ifstream engineFile(ENGINEPATH, std::ios::binary);
    if(!engineFile)
    {
        std::cout << "Error opening engine file." << std::endl;
        return 0;
    }
    
    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    // std::cout << "fsize: " << fsize << std::endl;

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    if(!engineFile)
    {
        std::cout << "Error loading engine file." << std::endl;
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

    // std::vector<int> output(1, 0);
    //outputs 0s -> 1, 1s -> 0, 0.1s -> 8
    // std::vector<float> testInput(784, 0.1);
    std::vector<__half>testInput(784, __float2half(1));
    // std::generate_n(testInput.begin(), 784, gen_rand());
    // std::cout << "Test vector: " << testInput.data() << std::endl;
    // printVec(testInput);

    void* buffers[2];
    int inputIndex = engine->getBindingIndex("input_0");
    int outputIndex = engine->getBindingIndex("output_0");

    // std::cout << "Input idx: " << inputIndex << " Output idx: " << outputIndex << std::endl;

    // cudaMalloc(&buffers[inputIndex], 784 * sizeof(float));
    cudaMalloc(&buffers[inputIndex], 784 * sizeof(__half));
    // cudaMalloc(&buffers[outputIndex], 1 * sizeof(int));
    cudaHostAlloc(&buffers[outputIndex], 1*sizeof(int), cudaHostAllocDefault);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t end;
    cudaEventCreate(&end);
    float totalTime = 0.0;

    for (int i = 0; i < NUMSAMPLES; ++i)
    {
        float elapsedTime;

        // std::generate_n(testInput.begin(), 784, gen_rand());

        cudaEventRecord(start, stream);

        // cudaMemcpyAsync(buffers[inputIndex], testInput.data(), 784 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buffers[inputIndex], testInput.data(), 784 * sizeof(__half), cudaMemcpyHostToDevice, stream);
        context->enqueue(1, buffers, stream, nullptr);
        // cudaMemcpyAsync(output.data(), buffers[outputIndex], 1 * sizeof(int), cudaMemcpyDeviceToHost, stream);

        cudaEventRecord(end, stream);
        cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&elapsedTime, start, end);

        totalTime += elapsedTime;

        if (i % 1000 == 0)
        {
            std::cout << "Finished running " << i << " samples" << std::endl;
            std::cout << ((int*)buffers[outputIndex])[0] << std::endl;
            // printVec(output);
        }
    }

    std::cout << "Average inference time per sample: " << totalTime/NUMSAMPLES << "ms" << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    // cudaFreeHost(buffers[outputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}

