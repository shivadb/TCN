#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <typeinfo>
#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>

// #include "sampleUtils.h"
#include "logger.h"

#define ENGINEPATH "/runfa/shivb/TCN/TCN/mnist_pixel/models/aug_k7l6.engine"
#define DLACore -1

using namespace nvinfer1;
// using namespace sample;


struct TRTDestroy {
    template<class T> 
    void operator()(T* obj) const {
        obj->destroy();
    }
};

template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

void printVec(std::vector <float> const &a) {
   for(int i=0; i < a.size(); i++)
   std::cout << a.at(i) << ' ';
   std::cout << std::endl;
}

int main(int argc, char const *argv[])
{

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

    std::vector<float> output(10, 0);
    std::vector<float> testInput(784, 1);
    // std::cout << "Test vector: " << testInput.data() << std::endl;
    // printVec(testInput);

    void* buffers[2];
    int inputIndex = engine->getBindingIndex("input_0");
    int outputIndex = engine->getBindingIndex("output_0");

    // std::cout << "Input idx: " << inputIndex << " Output idx: " << outputIndex << std::endl;

    cudaMalloc(&buffers[inputIndex], 784 * sizeof(float));
    cudaMalloc(&buffers[outputIndex], 10 * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(buffers[inputIndex], testInput.data(), 784 * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueue(1, buffers, stream, nullptr);
    cudaMemcpyAsync(output.data(), buffers[outputIndex], 10 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::cout << "Output: " << std::endl;
    printVec(output);

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    return 0;
}

