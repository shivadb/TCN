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

#define ENGINEPATH "/runfa/shivb/TCN/TCN/mnist_pixel/models_trt/aug_k7l6_trt_amax.engine"
#define DLACore -1
#define NUMSAMPLES 25000
#define FP16 false
#define OUTBUFFERLEN 10
// #define INTYPE "pinned" //one of "regular", "pinned", or "zero"
// #define OUTTYPE "zero" //one of "regular" or "zero"

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

    int inputIndex = engine->getBindingIndex("input_0");
    int outputIndex = engine->getBindingIndex("output_0");

    // std::vector<int> outputs(OUTBUFFERLEN, 0);
    int* outputs;
    cudaHostAlloc((void**) &outputs, OUTBUFFERLEN*sizeof(int), cudaHostAllocMapped);
    
    float* pinnedInput;
    cudaMallocHost((void **) &pinnedInput, 784 * sizeof(float));

    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], 784*sizeof(float));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < 10; i++)
    {
        float inputVal = i/10.0;
        for (int j=0; j<784; j++) pinnedInput[j] = inputVal;

        buffers[outputIndex] = &outputs[i];

        cudaMemcpyAsync(buffers[inputIndex], pinnedInput, 784*sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueue(1, buffers, stream, nullptr);

        cudaStreamSynchronize(stream);
    }

    for (int i = 0; i < OUTBUFFERLEN; i++) std::cout << outputs[i] << " ";
    std::cout << std::endl;


    cudaStreamDestroy(stream);
    cudaFreeHost(pinnedInput);
    cudaFree(buffers[inputIndex]);
    cudaFreeHost(outputs);


    return 0;
}

