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
#define INTYPE "regualr" //one of "regular", "pinned", or "zero"
#define OUTTYPE "regular" //one of "regular" or "zero"

using namespace nvinfer1;
// using namespace sample;


struct TRTDestroy {
    template<class T> 
    void operator()(T* obj) const {
        obj->destroy();
    }
};

template<typename T>
struct gen_rand { 
    T range;
public:
    gen_rand(T r=1.0) : range(r) {}
    T operator()() { 
        return (rand()/(T)RAND_MAX) * range;
    }
};

template<>
struct gen_rand<__half> {
    __half range;
public:
    gen_rand(__half r = __float2half(1.0)) : range(r) {}
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

template<typename T>
std::vector<T> initializeVec(int l, float val)
{
    std::vector<T> vec(l, val);
    return vec;
}

template <> 
std::vector<__half> initializeVec<__half>(int l, float val) 
{
    std::vector<__half> vec(l, __float2half(val));
    return vec;
}

template<typename T>
void testRuntime(TRTUniquePtr<IExecutionContext> &context, int inputIndex, int outputIndex)
{
    std::vector<int> output(1, 0);
    std::vector<T> testInput = initializeVec<T>(784, 0);

    T* pinnedTestInput;
    cudaMallocHost((void **) &pinnedTestInput, 784 * sizeof(T));

    void* buffers[2];

    if (INTYPE == "zero")
    {
        std::cout << "Using zerocopy input buffer" << std::endl;
        cudaHostAlloc(&buffers[inputIndex], 784 * sizeof(T), cudaHostAllocWriteCombined);
    }
    else
    {
        cudaMalloc(&buffers[inputIndex], 784 * sizeof(T));
    }
    

    if (OUTTYPE == "zero")
    {
        std::cout << "Using zerocopy output buffer" << std::endl;
        cudaHostAlloc(&buffers[outputIndex], 1*sizeof(int), cudaHostAllocMapped);
    }
    else
    {
        std::cout << "Using regular output buffer" << std::endl;
        cudaMalloc(&buffers[outputIndex], 1 * sizeof(int));      
    }
    
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

        // std::generate_n(testInput.begin(), 784, gen_rand<T>());

        cudaEventRecord(start, stream);

        if (INTYPE == "regular")
        {
            cudaMemcpyAsync(buffers[inputIndex], testInput.data(), 784 * sizeof(T), cudaMemcpyHostToDevice, stream);
        }
        else if (INTYPE == "pinned")
        {
            cudaMemcpyAsync(buffers[inputIndex], pinnedTestInput, 784 * sizeof(T), cudaMemcpyHostToDevice, stream);
        }
        context->enqueue(1, buffers, stream, nullptr);

        if (OUTTYPE != "zero")
        {
            cudaMemcpyAsync(output.data(), buffers[outputIndex], 1 * sizeof(int), cudaMemcpyDeviceToHost, stream);
        }

        cudaEventRecord(end, stream);
        // cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&elapsedTime, start, end);

        totalTime += elapsedTime;

        if (i % 1000 == 0)
        {
            std::cout << "Finished running " << i << " samples" << std::endl;
        }
    }

    std::cout << "Average inference time per sample: " << totalTime/NUMSAMPLES << "ms" << std::endl;

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFreeHost(pinnedTestInput);

    if (INTYPE == "zero")
    {
        cudaFreeHost(buffers[inputIndex]);
    }
    else
    {
        4444
    }
    
    if (OUTTYPE == "zero")
    {
        cudaFreeHost(buffers[outputIndex]);
    }
    else
    {
        cudaFree(buffers[outputIndex]);
    }
}

template<typename T>
void verifyOutput(TRTUniquePtr<IExecutionContext> &context, int inputIndex, int outputIndex, T testValue)
{
    std::vector<int> output(1, 0);
    std::vector<T> testInput(784, testValue);
    void* buffers[2];

    T* pinnedTestInput;
    cudaMallocHost((void **) &pinnedTestInput, 784 * sizeof(T));
    for (int i=0; i<784; i++) pinnedTestInput[i] = testValue;

    if (INTYPE == "zero")
    {
        std::cout << "Using zerocopy input buffer" << std::endl;
        cudaHostAlloc(&buffers[inputIndex], 784 * sizeof(T), cudaHostAllocWriteCombined);
        for (int i=0; i<784; i++) ((T*)buffers[inputIndex])[i] = testValue;
    }
    else
    {
        cudaMalloc(&buffers[inputIndex], 784 * sizeof(T));
    }
    

    if (OUTTYPE == "zero")
    {
        std::cout << "Using zerocopy output buffer" << std::endl;
        cudaHostAlloc(&buffers[outputIndex], 1*sizeof(int), cudaHostAllocMapped);
    }
    else
    {
        std::cout << "Using regular output buffer" << std::endl;
        cudaMalloc(&buffers[outputIndex], 1 * sizeof(int));      
    }


    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    if (INTYPE == "regular")
    {
        cudaMemcpyAsync(buffers[inputIndex], testInput.data(), 784 * sizeof(T), cudaMemcpyHostToDevice, stream);
    }
    else if (INTYPE == "pinned")
    {
        cudaMemcpyAsync(buffers[inputIndex], pinnedTestInput, 784 * sizeof(T), cudaMemcpyHostToDevice, stream);
    }
    context->enqueue(1, buffers, stream, nullptr);

    if (OUTTYPE == "zero")
    {
        cudaStreamSynchronize(stream);
        std::cout << ((int*)buffers[outputIndex])[0] << std::endl;
    }
    else
    {
        cudaMemcpyAsync(output.data(), buffers[outputIndex], 1 * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        printVec(output);
    }

    cudaStreamDestroy(stream);
    cudaFreeHost(pinnedTestInput);

    if (INTYPE == "zero")
    {
        cudaFreeHost(buffers[inputIndex]);
    }
    else
    {
        cudaFree(buffers[inputIndex]);
    }
    
    if (OUTTYPE == "zero")
    {
        cudaFreeHost(buffers[outputIndex]);
    }
    else
    {
        cudaFree(buffers[outputIndex]);
    }

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

    if (FP16)
    {
        std::cout << "fp16" << std::endl;
        // verifyOutput<__half>(context, inputIndex, outputIndex, __float2half(0));
        // verifyOutput<__half>(context, inputIndex, outputIndex, __float2half(0.1));
        // verifyOutput<__half>(context, inputIndex, outputIndex, __float2half(1));
        testRuntime<__half>(context, inputIndex, outputIndex);
    }
    else
    {
        // verifyOutput<float>(context, inputIndex, outputIndex, 0);
        // verifyOutput<float>(context, inputIndex, outputIndex, 0.1);
        // verifyOutput<float>(context, inputIndex, outputIndex, 1);
        testRuntime<float>(context, inputIndex, outputIndex);
    }

    return 0;
}

