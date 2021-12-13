# TCN 
Deployment pipeline for TCN model used on streaming data. The pipeline is demonstrated using MNIST data for simplicity.

Original implementation: https://github.com/locuslab/TCN

An image generated from the provided Dockerfile in `Environment\Dockerfile` can be used to reproduce all experiments.

## Model Training
### Original Implementation
-    Convert 28x28 image to 1D 784 vector
-   Use the final element of the output vector to train the network
-   Expects the full 784 vector to perform inference; is not trained to make intermediate predictions

### Streaming Data Implementation
-   Idea is to allow the model to converge to the correct classification as each pixel is provided
-   Training images are randomly clipped so the model is trained to classify with missing data

	-   Maximum clipped size is set to 1/4 of the model's receptive field. This means the model should be able to converge to the correct classification once it has 3/4 of the data for the current image
	-   The clipping is performed at the beginning of the image and the end of another random digit is stitched in  place of the missing data to simulate streaming data

-   Examples:

![Augmented Images](https://github.com/shivadb/TCN/blob/master/docs/augmented_train_samples.PNG)

Training can be performed using `pmnist_aug_test.py`. Following is an example command:
	`python pmnist_aug_test.py --ksize 7 --levels 6 --savemodel --modelname k7l6 --seqaugment`

## Model Evaluation
### Qualitative
![Qualitative Demo](https://github.com/shivadb/TCN/blob/master/docs/augmented_demo.gif)

### Quantitative
Original Implementation:

-   The original TCN implementation evaluates accuracy by comparing the output produced after processing all 784 pixels
-   The [paper](https://arxiv.org/abs/1803.01271) reports a test accuracy of 99.0% on MNIST data
-   When the augmented model is evaluated using this method, we obtain a test accuracy of 93.0%

Sequential Implementation:

-   Our model is designed to output a prediction after processing each pixel. Due to the augmentation performed for training, it makes more sense to consider the last n outputs to compute the final prediction
-   In the below table we consider the last n outputs before the full image is seen by the model, and take the highest occurring output as the final prediction for the current digit:

```markdown
| n   	| Accuracy (%) 	|
|-----	|--------------	|
| 1   	| 93.0         	|
| 50  	| 96.9         	|
| 100 	| 97.9         	|
| 150 	| 98.2         	|
| 200 	| 98.3         	|
```

The model can be qualitatively verified using the following Ipython notebook: `Torch Demo.ipynb`

The model can be quantitatively evaluated using: `Accuracy Test.ipynb`

# Inference Optimization

There are various approaches for inference optimization:
- ONNX Runtime
- TensorRT Engine

## Building a TensorRT Engine
A TensorRT engine can be built using `convert_trt.py`. 
Ex: `python convert_trt.py --mdlname aug_k7l6 --applymax --onnx`

This serialized engine can be used with both Python and C++ APIs. Note that this engine should be built on the inference device directly.

## C++ Inference and Multithreading
More details on the approach can be found in `docs\TCN.pdf`. The following results are generated using `Cpp\mnist_multistream_v3.cpp`

`Cpp\compile.sh` can be used to compile the C++ files in the provided Docker environment. 

The following results are obtained by executing the multistreaming and multithreading  implementation in `mnist_multistream_v3.cpp` on a GTX 1080 GPU.


```markdown
|     HDD to Pinned Mem   (# threads)    	|     CUDA Copy and   Execution (# threads)    	|     Avg Execution Time   (ms)    	|
|----------------------------------------	|----------------------------------------------	|----------------------------------	|
|     1                                  	|     1                                        	|     0.39                         	|
|     1                                  	|     2                                        	|     0.33                         	|
|     2                                  	|     2                                        	|     0.33                         	|
|     3                                  	|     2                                        	|     0.33                         	|
|     4                                  	|     2                                        	|     0.32                         	|
|     6                                  	|     2                                        	|     0.32                         	|
|     8                                  	|     2                                        	|     0.33                         	|
|     4                                  	|     3                                        	|     0.42                         	|
|     8                                  	|     3                                        	|     0.40                         	|
|     4                                  	|     4                                        	|     0.43                         	|
|     8                                  	|     4                                        	|     0.42                         	|
|     16                                 	|     4                                        	|     0.42                         	|
```
