# ImageNet-CUDA
AlexNet/ImageNet is a deep convolutional neural network (CNN) for image classification on the ImageNet dataset, achieving state-of-the-art performance at the time of publication. The network architecture features multiple convolutional and fully connected layers to let use techniques such as data augmentation and dropout regularization to improve generalization for classification tasks. This is specifically programmed in CUDA to analyze parallel workloads with various optimizations.

![alexnet](https://github.com/AdithyaDV/AlexNet-CUDA/assets/44144424/982cfb42-9f15-4fb5-ac14-dd4c5aab30da)
![anet-table](https://github.com/AdithyaDV/AlexNet-CUDA/assets/44144424/adece48c-e459-4b75-8b73-f3cf632796cd)

## Implementation 
Has the below beautiful abstract visualisations that compares both CPU and GPU artifacts
![CPU](https://github.com/AdithyaDV/AlexNet-CUDA/assets/44144424/1e1efd49-250f-45f8-af41-c8c18d7b0eb1)
![GPU](https://github.com/AdithyaDV/AlexNet-CUDA/assets/44144424/a88c3584-e708-4767-b71d-8094cc74847e)

## Want to know the implementation?
Analysis.pdf is all yours. You can observe the explanation of the implementation. Furthermore head to src/cudaLib.cu for implementation of Convolution, Pooling and Softmax layers. You can run the application by 
1) Having a newer version of cuda installed.

2) cd ~/AlexNet-CUDA

3) cmake .

4) make

5) ./run

## References
1) Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2012. ImageNet classification with deep convolutional neural networks. In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1 (NIPS'12). Curran Associates Inc., Red Hook, NY, USA, 1097–1105.

2) https://docs.nvidia.com/cuda/cuda-c-programming-guide/
