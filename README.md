# Conditional_GAN

## Introduce
Conditional GAN is a big breakthrough in the development of GAN. Just like its name "Conditional", cGAN can generate image under some certain condition, and only need to have some small modify on the input of both generator and discriminator. 

For example, if I want to generate the images of car and bus, I can give the model the label that represent to car and bus, and it  will generate the images I request according to the conditions.

Furthermore, Conditional GAN can not only generate images, because the main idea of cgan is to control the model's output, similar structure can be implement to sequence generation 、 voice generation and even generate a video.

## Network Structure

![image](https://github.com/Yukino1010/Conditional_GAN/blob/master/resNet.png)

In this implementation, I replace CNN with residual structure (the left part of the picture) in order to build a deeper network, <br>
and using WGAN-GP to make the training more stable.

## Hyperparameters

- BATCH_SIZE = 20
- NOISE_DIM = 100
- LAMBDA = 10 
- LABEL_NUM = 30
- DROPOUT_RATE = 0.2

All the layers use "leaky_relu" as activative funtion<br>

Normalization:<br>

Generator ~ BatchNormalization<br>

Discriminator ~ LayerNormalization

## Result

<p align="center"><img width="400px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final2.png">
<img width="400px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final3.png" /></p>

<p align="center">
<img width="400px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final6.png?raw=true" >
<img width="400px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final7.png?raw=true">
</p>

<p align="center">
<i>yellow hair red eyes</i>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<i>yellow hair green eyes</i>
</p>

<p align="center">
<img width="400px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final8.png?raw=true" >
<img width="400px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final9.png?raw=true">
</p>

<p align="center">
<i>blue hair red eyes</i>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<i>final result</i>
</p>


## References

1. ***resNet*** [[arxiv](https://arxiv.org/pdf/1512.03385.pdf)]
2. ***Generative Adversarial Network based on Resnet*** [[arxiv](https://arxiv.org/abs/1707.04881)]
3. ***cWGANs*** [https://github.com/cameronfabbri/cWGANs]

