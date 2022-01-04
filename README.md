# Conditional_GAN

## Introduce
Conditional GAN is a big breakthrough in the development of GAN. Just like its name "Conditional", cGAN can generate image under some certain condition, and only need to have some small modify on the input of both generator and discriminator.  For example, if I want to generate the images of car and bus, I can give the model the label that represent to car and bus, and it  will generate the images I request according to the conditions.

Furthermore, Conditional GAN can not only generate images, because the main idea of cgan is to control the model's output, similar structure can be implement to sequence generation 、 voice generation and even generate a video.
## Network Structure


## Hyperparameters

## Data 

## Result

<p align="center"><img width="450px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final2.png">
<img width="425px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final3.png" /></p>

<p align="center">
<img width="425px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final6.png?raw=true" >
<img width="425px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final7.png?raw=true">
</p>

<p align="center">
<i>yellow hair red eyes</i>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<i>yellow hair green eyes</i>
</p>

<p align="center">
<img width="425px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final8.png?raw=true" >
<img width="425px" src="https://github.com/Yukino1010/Conditional_GAN/blob/master/outputs/final9.png?raw=true">
</p>

<p align="center">
<i>blue hair red eyes</i>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<i>final result</i>
</p>


## References

1. ***cWGANs*** [https://github.com/cameronfabbri/cWGANs]
2. ***Generative Adversarial Network based on Resnet for Conditional Image Restoration*** [[arxiv](https://arxiv.org/abs/1707.04881)]
