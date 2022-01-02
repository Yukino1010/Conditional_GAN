# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:44:17 2021

@author: s1253
"""

import os
import time
import pathlib
import pickle
import tensorflow as tf
import numpy as np
from Model import Generator, Discriminator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


TARGET_IMG_SIZE = 64 

BATCH_SIZE = 20
NOISE_DIM = 100
LAMBDA = 10 

EPOCHs = 100
CURRENT_EPOCH = 1 
SAVE_EVERY_N_EPOCH = 1
N_CRITIC = 5 


'''  load data and data augumentation  '''

data_path = pathlib.Path('images')

file_list = [str(path) for path in data_path.glob('*.jpg')]
file_list = file_list[:7785]

d = len(file_list)

import cv2

data=np.empty((d,64,64,3))

for i, img in enumerate(file_list):
    img = cv2.imread(img) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    
    img = (img - 127.5) / 127.5
    data[i]=img   
    

image_generator = ImageDataGenerator(
    
    featurewise_std_normalization=True,
    featurewise_center=True,
    rotation_range = 5,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    horizontal_flip=True,
    zoom_range= 0.1,
    fill_mode='nearest'
    
    )

label = pickle.load(open("data_tag.pkl", "rb"))

train_data = image_generator.flow(data, label, batch_size=BATCH_SIZE)


#%%

# load model

MODEL_NAME = 'CGAN'
OUTPUT_PATH = 'outputs'


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)



generator = Generator(input_dim=NOISE_DIM,
                      label_len = 30,
                      generator_shape=(4, 4, 512),
                      batch_norm=True,
                      dropout=0.2,
                      generator_upsample=[2,2,1,2,2,1],
                      generator_conv_filters=[256,128,128,128,64,3],
                      generator_conv_kernal=[4,4,4,4,4,4],
                      generator_short_cut=[1,1,0,0,1,1]
                      ).build_layer()



plot_model(generator, "generator.png", show_shapes=(True))

discriminator = Discriminator(discriminator_input=(TARGET_IMG_SIZE, TARGET_IMG_SIZE, 3),
                         label_len = 30,  # have 30 different types of image
                         norm="layer_norm",
                         activation="leaky_relu",
                         discriminator_upsampling=[1,2,2,2,2],
                         discriminator_conv_filters=[64,128,128,256,1],
                         discriminator_conv_kernal=[5,5,5,4,4],
                         discriminator_short_cut=[0,1,1,1,0]
                         ).build_discriminator()

    

plot_model(discriminator, "discriminator.png", show_shapes=(True))

#generator.summary()
#discriminator.summary()




checkpoint_path = os.path.join("checkpoints", MODEL_NAME)

G_optimizer = tf.keras.optimizers.Adam(0.0001,beta_1=0, beta_2=0.9)
D_optimizer = tf.keras.optimizers.Adam(0.0001,beta_1=0, beta_2=0.9)

ckpt = tf.train.Checkpoint(models=[generator, discriminator])
                           

# save model

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    latest_epoch = int(ckpt_manager.latest_checkpoint.split('-')[1])
    
    CURRENT_EPOCH = latest_epoch * SAVE_EVERY_N_EPOCH
    print ('Latest checkpoint of epoch {} restored!!'.format(CURRENT_EPOCH))
    
#%%

# Wasserstein loss

def WGAN_GP_train_d_step(real_image, label, batch_size):

    noise = tf.random.normal([batch_size, NOISE_DIM])
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)

    with tf.GradientTape(persistent=True) as d_tape:
        
        with tf.GradientTape() as gp_tape:
            
            fake_image = generator([noise, label], training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred = discriminator([fake_image_mixed, label], training=True)
            
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
        
        fake_pred = discriminator([fake_image, label], training=True)
        real_pred = discriminator([real_image, label], training=True)
        
        D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
    
    D_gradients = d_tape.gradient(D_loss,
                                            discriminator.trainable_variables)
    
    D_optimizer.apply_gradients(zip(D_gradients,
                                                discriminator.trainable_variables))
    




def WGAN_GP_train_g_step(real_image, label, batch_size):
   
    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as g_tape:
        
        fake_image = generator([noise, label], training=True)
        fake_pred= discriminator([fake_image, label], training=True)
        
        G_loss = -tf.reduce_mean(fake_pred) 
        
    G_gradients = g_tape.gradient(G_loss,
                                            generator.trainable_variables)
   
    G_optimizer.apply_gradients(zip(G_gradients,
                                                generator.trainable_variables))


#%%

n_critic_count = 0

num_examples_to_generate = 18
sample_noise = np.random.normal(size=[num_examples_to_generate, NOISE_DIM])     

label_test = [5]*6 + [28]*6 + [17]*6
label_test = np.reshape(label_test, [18,1])



import matplotlib.pyplot as plt


def generate_and_save_images(model, epoch, test_input, OUTPUT_PATH,figure_size=(12,6), subplot=(3,6), save=True):
    predictions = model.predict(test_input)
    
    for i in range(predictions.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        axs.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, 'image_at_epoch_{:04d}.png'.format(epoch)))   
    plt.show()


'''
# Training

for epoch in range(CURRENT_EPOCH, EPOCHs+1):
    start = time.time()
    print('Start of epoch %d' % (epoch,))
 
    
    for step, (image, label) in enumerate(train_data):
        current_batch_size = image.shape[0]
        
        # Train critic (discriminator)
        
        WGAN_GP_train_d_step(image, label, batch_size=current_batch_size)
        n_critic_count += 1
        
        if n_critic_count >= N_CRITIC: 
            
            # Train generator
            WGAN_GP_train_g_step(image, label, batch_size=current_batch_size)
            n_critic_count = 0
        
        if step % 50 == 0:
            print ('.', end='')
        
        if step > 1000:  # generate 1000 batch
            break
    
    
    
    if epoch % SAVE_EVERY_N_EPOCH == 0:
        
        #save model
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch,
                                                             ckpt_save_path))
    
    
    print ('Time taken for epoch {} is {} sec\n'.format(epoch,
                                                      time.time()-start))
    
    generate_and_save_images(generator, epoch, [sample_noise, label_test], OUTPUT_PATH, figure_size=(12,6), subplot=(3,6), save=True)

'''


