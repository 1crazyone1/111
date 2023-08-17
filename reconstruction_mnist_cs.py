import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils
import os
import datetime
import keras
import time
from scipy.io import savemat
import pathlib
import cv2
from PIL import Image
from keras import backend, layers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, add
from keras.layers import Dense,Flatten,UpSampling2D
from tensorflow.keras.optimizers import Adam,Nadam, SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
import argparse
import math

from mnist_cs import generator as generator_wgan
# from Change_WGAN import checkpoint as checkpoint_wgan
# from Change_GAN import generator as generator_gan
# from mnist_cs import checkpoint as checkpoint_gan
from mnist_cs import generate_plot_image,cal_signal,datasets


OUTPUT_CHANNELS = 1
BATCH_SIZE = 16
num_exp_to_generate = 16
EPOCH = 10
noise_dim = 100
CS_ratio = 98
#CS_ratio = 50
#CS_ratio = 25
image_size = 28
BUFFER_SIZE = 6000

np.random.seed(2)
randn_B = np.random.randn(CS_ratio,784)
mask = np.random.randn(CS_ratio,784)
mask = mask.astype(np.float32)
mask_1 = np.array(mask).reshape(CS_ratio,28,28)
mask_2=mask_1.transpose(1,2,0)
m_matrix_tf = np.zeros((BATCH_SIZE,28,28,CS_ratio))
for i in range(BATCH_SIZE):
    m_matrix_tf[i,:,:,:]=mask_2
m_matrix_tf = m_matrix_tf.astype(np.float32)
m_matrix_tf = tf.convert_to_tensor(m_matrix_tf)

def load_lung(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    # image = image[:,:,:1]
    return image

def preprocess(x):
  #预处理函数
  x = tf.cast(x,dtype=tf.float32)/127.5 - 1
  return x

#定义下采样卷积模块
def downsample(filters, size, apply_batchnorm=True):
  #initializer = tf.random_normal_initializer(0., 0.02)
  initializer = tf.keras.initializers.VarianceScaling(scale=0.02, mode='fan_in', distribution='uniform', seed=None)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  #result.add(tf.keras.layers.LeakyReLU())
  result.add(tf.keras.layers.ReLU())
  return result

#定义上采样卷积模块
def upsample(filters, size, apply_dropout=False):
  #initializer = tf.random_normal_initializer(0., 0.02)
  initializer = tf.keras.initializers.VarianceScaling(scale=0.02, mode='fan_in', distribution='uniform', seed=None)
  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  #result.add(tf.keras.layers.LeakyReLU())
  return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[CS_ratio, 1])  # length
    down_stack = [
        downsample(2, 4, apply_batchnorm=False),  # (bs, 32, 32, 64) #yao gai
        downsample(4, 4),  # (bs, 16, 16, 32)
        downsample(8, 4),  # (bs, 8, 8, 64)
        downsample(16, 4),  # (bs, 4, 4, 128)
        downsample(32, 4),  # (bs, 2, 2, 128)
        #downsample(32, 4),  # (bs, 1, 1, 128)
    ]
    up_stack = [
        upsample(32, 4, apply_dropout=False),  # (bs, 2, 2, 256)
        upsample(32, 4, apply_dropout=False),  # (bs, 4, 4, 256)
        upsample(16, 4),  # (bs, 8, 8, 128)
        upsample(8, 4),  # (bs, 16, 16, 64)
        upsample(4, 4),  # (bs, 32, 32, 32) #yao gai
    ]
    # initializer = tf.random_normal_initializer(0., 0.02)
    initializer = tf.keras.initializers.VarianceScaling(scale=0.02, mode='fan_in', distribution='uniform', seed=None)
    dense1 = tf.keras.layers.Dense(1024, activation='relu')#1024
    reshape1 = tf.keras.layers.Reshape((CS_ratio,)) # length
    reshape2 = tf.keras.layers.Reshape((32, 32, 1)) #(32,32,1)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer)
    x = inputs
    x = reshape1(x)
    x = dense1(x)
    x = reshape2(x)
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # print(x)
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    new_last = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 5, strides=1, activation='relu')  # FL用relu yao gai =>28,28
    x = new_last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

generator_unet = Generator()
generator_unet.summary()

def generator_unet_loss(image_gan, image_unets,true_signal, false_signal):
    image_loss = tf.reduce_mean(tf.abs(image_unets - image_gan))
    signal_loss = tf.reduce_mean(tf.abs(true_signal - false_signal))
    gen_total_loss = image_loss
    return gen_total_loss, image_loss, signal_loss

generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)#5e-5
checkpoint_dir = './training_checkpoints_reconstruction_mnist_cs5_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator_unet)

log_dir="logs_reconstruction_mnist_cs5_1/"
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit_reconstruction_mnist_cs5_1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def compute_signal(images,image_gan):
    true_signal = np.zeros((BATCH_SIZE, CS_ratio, 1))
    false_signal = np.zeros((BATCH_SIZE, CS_ratio, 1))
    for i in range(BATCH_SIZE):
        # print(images[i].shape)
        real_image = np.array(images[i]).reshape(784, 1)
        fake_image = np.array(image_gan[i]).reshape(784, 1)
        real_signal = np.dot(randn_B, real_image)
        fake_signal = np.dot(randn_B, fake_image)
        true_signal[i, :, :] = real_signal  # 真实信号
        false_signal[i, :, :] = fake_signal  # 虚假信号
    max1 = tf.reduce_max(true_signal)
    min1 = tf.reduce_min(true_signal)
    max2 = tf.reduce_max(false_signal)
    min2 = tf.reduce_min(false_signal)
    # print(min1,max1)
    # print(min2,max2)
    true_signal = true_signal / 100
    false_signal = false_signal / 100
    true_signal = true_signal.astype(np.float32)
    false_signal = false_signal.astype(np.float32)
    # true_signal = images * m_matrix_tf
    # false_signal = image_gan * m_matrix_tf
    #
    # true_signal = tf.reduce_mean(true_signal, axis=1)
    # true_signal = tf.reduce_mean(true_signal, axis=1)
    #
    # false_signal = tf.reduce_mean(false_signal, axis=1)
    # false_signal = tf.reduce_mean(false_signal, axis=1)
    return true_signal,false_signal


def start_train(images,epoch):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape:
        image_gan = generator_wgan(noise, training=True)
        true_signal,false_signal = compute_signal(images,image_gan)

        image_unets = generator_unet(false_signal,training=True)
        gen_total_loss,image_loss,signal_loss = generator_unet_loss(image_gan,image_unets,true_signal,false_signal)
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator_unet.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator_unet.trainable_variables))
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('image_loss', image_loss, step=epoch)
        tf.summary.scalar('signal_loss', signal_loss, step=epoch)

def cal_psnr_ssim(pre_images_gan,pre_images_unets):
    img1 = pre_images_gan[0]
    img2 = pre_images_unets[0]
    # print(img1.shape,img2.shape)
    # img1 = np.array(img1).reshape(28, 28, 1)
    # img2 = np.array(img2).reshape(28, 28, 1)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)
    psnr = tf.image.psnr(img1, img2, max_val=1.0)
    ssim = tf.image.ssim(img1, img2, max_val=1.0)
    return psnr,ssim

def generate_image(generator_wgan,test_noise,generator_unet,epoch):
    pre_images_gan = generator_wgan(test_noise, training=False)
    pre_signal,pre_signal_1 = compute_signal(pre_images_gan,pre_images_gan)
    pre_images_unets = generator_unet(pre_signal)

    # psnr,ssim = cal_psnr_ssim(pre_images_gan,pre_images_unets)
    # print("psnr",psnr)
    # print("ssim",ssim)
    fig = plt.figure(figsize=(4, 4))
    for i in range(pre_images_unets.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((pre_images_unets[i, :, :, 0] + 1) / 2, cmap='gray')
        plt.axis('off')
        plt.savefig("./reconstruction_mnist_cs5_1/"+str(epoch)+"_" +"_unets"+ ".png")
    #plt.show()

list_psnr = []
list_ssim = []
def generate_image_test(image_signal,generator_unet,images,epoch):
    # print("dataset:", image_signal.shape)
    pre_images_unets = generator_unet(image_signal)

    fig = plt.figure(figsize=(4, 4))
    display_list = [images[0], pre_images_unets[0]]
    img1 = display_list[0]
    img2 = display_list[1]
    # print(img1.shape)
    img1 = np.array(img1).reshape(28, 28, 1)
    img2 = np.array(img2).reshape(28, 28, 1)
    # savemat(r"C:\Users\程龙\PycharmProjects\pythonProject\GAN_CS\matrix1\img1_t_%d.mat" %(epoch), {"Array_img1":img1})
    # savemat(r"C:\Users\程龙\PycharmProjects\pythonProject\GAN_CS\matrix2\img2_t_%d.mat "%(epoch), {"Array_img2":img2})
    psnr, ssim = cal_psnr_ssim(images,pre_images_unets)
    print(psnr,ssim)
    list_psnr.append(psnr)
    list_ssim.append(ssim)
    title = ['Ground Truth', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.savefig("./reconstruction_mnist_cs_test2_1/" + str(epoch) + "_" + ".png")

def fit(dataset,epochs):
    seed = tf.random.normal([num_exp_to_generate, noise_dim])  # 随机正态分布
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)
        for images in dataset:
            start_train(images,epoch)
            print('.', end='')
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
        generate_image(generator_wgan, seed, generator_unet,epoch)
    checkpoint.save(file_prefix=checkpoint_prefix)


dir_root_test = pathlib.Path('D:/Gan_Cifa/compsensing_dip-master/data/mnist/sub')
dir_path_test = list(dir_root_test.glob('*.jpg'))
dir_path_test.sort()

test_num = 0
x_test = np.zeros([100,image_size,image_size,1])
for img_path in dir_path_test:
    img_path = os.path.join(img_path)
    img = load_lung(img_path)
    x_test[test_num] = img
    test_num += 1

val_set = tf.data.Dataset.from_tensor_slices((x_test))
val_set = val_set.map(preprocess).batch(100)
print(len(val_set))
print(val_set)

def test_data(image_signal, generator_unet, images, epoch):
    for i in range(images.shape[0]):
        # print("val_set:",image_signal.shape)
        pre_image = generator_unet(image_signal)
        # print("images",images.shape)
        # print("pre_image", pre_image.shape)
        display_list = [images[i], pre_image[i]]
        img1 = display_list[0]
        img2 = display_list[1]
        img1 = np.array(img1).reshape(image_size, image_size, 1)
        img2 = np.array(img2).reshape(image_size, image_size, 1)
        new_path_1 = "D:/data/cell_data/cv2_mnist/mnist_98/" + str(epoch) + "_1" + ".jpg"
        new_path_2 = "D:/data/cell_data/cv2_mnist/mnist_98/" + str(epoch) + "_0" + ".jpg"
        cv2.imwrite(new_path_1, img1*255.0)
        cv2.imwrite(new_path_2, img2*255.0)
        #savemat(r"C:\Users\程龙\PycharmProjects\pythonProject\GAN_CS\matrix1\img1_t_%d.mat" % (epoch),{"Array_img1": img1})
        #savemat(r"C:\Users\程龙\PycharmProjects\pythonProject\GAN_CS\matrix2\img2_t_%d.mat " % (epoch),{"Array_img2": img2})
        img1 = tf.image.convert_image_dtype(img1, tf.float32)
        img2 = tf.image.convert_image_dtype(img2, tf.float32)
        psnr = tf.image.psnr(img1, img2, max_val=1.0)
        ssim = tf.image.ssim(img1, img2, max_val=1.0)
        print(epoch, psnr, ssim)
        title = ['Ground Truth', 'Predicted Image']
        # for i in range(2):
        #     plt.subplot(1, 2, i + 1)
        #     plt.title(title[i])
        #     plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
        #     plt.axis('off')
        #     if i == 1:
        #         plt.savefig("./reconstruction_mnist_cs_test5_1/" + str(epoch) + "_" + ".png")
        epoch = epoch + 1




checkpoint.restore(r"D:\cs_train_weights\training_checkpoints_reconstruction_mnist_cs3\ckpt-3") #cs_ratio = 98
# checkpoint.restore(r"C:\Users\程龙\PycharmProjects\pythonProject\GAN_CS\training_checkpoints_reconstruction_mnist_cs2_1\ckpt-2")#cs_ratio = 50
#checkpoint.restore(r"C:\Users\程龙\PycharmProjects\pythonProject\GAN_CS\training_checkpoints_reconstruction_mnist_cs5_1\ckpt-2")#cs_ratio = 25

# fit(datasets,EPOCH)


# datasets
# num = 0
# for images in datasets.take(100):
#     # print(images.shape)
#     images_signal,pre = compute_signal(images,images)
#     generate_image_test(images_signal,generator_unet,images,num)
#     # test_data(images_signal,generator_unet,images,num)
#     num = num + 1

def compute_val_signal(images):
    test_num = images.shape[0]
    true_signal = np.zeros((test_num, CS_ratio, 1))
    for i in range(test_num):
        # print(images[i].shape)
        real_image = np.array(images[i]).reshape(784, 1)
        real_signal = np.dot(randn_B, real_image)
        true_signal[i, :, :] = real_signal  # 真实信号
    true_signal = true_signal / 100
    true_signal = true_signal.astype(np.float32)
    return true_signal

# val_set
val_num = 0
for images in val_set.take(1):
    # print(image.shape)
    image_signal = compute_val_signal(images)
    test_data(image_signal, generator_unet,images,val_num)
    val_num = val_num + 1




sum1 = 0
sum2 = 0
np_psnr = np.array(list_psnr)
np_ssim = np.array(list_ssim)
for i in range(len(np_psnr)):
    sum1 = sum1 +np_psnr[i]

for j in range(len(np_ssim)):
    sum2 = sum2 +np_ssim[j]

sum1 = sum1/100
sum2 = sum2/100
print("average psnr is {}\n".format(sum1))
print("average ssim is {}\n".format(sum2))


