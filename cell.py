import pathlib
import keras
import matplotlib
from keras import layers
import cv2
# from scipy.signal import normalize
from numpy.f2py.symbolic import normalize
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils
import os
import datetime
import keras
import time
from skimage import filters
from PIL import Image
from keras import backend, layers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, add
from keras.layers import Dense,Flatten,UpSampling2D
from tensorflow.keras.optimizers import Adam,Nadam, SGD,RMSprop
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
import argparse
import math
# matplotlib.use('Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# image_size = 256
new_image_size = 55

epochs = 200#训练回合数
BUFFER_SIZE = 6000
BATCH_SIZE = 16
# image_size = 128
# image_size = 256

CS_RATIO = 189
# CS_RATIO = 378

g_lr = 0.0001
d_lr = 0.001
is_training = True#是否训练的标志
z_dim = 100#采样向量的长度


path = "D:/data/cell_data/cell/"


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

dir_root = pathlib.Path('D:/data/cell_data/cell')
dir_path_cell = list(dir_root.glob('*.jpg'))
dir_path_cell.sort()

cell_num = 0
x_cell = np.zeros([33296,new_image_size,new_image_size,1])

for img_path in dir_path_cell:
    img_path = os.path.join(img_path)
    img = load_lung(img_path)
    x_cell[cell_num] = img
    cell_num += 1

print(cell_num)
dataset = tf.data.Dataset.from_tensor_slices((x_cell))
dataset = dataset.shuffle(BUFFER_SIZE).map(preprocess).batch(BATCH_SIZE)
print(len(dataset))
print(dataset)

# dir_root_test = pathlib.Path('D:/data/cell_data/test_data')
# dir_path_test = list(dir_root_test.glob('*.jpg'))
# dir_path_test.sort()
# test_num = 0
# x_test = np.zeros([16,new_image_size,new_image_size,1])
# for img_path in dir_path_test:
#     img_path = os.path.join(img_path)
#     img = load_lung(img_path)
#     x_test[test_num] = img
#     test_num += 1
#
# val_set = tf.data.Dataset.from_tensor_slices((x_test))
# val_set = val_set.shuffle(BUFFER_SIZE).map(preprocess).batch(BATCH_SIZE)
# print(len(val_set))
# print(val_set)





#prepare CS
np.random.seed(2)
mask = np.random.randn(CS_RATIO,new_image_size * new_image_size)
mask = mask.astype(np.float32)
mask_1 = np.array(mask).reshape(CS_RATIO,new_image_size,new_image_size)
mask_2=mask_1.transpose(1,2,0)
m_matrix_tf = np.zeros((BATCH_SIZE,new_image_size,new_image_size,CS_RATIO))
for i in range(BATCH_SIZE):
    m_matrix_tf[i,:,:,:]=mask_2
m_matrix_tf = m_matrix_tf.astype(np.float32)
m_matrix_tf = tf.convert_to_tensor(m_matrix_tf)


def generator_model():
    inputs = tf.keras.layers.Input(shape=(z_dim,))
    x = keras.layers.Dense(units=new_image_size * new_image_size, activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.05))(inputs)
    x = tf.reshape(x, (-1, new_image_size, new_image_size, 1))
    x = keras.layers.Conv2D(64, 11, padding='same', activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)
    x = keras.layers.Conv2D(32, 1, padding='same', activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)
    x = keras.layers.Conv2D(16, 5, padding='same', activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)
    # x = keras.layers.Conv2D(8, 5, padding='same', activation=tf.nn.leaky_relu,
    #                          kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)
    x = keras.layers.Conv2D(1, 7, padding='same', activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)
    x = keras.layers.Conv2D(64, 11, padding='same', activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)
    x = keras.layers.Conv2D(32, 1, padding='same', activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)
    x = keras.layers.Conv2D(16, 5, padding='same', activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)
    # x = keras.layers.Conv2D(8, 5, padding='same', activation=tf.nn.leaky_relu,
    #                          kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)
    x = keras.layers.Conv2D(1, 7, padding='same', activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.1))(x)

    return keras.Model(inputs=inputs, outputs=x)

def discriminator_model():
    # 下面搭建判别器架构，同样采用序贯模型
    inputs = tf.keras.layers.Input(shape=[CS_RATIO, 1])  # length
    reshape1 = tf.keras.layers.Reshape((CS_RATIO,))
    dense1 = tf.keras.layers.Dense(new_image_size * new_image_size, activation='tanh')
    reshape2 = tf.keras.layers.Reshape((new_image_size, new_image_size, 1))

    x = inputs
    x = reshape1(x)
    x = dense1(x)
    x = reshape2(x)
    # # normal
    x = keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(new_image_size, new_image_size ,1))(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    # # downsample
    x = keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    # # downsample
    x = keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    # # downsample
    x = keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    # # classifier
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

generator = generator_model()
generator.summary()
discriminator = discriminator_model()
discriminator.summary()

def gradient_penalty(discriminator, batch_x, fake_image):
  #梯度惩罚项计算函数
  batchsz = batch_x.shape[0]
  #每个样本均随机采样t,用于插值
  t = tf.random.uniform([batchsz,1])
  #自动扩展为x的形状:[b,1,1,1] => [b,h,w,c]
  t = tf.broadcast_to(t, batch_x.shape)
  #在真假图片之间作线性插值
  interplate = t * batch_x + (1-t) * fake_image
  #在梯度环境中计算 D 对插值样本的梯度
  with tf.GradientTape() as tape:
    tape.watch([interplate])
    d_interplate_logits = discriminator(interplate)
  grads = tape.gradient(d_interplate_logits, interplate)
  #计算每个样本的梯度的范数:[b,h,w,c] => [b,-1]
  grads = tf.reshape(grads, [grads.shape[0],-1])
  gp = tf.norm(grads,axis=1)
  #计算梯度惩罚项
  gp = tf.reduce_mean((gp-1.)**2)
  return gp

def cal_signal(images, gen_image):
    # print(images.shape)
    # print(m_matrix_tf.shape)
    true_signal = images * m_matrix_tf
    false_signal = gen_image * m_matrix_tf

    true_signal = tf.reduce_mean(true_signal, axis=1)
    true_signal = tf.reduce_mean(true_signal, axis=1)# (64,256)

    false_signal = tf.reduce_mean(false_signal, axis=1)
    false_signal = tf.reduce_mean(false_signal, axis=1)
    # tf.Tensor([], shape = (batch_size, image_size, image_size, CS_ratio), dtype = float32))
    # print(true_signal,false_signal)
    # print(images.shape,gen_image.shape)          #(64, 32, 32, 1)
    # print(m_matrix_tf.shape)                     #(64, 32, 32, 256)
    # print(true_signal.shape,false_signal.shape)  #(64, 32, 32, 256)
    return true_signal, false_signal

def generator_loss(fake_out,gen_image,images):
    true_signal,false_signal = cal_signal(images,gen_image)
    l1_loss = tf.reduce_mean(tf.abs(true_signal - false_signal))
    fake_loss = cross_entropy(tf.ones_like(fake_out), fake_out) # 希望假的判别为1
    bg_loss = tf.reduce_mean(tf.abs(gen_image))
    loss = - tf.reduce_mean(fake_out) # WGAN-GP G 损失函数，最大化假样本的输出值1
    gen_total_loss = fake_loss

    return gen_total_loss, bg_loss, l1_loss, loss

def discriminator_loss(real_out, fake_out, gp_loss):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out) # 希望真的判别为1
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out) # 希望假的判别为0
    # disc_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out) + 10.* gp_loss
    disc_loss = real_loss + fake_loss
    return disc_loss

# Define the Optimizers and Checkpoint-saver
generator_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=0.5)
checkpoint_dir = './training_checkpoints_cell4'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

log_dir="logs_cell4/"
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit_cell4/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def train_step(images,epoch):
    noise = tf.random.normal([BATCH_SIZE, z_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_image = generator(noise, training = True)

        true_signal, false_signal = cal_signal(images, gen_image)

        fake_out = discriminator(false_signal, training = True)
        #fake_out = discriminator(gen_image, training = True)
        real_out = discriminator(true_signal, training=True)

        gen_total_loss, bg_loss, l1_loss, loss = generator_loss(fake_out,gen_image,images)
        gp_loss = gradient_penalty(discriminator,true_signal,false_signal)
        disc_loss = discriminator_loss(real_out, fake_out, gp_loss)


    gradient_gen = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
        tf.summary.scalar('bg_loss', bg_loss, step=epoch)
        tf.summary.scalar('l1_loss', l1_loss, step=epoch)
        tf.summary.scalar('gp_loss', gp_loss, step=epoch)
        tf.summary.scalar('loss', loss, step=epoch)

def generate_plot_image(gen_model, test_noise,epoch):
    pre_images = gen_model(test_noise, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(BATCH_SIZE):
        plt.subplot(4, 4, i+1)
        plt.imshow((pre_images[i] + 1) / 2, cmap='gray')
        plt.axis('off')
        plt.savefig("./cell4/"+str(epoch)+"_" + ".png")
    #plt.show()

def generate_test_image(images,gen_model, test_noise,epoch):
    pre_images = gen_model(test_noise, training=False)
    # print(pre_images.shape)

    plt.imshow(pre_images[0] * 0.5 + 0.5, cmap='gray')
    # plt.imshow(pre_images[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.savefig("./cell_test4/" + str(epoch) + "_" + ".png")
    # new_image = np.array(pre_images[0])
    # new_path = "./cell_test1/" + str(epoch) + "_" + ".png"
    # cv2.imwrite(new_path, new_image*255.0)


def train(dataset, epochs):
    seed = tf.random.normal([BATCH_SIZE, z_dim])  # 随机正态分布
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)
        for image_batch in dataset:
            # 优化
            train_step(image_batch,epoch)
            print('.', end='')
        if (epoch + 1) % 100 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
        generate_plot_image(generator, seed, epoch)
        # if epoch == 0:
        #     discriminator.summary()
    checkpoint.save(file_prefix=checkpoint_prefix)

# train(dataset,epochs)
# checkpoint.restore(r"C:\Users\程龙\PycharmProjects\pythonProject\image_cell\training_checkpoints_cell1\ckpt-6")
# checkpoint.restore(r"C:\Users\程龙\PycharmProjects\pythonProject\image_cell\training_checkpoints_cell2\ckpt-2")
# checkpoint.restore(r"C:\Users\程龙\PycharmProjects\pythonProject\image_cell\training_checkpoints_cell3\ckpt-3") # cs_ratio = 378
# checkpoint.restore(r"C:\Users\程龙\PycharmProjects\pythonProject\image_cell\training_checkpoints_cell4\ckpt-3") # cs_ratio = 189

# num = 0
# for images in dataset.take(100):
#     seed = tf.random.normal([BATCH_SIZE, z_dim])  # 随机正态分布
#     # generate_plot_image(generator,seed,num)
#     generate_test_image(images,generator, seed, num)
#     num = num + 1

# for images in dataset.take(2):
#     for i in range(1):
#         plt.subplot(1, 1, i+1)
#         plt.imshow((images[i] + 1) / 2,cmap='gray')
#         plt.axis('off')
#     plt.show()


# DCGAN(深度卷积生成对抗网络)是一种流行的用于生成逼真图像的神经网络架构。以下是提高DCGAN图像质量的一些技巧:
#
# 增加生成器和鉴别器网络的深度:更深的网络可以学习更复杂的表示并生成更高质量的图像。
#
# 使用批量归一化:批量归一化可以帮助稳定训练过程，提高生成图像的质量。
#
# 使用更好的损失函数:损失函数的选择对于生成图像的质量至关重要。您可以尝试不同的损失函数，如Wasserstein损失、铰链损失或特征匹配损失。
#
# 使用高质量的训练数据:生成的图像的质量在很大程度上取决于训练数据的质量。使用高分辨率的高质量图像可以获得更好的结果。
#
# 增加训练时间:训练dcgan可能需要很长时间，训练模型的时间越长，图像质量就越好。您还可以使用渐进式增长等技术来提高生成图像的质量。
#
# 使用数据增强:旋转、缩放和裁剪等数据增强技术可以帮助模型学习生成更多样化的图像。
#
# 微调超参数:微调超参数，如学习率、批大小和训练周期数，可以对图像质量产生重大影响。
#
# 尝试不同的架构:DCGAN只是生成图像的一种架构。您可以尝试不同的架构，如StyleGAN或BigGAN，以查看它们是否为您的特定用例生成了质量更好的图像。


