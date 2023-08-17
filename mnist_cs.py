import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils
import os
import datetime
import keras
import time
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

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

(train_images, train_labels),(_, _) = tf.keras.datasets.mnist.load_data() # _表示占位符
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') # 使其变为四维
# 归一化处理，使其在（-1， 1）之间
train_images = (train_images - 127.5) / 127.5
# 一些参数
BATCH_SIZE = 16
BUFFER_SIZE = 6000

#CS_ratio = 50
#CS_ratio = 98
CS_ratio = 25

image_size = 28
EPOCHS = 300
noise_dim = 100
depth = 4
width = 128
leaky_relu_slope = 0.2
dropout_rate = 0.4
num_exp_to_generate = 16
learning_rate = 2e-4
# noise_size = 64
OUTPUT_CHANNELS = 1

# 创建数据集
datasets = tf.data.Dataset.from_tensor_slices(train_images)
# 全部范围乱序取出BATCH_SIZE数据
datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def generator_model():
    #下面搭建生成器的架构，首先导入序贯模型（sequential），即多个网络层的线性堆叠
    model = Sequential()
    #添加一个全连接层，输入为100维向量，输出为1024维
    model.add(Dense(input_dim=100, units=1024))
    #添加一个激活函数tanh
    model.add(Activation('tanh'))
    #添加一个全连接层，输出为128×7×7维度
    model.add(Dense(128*7*7))
    #添加一个批量归一化层，该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #Reshape层用来将输入shape转换为特定的shape，将含有128*7*7个元素的向量转化为7×7×128张量
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    #2维上采样层，即将数据的行和列分别重复2次
    model.add(UpSampling2D(size=(2, 2)))
    #添加一个2维卷积层，卷积核大小为5×5，激活函数为tanh，共64个卷积核，并采用padding以保持图像尺寸不变
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    #卷积核设为1即输出图像的维度
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model

# def generator_model():
#     noise_input = keras.Input(shape=(noise_dim,))
#     x = layers.Dense(4 * 4 * width, use_bias=False)(noise_input)
#     x = layers.BatchNormalization(scale=False)(x)
#     x = layers.ReLU()(x)
#     x = layers.Reshape(target_shape=(4, 4, width))(x)
#     for _ in range(depth - 2):
#         x = layers.Conv2DTranspose(
#             width,
#             kernel_size=4,
#             strides=2,
#             padding="same",
#             use_bias=False,
#         )(x)
#         x = layers.BatchNormalization(scale=False)(x)
#         x = layers.ReLU()(x)
#     image_output = layers.Conv2DTranspose(
#         1,
#         kernel_size=4,
#         strides=2,
#         padding="same",
#         activation="sigmoid",
#     )(x)
#     x = image_output
#     new_last = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 5, strides=1, activation='relu')
#     x = new_last(x)
#
#     return keras.Model(noise_input, x, name="generator")


def discriminator_model():
    # 下面搭建判别器架构，同样采用序贯模型
    inputs = tf.keras.layers.Input(shape=[CS_ratio, 1])  # length
    reshape1 = tf.keras.layers.Reshape((CS_ratio,))
    dense1 = tf.keras.layers.Dense(image_size * image_size, activation='tanh')
    reshape2 = tf.keras.layers.Reshape((image_size, image_size, 1))

    x = inputs
    x = reshape1(x)
    x = dense1(x)
    x = reshape2(x)

    x = Conv2D(64, (5, 5), padding='same',activation='tanh',name='block1',input_shape=(image_size,image_size,1))(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='same',name='block2')(x)
    x = Conv2D(128, (5, 5),activation='tanh',name='block3')(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='same',name='block4')(x)
    x = Flatten()(x)
    x = Dense(1024,activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

    # for _ in range(depth):
    #     x = layers.Conv2D(
    #         width,
    #         kernel_size=4,
    #         strides=2,
    #         padding="same",
    #         use_bias=False,
    #     )(x)
    #     x = layers.BatchNormalization(scale=False)(x)
    #     x = layers.LeakyReLU(alpha=leaky_relu_slope)(x)
    # x = layers.Flatten()(x)
    # x = layers.Dropout(dropout_rate)(x)
    # output_score = layers.Dense(1)(x)
    #
    # return keras.Model(inputs, output_score, name="discriminator")


    #model = Sequential()
    #model.add(x)
    # # 添加2维卷积层，卷积核大小为5×5，激活函数为tanh，输入shape在‘channels_first’模式下为（samples,channels，rows，cols）
    # # 在‘channels_last’模式下为（samples,rows,cols,channels），输出为64维
    # model.add(Conv2D(64, (5, 5),padding='same',input_shape=(image_size,image_size,1)))
    # model.add(Activation('tanh'))
    # # 为空域信号施加最大值池化，pool_size取（2，2）代表使图片在两个维度上均变为原长的一半
    # model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Conv2D(128, (5, 5)))
    # model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # # Flatten层把多维输入一维化，常用在从卷积层到全连接层的过渡
    # model.add(Flatten())
    # model.add(Dense(1024))
    # model.add(Activation('tanh'))
    # # 一个结点进行二值分类，并采用sigmoid函数的输出作为概念
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    # return model

# loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

ALPHA = 0
BETA = 0

def generator_loss(fake_out,gen_image,images):
    true_signal,false_signal = cal_signal(images,gen_image)
    l1_loss = tf.reduce_mean(tf.abs(true_signal - false_signal))
    fake_loss = cross_entropy(tf.ones_like(fake_out), fake_out) # 希望假的判别为1
    bg_loss = tf.reduce_mean(tf.abs(gen_image))
    loss = - tf.reduce_mean(fake_out) # WGAN-GP G 损失函数，最大化假样本的输出值

    gen_total_loss = fake_loss
    return gen_total_loss, bg_loss, l1_loss, loss

def gradient_penalty(discriminator, batch_x, fake_out):
    # 梯度惩罚项计算函数
    batchsz = batch_x.shape[0]
    #print(batch_x.shape)
    #print(fake_out.shape)
    # 每个样本均随机采样t,用于插值
    t = tf.random.uniform([batchsz, 1])
    # 自动扩展为x 的形状，[b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)
    # 在真假图片之间做线性插值
    interplate = t * batch_x + (1 - t) * fake_out
    # 在梯度环境中计算D 对插值样本的梯度
    with tf.GradientTape() as tape:
        tape.watch([interplate])  # 加入梯度观察列表
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits, interplate)
    # 计算每个样本的梯度的范数:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    # 计算梯度惩罚项
    gp = tf.reduce_mean((gp - 1.) ** 2)
    return gp



def discriminator_loss(real_out, fake_out, true_signal, false_signal):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out) # 希望真的判别为1
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out) # 希望假的判别为0
    gp_loss = gradient_penalty(discriminator, true_signal, false_signal)
    # disc_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out) + 10.* gp_loss
    disc_loss = real_loss + fake_loss
    return disc_loss, gp_loss



np.random.seed(2)
mask = np.random.randn(CS_ratio,784)
mask = mask.astype(np.float32)
mask_1 = np.array(mask).reshape(CS_ratio,28,28)
mask_2=mask_1.transpose(1,2,0)
m_matrix_tf = np.zeros((BATCH_SIZE,28,28,CS_ratio))
for i in range(BATCH_SIZE):
    m_matrix_tf[i,:,:,:]=mask_2
m_matrix_tf = m_matrix_tf.astype(np.float32)
m_matrix_tf = tf.convert_to_tensor(m_matrix_tf)



#print(randn_B)
# 创建生成器模型
generator = generator_model()
generator.summary()
# 创建辨别器模型
discriminator = discriminator_model()
discriminator.summary()

# Define the Optimizers and Checkpoint-saver
generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
checkpoint_dir = './training_checkpoints_mnist_cs5'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

log_dir="logs_mnist_cs5/"
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit_mnist_cs5/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def cal_signal(images, gen_image):
    # true_signal = np.zeros((BATCH_SIZE, K, 1))
    # false_signal = np.zeros((BATCH_SIZE, K, 1))
    #print(images.shape)
    #print(randn_B.shape)
    true_signal = images * m_matrix_tf
    false_signal = gen_image * m_matrix_tf

    true_signal = tf.reduce_mean(true_signal, axis=1)
    true_signal = tf.reduce_mean(true_signal, axis=1)

    false_signal = tf.reduce_mean(false_signal, axis=1)
    false_signal = tf.reduce_mean(false_signal, axis=1)
    # for i in range(BATCH_SIZE):
    #     real_image = np.array(images[i]).reshape(784, 1)
    #     fake_image = np.array(gen_image[i]).reshape(784, 1)
    #     real_signal = np.dot(randn_B, real_image)
    #     fake_signal = np.dot(randn_B, fake_image)
    #     true_signal[i, :, :] = real_signal  # 真实信号
    #     false_signal[i, :, :] = fake_signal  # 虚假信号
    # true_signal = true_signal / 130
    # false_signal = false_signal / 130
    # true_signal = true_signal.astype(np.float32)
    # false_signal = false_signal.astype(np.float32)
    return true_signal,false_signal


def train_step(images,epoch):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_image = generator(noise, training=True)
        #print(gen_image)

        true_signal, false_signal = cal_signal(images, gen_image)

        fake_out = discriminator(false_signal, training=True)
        real_out = discriminator(true_signal, training=True)


        gen_total_loss, bg_loss, l1_loss, loss = generator_loss(fake_out,gen_image,images)
        disc_loss,gp_loss = discriminator_loss(real_out, fake_out, true_signal,false_signal)



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
    for i in range(pre_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((pre_images[i, :, :, 0] + 1) / 2, cmap='gray')
        plt.axis('off')
        plt.savefig("./mnist_cs5/"+str(epoch)+"_" + ".png")
    #plt.show()

def generate_test_image(images,gen_model, test_noise,epoch):
    pre_images = gen_model(test_noise, training=False)
    # print(pre_images.shape)

    plt.imshow(pre_images[0] * 0.5 + 0.5, cmap='gray')
    plt.axis('off')
    plt.savefig("./mnist_cs_test5/" + str(epoch) + "_" + ".png")



def train(dataset, epochs):
    seed = tf.random.normal([num_exp_to_generate, noise_dim])  # 随机正态分布
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)
        for image_batch in dataset:
            # 优化
            train_step(image_batch,epoch)
            print('.', end='')
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
        generate_plot_image(generator, seed,epoch)
        # if epoch == 0:
        #     discriminator.summary()
    checkpoint.save(file_prefix=checkpoint_prefix)

# train(datasets,EPOCHS)
# checkpoint.restore(r"D:\cs_train_weights\training_checkpoints_mnist_cs3\ckpt-5") #cs_ratio = 98
#checkpoint.restore(r"C:\Users\程龙\PycharmProjects\pythonProject\GAN_CS\training_checkpoints_mnist_cs4\ckpt-1") #cs_ratio = 50
checkpoint.restore(r"C:\Users\程龙\PycharmProjects\pythonProject\GAN_CS\training_checkpoints_mnist_cs5\ckpt-2") #cs_ratio = 25

# num = 0
# for images in datasets.take(100):
#     seed = tf.random.normal([num_exp_to_generate, noise_dim])  # 随机正态分布
#     # generate_plot_image(generator,seed,num)
#     generate_test_image(images, generator, seed, num)
#     num = num + 1
















