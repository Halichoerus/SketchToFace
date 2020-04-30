from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.layers import Concatenate
import numpy as np
from numpy.random import randint
from numpy import load
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import os

# Generation resolution - Must be square 
GENERATE_RES = 8 # (1=32, 2=64, 3=96, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3

# Preview image 
PREVIEW_ROWS = 7
PREVIEW_COLS = 4
PREVIEW_MARGIN = 16
SAVE_FREQ = 100

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
DATA_PATH = ''
EPOCHS = 30
BATCH_SIZE = 16


#datagen = ImageDataGenerator(
#  rescale=1./255,
#  shear_range=0.2,
#  zoom_range=0.2,
#  horizontal_flip=True)

datagen = ImageDataGenerator()

images_path = os.path.join(DATA_PATH,'data')

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.trainable = False
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

def save_images(cnt,noise, gen):
  image_array = np.full(( 
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3), 
      255, dtype=np.uint8)
  
  generated_images = gen.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  alternate = False
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        if image_count == noise.shape[0]:
          break

        r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        if alternate:
            image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255
            alternate = False
            image_count += 1
        else:
            image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = noise[image_count]
            alternate = True


          
  output_path = os.path.join(DATA_PATH,'output')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  filename = os.path.join(output_path,"train-{}.png".format(cnt))
  im = Image.fromarray(image_array)
  im.save(filename)


# Makes negative labels between 0-0.3 and positive 0.7-1
def add_label_smoothing(y):
    for i in range(y.size):
        if y[i] == 1:
            y[i] -= np.random.random() * 0.3
        else:
            y[i] += np.random.random() * 0.3

# Adds noise_percent% noise to the labels yr and yf by swapping random values
def add_label_noise(yr, yf, noise_percent):
    num = int(yr.size * noise_percent)

    for i in range(num):
        rand1 = int(np.random.random() * yr.size)
        rand2 = int(np.random.random() * yf.size)
        temp = float(yr[rand1])
        yr[rand1] = yf[rand2]
        yf[rand2] = temp

#generator.load_weights("gen.h5")
#combined.load_weights("disc.h5")
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()




def train():
    training_set = datagen.flow_from_directory(
    'data',
    target_size=(GENERATE_SQUARE,GENERATE_SQUARE),
    batch_size=BATCH_SIZE,
    class_mode=None)
    
    outline_set = datagen.flow_from_directory(
    'dataOL',
    target_size=(GENERATE_SQUARE,GENERATE_SQUARE),
    batch_size=BATCH_SIZE,
    class_mode=None)

    img_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)
    discriminator = define_discriminator(img_shape)
    generator = define_generator(img_shape)
    combined = define_gan(generator, discriminator, img_shape)
    
    cnt = 1
    sampleSaveImages = outline_set.next()
    outline_set.reset()
    for epoch in range(EPOCHS):
        batchCount = 0
        printProgressBar(batchCount,training_set.__len__() , prefix = 'Epoch {} Progress:'.format(cnt), suffix = 'Complete', length = 50)
        for batch in range(training_set.__len__()):
            batchCount += 1
            printProgressBar(batchCount, training_set.__len__(), prefix = 'Epoch {} Progress:'.format(cnt), suffix = 'Complete', length = 50)
            x_real = training_set.next()
            x_realOL = outline_set.next()

            x_real = (x_real - 127.5) / 127.5
            x_realOL = (x_realOL - 127.5) / 127.5

            x_fake = generator.predict(x_realOL)

            # Create the labels with smoothing and noise
            y_real = np.ones((x_real.shape[0],discriminator.output_shape[1],discriminator.output_shape[1],1))
            y_fake = np.zeros((x_real.shape[0],discriminator.output_shape[1],discriminator.output_shape[1],1))

            
            # Train discriminator on real and fake
            if(x_real.shape[0] != BATCH_SIZE or batch == training_set.__len__()-1):
              # This is also the end of the epoch so lets save an image

              d_loss1 = discriminator.train_on_batch([x_realOL, x_real], y_real)
	    	  # update discriminator for generated samples
              d_loss2 = discriminator.train_on_batch([x_realOL, x_fake], y_fake)
	    	  # update the generator
              g_loss, _, _ = combined.train_on_batch(x_realOL, [y_real, x_real])
	    	  # summarize performance
              print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (cnt, d_loss1, d_loss2, g_loss))

              #if(cnt % 1 == 0):
              save_images(cnt, sampleSaveImages, generator)
              generator.save(os.path.join(output_path,"p2pface_generator_epoch_{}.h5".format(cnt)))
              combined.save(os.path.join(output_path,"p2pface_discrim_epoch_{}.h5".format(cnt)))
              cnt += 1
              #print("Epoch {}, Discriminator accuarcy: {}, Generator accuracy: {}".format(epoch, discriminator_metric[0],generator_metric.T))
              #print("Epoch {} done.".format(epoch))
            else:
              discriminator.train_on_batch([x_realOL, x_real], y_real)
	    	  # update discriminator for generated samples
              discriminator.train_on_batch([x_realOL, x_fake], y_fake)
	    	  # update the generator
              combined.train_on_batch(x_realOL, [y_real, x_real])


