from keras.preprocessing import image
import numpy as np
from gan import define_generator, define_gan, define_discriminator, GENERATE_SQUARE, IMAGE_CHANNELS, save_images
from PIL import Image

img_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)

#discriminator = define_discriminator(img_shape)
generator = define_generator(img_shape)
#combined = define_gan(generator, discriminator, img_shape)

generator.load_weights("p2pgeneratorold.h5")

img = image.load_img('sketch3.png', target_size = (256,256))

img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
result = generator.predict(img)

save_images(6, result, generator)