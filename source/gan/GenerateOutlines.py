from PIL import Image, ImageFilter
import os

for folder in os.listdir('data'):
    for file in os.listdir("data/{}".format(folder)):
        print(file)
        image = Image.open("data/{}/{}".format(folder, file))
        image = image.filter(ImageFilter.FIND_EDGES)
        image.save("dataOL/{}OL.png".format(file))
