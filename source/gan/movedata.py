import os
import shutil

for folder in os.listdir('data'):
    for file in os.listdir("data/{}".format(folder)):
        shutil.move("data/{}/{}".format(folder, file), "newData")
        
