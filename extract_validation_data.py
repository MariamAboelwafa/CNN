import os
import shutil
import random

data='D:/Mariam/final/256_ObjectCategories'

classes=os.listdir(data)
ref_copy=15
for i in range (0,257):
    newpath=r'validation_data/%s'
    os.makedirs(newpath)
    current=main_data + '/' + classes[i]
    for k in range(ref_copy):
        chosen_image = random.choice(os.listdir(current))
        file_to_move= current + '/' + chosen_one
        shutil.move(file_to_move, newpath)



