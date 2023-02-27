from os import path
from posixpath import split, splitext
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.time_messure import timing_decorator

def binarize(arr, img_size, net):
    width, height = img_size
    binarr = []
    for row in arr:
        for pixel in row:
            data = torch.Tensor(pixel)
            output = torch.argmax(net(data.view(-1,3)))
            binarr.append(output)
    return np.reshape(binarr, (height, width))

@timing_decorator
def apply_binarization(image, net):
    try:
        img = Image.open(image).convert("RGB")    
        arr = np.asarray(img)
        
        return binarize(arr, img.size, net)
    except IndexError as e:
        print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

def calculate_porosity(arr):
    total = arr.shape[0] * arr.shape[1]
    pores = 0
    for i in arr:
        for j in i:
            pores += int(j)
    return pores / total

def save_porosity(porosity, name, file, time):
    with open(file, "a") as f:
        f.write(f"{name} \t {porosity} \t {time}\n")

def show_image(arr):
    plt.imshow(arr, cmap="gray", interpolation="nearest")
    plt.show()
    
def save_image(arr, name):
    Image.fromarray((arr * 255).astype(np.uint8)).save(f"{name}-bin.png")
    
def convert_image_to_rgb_format(image_name):
    img = Image.open(image_name).convert("RGB")
    name, _ = path,splitext()
    img.save(f"{name}-converted.png")
     
