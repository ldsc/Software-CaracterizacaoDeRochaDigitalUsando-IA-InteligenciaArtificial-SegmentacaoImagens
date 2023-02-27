import torch
from os import path
from utils.image import apply_binarization, calculate_porosity, save_image, save_porosity
from rock_model import RockNetModel


def apply_model(model_file, image, output=None, save=False):
    print(f"Running model in {model_file} on {image} image...")
    
    model = RockNetModel()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    dirname, file = path.split(image)
    filename, _ = path.splitext(file)
    
    if output:
        dirname = output
        
    
    arr, time = apply_binarization(image, model)
    
    save_porosity(calculate_porosity(arr), filename, f"{dirname}/porosity.txt", time)   
    
    if save:
        save_image(arr, f"{dirname}/{filename}")
        
    print("Done!")
        