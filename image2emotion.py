import torch
import pandas as pd
import os.path as osp
import numpy as np
import torch.nn.functional as F
import cv2
from torchvision import transforms
from PIL import Image
from artemis.in_out.neural_net_oriented import torch_load_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch_load_model('./ImageToEmotionclassifier.pt')
image = cv2.imread('E:\\202208\\test.jpeg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]
img_dim = 256
normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
resample_method = Image.LANCZOS

def evaluate_data(model,image):
    with torch.no_grad():
        model.eval()
        #trans = transforms.ToTensor()
        #image = cv2.resize(image,dsize=(img_dim,img_dim),interpolation=cv2.INTER_LANCZOS4)
        input_img= Image.fromarray(image)
        trans = transforms.Compose([transforms.Resize((img_dim, img_dim),resample_method),transforms.ToTensor(),normalize])
        img = trans(input_img)
        img = img.to(device)
        img = img.view(-1,3,img_dim,img_dim)
        logits = model(img)
        result= F.softmax(logits, dim=-1).cpu()
        print(result)

evaluate_data(model,image)
