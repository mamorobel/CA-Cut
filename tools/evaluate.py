import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ImageData(Dataset):

    def __init__(self, df, transform):
        super().__init__()
        self.df = df
        self.transform = transform
        self.dir = "../data/labeled"
        self.label_dir = "../data/gt_image_labels"
        self.img_path, self.label_path = self.get_image_paths()

    def get_image_paths(self):
        images = []
        labels = []

        for image in self.df['image_name']:
            images_full_path = os.path.join(self.dir, image)
            labels_full_path = os.path.join(self.label_dir, image)

            if os.path.isfile(images_full_path) and image.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(images_full_path)
                labels.append(labels_full_path)

        return images, labels
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image = Image.open(self.img_path[index]).convert('RGB')
        image = self.transform(image)

        label = Image.open(self.label_path[index][:-3]+'png').convert('RGB')
        label = self.transform(label)

        v_x = self.df['vp_x'].iloc[index]
        v_y = self.df['vp_y'].iloc[index]
        l_x = self.df['l_x'].iloc[index]
        l_y = self.df['l_y'].iloc[index]
        r_x = self.df['r_x'].iloc[index]
        r_y = self.df['r_y'].iloc[index]

        return image, label, v_x, v_y, l_x, l_y, r_x, r_y
    
def l2_distance(model, data, device='cuda'):
    l_accuracy = []
    r_accuracy = []
    v_accuracy = []
    dr_diffs = []
    outliers = {}
    with torch.no_grad():
        for image, label, v_x, v_y, l_x, l_y, r_x, r_y in data:
            image = image.to(device).unsqueeze(0)
            label = label.to(device).unsqueeze(0)

            prediction = model(image).squeeze(0).cpu()

            r = prediction[0, :, :]
            g = prediction[1, :, :]
            b = prediction[2, :, :]

            channel_shape = r.shape

            r = torch.flatten(r)
            r = torch.nn.functional.softmax(r) * 255
            r = r.reshape(channel_shape)

            g = torch.flatten(g)
            g = torch.nn.functional.softmax(g) * 255
            g = g.reshape(channel_shape)

            b = torch.flatten(b)
            b = torch.nn.functional.softmax(b) * 255
            b = b.reshape(channel_shape)

            '''
            The following three lines are the (x,y) corrdinates
            for the vanishing, right, and left, predictions
            '''
            yv, xv = np.unravel_index(np.argmax(r), channel_shape)
            yr, xr = np.unravel_index(np.argmax(g), channel_shape)
            yl, xl = np.unravel_index(np.argmax(b), channel_shape)

            scaled = True

            if scaled:
                xv = (xv / 320) * 1280
                yv = (yv / 224) * 720

                xl = (xl / 320) * 1280
                yl = (yl / 224) * 720

                xr = (xr / 320) * 1280
                yr = (yr / 224) * 720

            def get_accuracy(xv, yv, xr, yr, xl, yl, v_x, v_y, l_x, l_y, r_x, r_y):
                # vanishing point
                point1 = np.array([xv, yv])
                point2 = np.array([v_x, v_y])
                vanishing_acc = np.linalg.norm(point1-point2)

                # right point
                point3 = np.array([xr, yr])
                point4 = np.array([r_x, r_y])
                right_acc = np.linalg.norm(point3-point4)

                #left point
                point5 = np.array([xl, yl])
                point6 = np.array([l_x, l_y])
                left_acc = np.linalg.norm(point5-point6)

                return vanishing_acc, right_acc, left_acc
            
            vanish_acc, right_acc, left_acc = get_accuracy(xv, yv, xr, yr, xl, yl, v_x, v_y, l_x, l_y, r_x, r_y)
            v_accuracy.append(vanish_acc)
            r_accuracy.append(right_acc)
            l_accuracy.append(left_acc)
        # plt.title('Channel-Wise Accuracy')
        return v_accuracy, r_accuracy, l_accuracy



def split_data(df, train_size: int):
    assert 0 < train_size < 101, 'train size should be in the range [0, 100]'

    div = int((len(df) * train_size)/100)

    train = df.iloc[:div]
    test  = df.iloc[div:]

    return train, test

def main(model):
    df = pd.read_csv("../data/gt_labels.csv")
    df = df.sort_values(by=['image_name'])
    
    seq1 = df[:273]
    seq2 = df[273:407]
    seq3 = df[407:719]
    seq4 = df[719:932]
    seq5 = df[932:1030]

    t1, v1 = split_data(seq1, train_size=80)
    t2, v2 = split_data(seq2, train_size=80)
    t3, v3 = split_data(seq3, train_size=80)
    t4, v4 = split_data(seq4, train_size=80)
    t5, v5 = split_data(seq5, train_size=80)

    training = [t1, t2, t3, t4, t5]
    train_sample = pd.concat(training)

    validation = [v1, v2, v3, v4, v5]
    validation_sample = pd.concat(validation)

    transform = transforms.Compose([
        transforms.Resize((224, 320)),
        transforms.ToTensor(),
    ])

    data = ImageData(validation_sample, transform=transform)
    
    return l2_distance(model, data)