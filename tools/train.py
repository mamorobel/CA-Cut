import os
import sys
import time
import yaml
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import evaluate

from PIL import Image 
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

class ImageData(Dataset):

    def __init__(self, df, transform):
        super().__init__()
        self.df = df
        self.transform = transform
        self.dir = "../data/labeled"
        self.label_dir = "../data/scaled_image_labels"
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
    
class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.checkpoint_path = '../checkpoints/resnet18_checkpoint.pth'
        torch.save(self.model.state_dict(), self.checkpoint_path)
        
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.base_layers = list(self.model.children())

        self.encoder()
        self.decoder()

    def residual_block(self, in_channel, out_channel, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, padding=padding),
            nn.ReLU()
        )
    
    def encoder(self):
        self.layer1 = nn.Sequential(*self.base_layers[:3])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = nn.Sequential(self.base_layers[5])
        self.layer4 = nn.Sequential(self.base_layers[6])
        self.bottleneck = nn.Sequential(self.base_layers[7])

        self.skip1 = self.residual_block(64, 64, 1, 0)
        self.skip2 = self.residual_block(64, 64, 1, 0)
        self.skip3 = self.residual_block(128, 128, 1, 0)
        self.skip4 = self.residual_block(256, 256, 1, 0)

        self.skip_orginal = self.residual_block(3, 3, 1, 0)

    def decoder(self):
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up4 = self.residual_block(256 + 512, 256, 1, 0)
        self.conv_up3 = self.residual_block(128 + 256, 128, 1, 0)
        self.conv_up2 = self.residual_block(64 + 128, 64, 1, 0)
        self.conv_up1 = self.residual_block(64 + 64, 3, 1, 0)
        self.out_conv = self.residual_block(3 + 3, 3, 1, 0)

    def forward(self, input):
        x_original = self.skip_orginal(input)

        '''
        encoder
        '''
        x1 = self.layer1(input)
        x1_skip = self.skip1(x1)

        x2 = self.layer2(x1)
        x2_skip = self.skip2(x2)

        x3 = self.layer3(x2)
        x3_skip = self.skip3(x3)

        x4 = self.layer4(x3)
        x4_skip = self.skip4(x4)

        '''
        bottleneck
        '''
        bottleneck = self.bottleneck(x4)

        '''
        decoder
        '''
        x = self.upsample(bottleneck)
        x = self.conv_up4(torch.cat((x, x4_skip), dim=1))

        x = self.upsample(x)
        x = self.conv_up3(torch.cat((x, x3_skip), dim=1))
        
        x = self.upsample(x)
        x = self.conv_up2(torch.cat((x, x2_skip), dim=1))

        x = self.upsample(x)
        x = self.conv_up1(torch.cat((x, x1_skip), dim=1))

        x = self.upsample(x)
        out = self.out_conv(torch.cat((x, x_original), dim=1))

        return out
    
def get_candidate(v_x, v_y, l_x, l_y, r_x, r_y, epochs, starting, ending):
    alpha = random.random()
    x = 0
    y = 0
    blur = 100
    if random.random() > .5:
        x = ((r_x - v_x) * alpha) + v_x + int(np.clip(np.random.normal(loc=0, scale=blur), -blur, blur))
        y = ((r_y - v_y) * alpha) + v_y + int(np.clip(np.random.normal(loc=0, scale=blur), -blur, blur))
    else:
        x = ((l_x - v_x) * alpha) + v_x + int(np.clip(np.random.normal(loc=0, scale=blur), -blur, blur))
        y = ((l_y - v_y) * alpha) + v_y + int(np.clip(np.random.normal(loc=0, scale=blur), -blur, blur))
    
    return int(x), int(y)

def custom_erase(img, candidates, height, width):
    x, y = candidates

    img_height, img_width = img.shape[:2]

    x_start = max(0, min(x, img_width - width))
    y_start = max(0, min(y, img_height - height))

    img[y_start:y_start + height, x_start:x_start + width] = [0, 0, 0]

    img = torch.from_numpy(img).permute(2,0,1)

    return img
    
def train(model, optimizer, loss_fn, train_loader, val_loader, img_height, img_width,config, device, epochs):
    criterion = loss_fn
    train_loss = []
    val_loss = []

    l_acc = []
    r_acc = []
    v_acc = []
    average_err = []

    lowest = float('inf')

    for epoch in range(epochs):
        training_loss = 0.0

        model.train()

        for batch in train_loader:
            optimizer.zero_grad()

            inputs, targets, v_x, v_y, l_x, l_y, r_x, r_y = batch

            if config['cutout']:
                width, height = config['mask_size']

                cutout = transforms.RandomErasing(
                    p=1, scale=(width, height), ratio=(1.0, 1.0), value=0, inplace=False
                )
            
                for i, (input, target) in enumerate(zip(inputs, targets)):
                    for j in range(config['num_cuts']):
                        inputs[i] = cutout(inputs[i])

            if config['ca_cut']:
                
                if config['num_uninformed'] != 0:
                    width, height = config['mask_size']
                    width = height = (width * height) / (img_width * img_height)
                    cutout = transforms.RandomErasing(
                        p=1, scale=(width, height), ratio=(1.0, 1.0), value=0, inplace=False
                    )

                for i, (input, target) in enumerate(zip(inputs, targets)):
                    num_cuts = config['num_cuts']
                    num_uninformed = config['num_uninformed']

                    for j in range(num_cuts-num_uninformed):
                        candidates = get_candidate(v_x[i], v_y[i], l_x[i], l_y[i], r_x[i], r_y[i], epochs, *config['curriculum'])
                        inputs[i] = custom_erase(inputs[i].cpu().permute(1,2,0).detach().numpy(), candidates, *config['mask_size'])
                    for j in range(num_uninformed):
                        inputs[i] = cutout(inputs[i])

            if config['h_flip']:
                for i, (input, target) in enumerate(zip(inputs, targets)):
                    if np.random.random() > .5:
                        inputs[i] = transforms.functional.hflip(input)
                        temp = torch.clone(target[1])
                        targets[i, 1] = target[2]
                        targets[i, 2] = temp                
                        targets[i] = transforms.functional.hflip(targets[i])

            if config['v_flip']:
                for i, (input, target) in enumerate(zip(inputs, targets)):
                    if np.random.random() > .5:
                        inputs[i] = transforms.functional.vflip(input)
                        targets[i] = transforms.functional.vflip(targets[i])

            inputs = transforms.GaussianBlur(kernel_size=(11, 11))(inputs)
            inputs = transforms.ColorJitter(brightness=0.5, contrast=.5, saturation=0.5, hue=0.5)(inputs)

            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)

            targets = torch.flatten(targets, start_dim=0, end_dim=1)
            targets = torch.flatten(targets, start_dim=1)

            output = torch.flatten(output, start_dim=0, end_dim=1)
            output = torch.flatten(output, start_dim=1)
            
            loss = criterion(output, targets)
            
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        
        training_loss = training_loss/len(train_loader)
        train_loss.append(training_loss)

        model.eval()

        validation_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, v_x, v_y, l_x, l_y, r_x, r_y = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)

                targets = torch.flatten(targets, start_dim=0, end_dim=1)
                targets = torch.flatten(targets, start_dim=1)
                
                output = torch.flatten(output, start_dim=0, end_dim=1)
                output = torch.flatten(output, start_dim=1)

                loss = criterion(output, targets).item()
                validation_loss += loss

        validation_loss = validation_loss/len(val_loader)
        val_loss.append(validation_loss)

        v_err, r_err, l_err = evaluate.main(model=model)
        v_acc.append(np.average(v_err))
        r_acc.append(np.average(r_err))
        l_acc.append(np.average(l_err))

        if validation_loss < lowest:
            torch.save(model.state_dict(), f"../checkpoints/{config['model_name']}_best.pth")
            lowest = validation_loss

        print(f'Epoch: {epoch}, Train Loss: {training_loss:.4f}, Val Loss: {validation_loss:.4f}')

        plt.figure(num=1)
        plt.plot(list(range(len(train_loss))), train_loss, label='training loss')
        plt.plot(list(range(len(val_loss))), val_loss, label='validation loss')
        plt.legend()
        plt.savefig('../plots/train_validation loss plot.png')
        plt.close()

        fig, ax1 = plt.subplots(num=2)
        overall = np.average([np.average(v_err),
                      np.average(r_err),
                      np.average(l_err)])
        average_err.append(overall)

        # Left Y-axis (validation errors)
        ax1.plot(list(range(len(v_acc))), average_err, '-r', label='average error')
        ax1.set_ylabel("Error metrics")
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(list(range(len(val_loss))), val_loss, '--k', label='validation loss')
        ax2.set_ylabel("Validation Loss")
        ax2.legend(loc='upper right')

        plt.savefig('../plots/validation_err_with_loss.png')
        plt.close()
    
def split_data(df, train_size: int):
    assert 0 < train_size < 101, 'train size should be in the range [0, 100]'

    div = int((len(df) * train_size)/100)

    train = df.iloc[:div]
    test  = df.iloc[div:]

    return train, test

def main(args):
    config = args.config
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    df = pd.read_csv("../data/scaled_labels.csv")

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

    img_height, img_width = 224, 320

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])

    train_data = ImageData(df=train_sample, transform=transform)
    validation_data = ImageData(df=validation_sample, transform=transform)

    train_loader = DataLoader(dataset=train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset=validation_data, batch_size=config['batch_size'], shuffle=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UNet()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    train(model, optimizer, loss_fn, train_loader, val_loader, img_height, img_width,
          config, device, epochs=config['n_epochs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training CA-Cut Model')
    parser.add_argument('--config', type=str, required=True, help='Location of the configuration file you would like to run')

    args = parser.parse_args()
    main(args)