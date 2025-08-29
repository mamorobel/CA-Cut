import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

def generate_labels(csv_file,
                    original_output_lbls_dir,
                    scaled_output_lbls_dir,
                    scaled_output_csv_dir,
                    orignial_img_size,
                    scaled_img_size):
    
    df = pd.read_csv(csv_file)

    for i, row in df.iterrows():
        vx, vy = row.vp_x, row.vp_y
        lx, ly = row.l_x, row.l_y
        rx, ry = row.r_x, row.r_y

        '''
        left
        '''
        if lx < 0:
            a = (vy - ly)/(vx - lx)
            b = vy - (a * vx)
            lx = 0
            ly = a * lx + b

        elif ly == original_img_size[1]:
            ly = original_img_size[1] - 1

        '''
        right
        '''
        if rx >= original_img_size[0]:
            a = (vy - ry)/(vx - rx)
            b = vy - (a * vx)
            rx = original_img_size[0] - 1
            ry = a * rx + b

        elif ry == original_img_size[1]:
            ry = original_img_size[1] - 1

        '''
        scaling
        '''
        factor_x, factor_y = scaled_img_size[0]/original_img_size[0], scaled_img_size[1]/original_img_size[1]

        s_vx, s_vy = int(vx * factor_x), int(vy * factor_y)
        s_lx, s_ly = int(lx * factor_x), int(ly * factor_y)
        s_rx, s_ry = int(rx * factor_x), int(ry * factor_y)

        df.loc[i, 'vp_x'] = s_vx
        df.loc[i, 'vp_y'] = s_vy
        df.loc[i, 'l_x'] = s_lx
        df.loc[i, 'l_y'] = s_ly
        df.loc[i, 'r_x'] = s_rx
        df.loc[i, 'r_y'] = s_ry

        '''
        image generation
        '''

        # scaled label generation
        img = np.zeros((scaled_img_size[1], scaled_img_size[0], 3), dtype=np.uint8)
        img[s_vy, s_vx, 0] = 255
        img[s_ry, s_rx, 1] = 255
        img[s_ly, s_lx, 2] = 255

        img = Image.fromarray(img, mode='RGB')

        if not os.path.exists(scaled_output_lbls_dir):
            os.makedirs(scaled_output_lbls_dir)

        img_path = os.path.join(scaled_output_lbls_dir, row.image_name[:-3] + 'png')
        img.save(img_path)

        # original label generation
        img = np.zeros((original_img_size[1], original_img_size[0], 3), dtype=np.uint8)
        # print('ry: ',  ry, ' rx: ', rx)
        img[int(vy), int(vx), 0] = 255
        img[int(ry), int(rx), 1] = 255
        img[int(ly), int(lx), 2] = 255

        img = Image.fromarray(img, mode='RGB')

        if not os.path.exists(original_output_lbls_dir):
            os.makedirs(original_output_lbls_dir)

        img_path = os.path.join(original_output_lbls_dir, row.image_name[:-3] + 'png')
        img.save(img_path)
    
    df.to_csv(scaled_output_csv_dir, index=False)

if __name__ == '__main__':
    csv_file = '../data/gt_labels.csv'
    
    original_output_lbls_dir = '../data/gt_image_labels'
    scaled_output_lbls_dir = '../data/scaled_image_labels'
    scaled_output_csv_dir = '../data/scaled_labels.csv'

    original_img_size = (1280, 720)
    scaled_img_size = (320, 224)

    generate_labels(csv_file, original_output_lbls_dir, scaled_output_lbls_dir, scaled_output_csv_dir, original_img_size, scaled_img_size)