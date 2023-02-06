import pandas as pd
from PIL import Image
from os.path import join
import numpy as np

split = 'train'
base_path = 'resources/data/kidney'
df = pd.read_csv(base_path, split)
colors = []
for idx, row in df.iterrows():
    img = Image.open(join(base_path, row['file_path']))
    img_array = np.asarray(img)
    avg_color_per_row = np.average(np.asarray(img), axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    colors.append(avg_color)
colors = np.array(colors)
