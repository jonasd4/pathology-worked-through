from PIL import Image
import matplotlib.pyplot as plt
import os

base = 'Kidney_renal_clear_cell_carcinoma/0/'
type = 'TCGA-DV'
for path in os.listdir(base):
    if path.startswith(type):
        for file in list(os.listdir(os.path.join(base, path)))[:1]:
            img = Image.open(os.path.join(base, f"{path}/{file}"))
            plt.axis('off')
            plt.imshow(img)
            plt.show()
