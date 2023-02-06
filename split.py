import os
import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tissue-type', default='kidney')
args = parser.parse_args()

path = f'resources/datasets/{args.tissue_type}'
seed = 0
classes = sorted(list(os.listdir(path)))

patches = []
for cls in os.listdir(path):
    cls_path = join(path, cls, '0')
    for slide in os.listdir(cls_path):
        slide_path = join(cls_path, slide)
        for patch in os.listdir(slide_path):
            patches.append({
                'file_path': join(cls, '0', slide, patch),
                'cls': cls,
                'label': classes.index(cls),
                'barcode': slide,
                'tss': slide.split('-')[1],
                'patient_id': slide.split('-')[2]
            })
df = pd.DataFrame(patches)


def make_equal_size(df):
    N = df.label.value_counts().min()
    subsets = []
    for label in df.label.unique():
        subsets.append(df[df.label == label].sample(N, random_state=seed))
    return pd.concat(subsets).sample(frac=1.0, random_state=seed)


train_patients, test_patients = train_test_split(df.patient_id.unique(), test_size=0.33, random_state=seed)

train = make_equal_size(df[df.patient_id.isin(train_patients)])
test = make_equal_size(df[df.patient_id.isin(test_patients)])
train.to_csv(join(path, 'train.csv'), index=False)
test.to_csv(join(path, 'test.csv'), index=False)
