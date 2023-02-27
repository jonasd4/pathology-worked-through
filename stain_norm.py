import argparse
import torch
from torchvision import transforms
import torchstain
import cv2
import os
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil


class PatchDataset(VisionDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths = []
        for cls in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, cls)):
                for path in os.listdir(os.path.join(self.root, cls)):
                    self.paths.append(os.path.join(cls, path))

    def __getitem__(self, index: int):
        path = self.paths[index]

        image_path = os.path.join(self.root, path)
        sample = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.paths)


parser = argparse.ArgumentParser()
parser.add_argument('--ref', metavar='--r', type=str, nargs='?',
                    help="set which reference image to use.")
parser.add_argument('--img', metavar='--i', type=str, nargs='?',
                    help="set path to which folder containing the images to normalize.")
parser.add_argument('--out', metavar='--o', type=str, nargs='?', default="./",
                    help="set path to store the output.")
args = parser.parse_args()

target = cv2.cvtColor(cv2.imread(args.ref), cv2.COLOR_BGR2RGB)

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
torch_normalizer.fit(T(target))

dataset = PatchDataset(root=args.img, transform=T)
loader = DataLoader(dataset, batch_size=1, prefetch_factor=64, num_workers=8)
for x, path in tqdm(loader):
    path = path[0]
    os.makedirs(os.path.dirname(os.path.join(args.out, path)), exist_ok=True)
    try:
        norm, H, E = torch_normalizer.normalize(I=x.squeeze(0), stains=True)
        cv2.imwrite(os.path.join(args.out, path), cv2.cvtColor(norm.cpu().numpy().astype("uint8"), cv2.COLOR_RGB2BGR))
    except:
        print('error')
        shutil.copy(os.path.join(args.img, path), os.path.join(args.out, path))
        # cv2.imwrite(os.path.join(args.out, path), cv2.cvtColor(x.squeeze(0).cpu().numpy().astype("uint8"), cv2.COLOR_RGB2BGR))
