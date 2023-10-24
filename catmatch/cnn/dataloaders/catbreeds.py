import os
from pathlib import Path
from venv import create

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io.image import read_image  # type: ignore


class CatbreedDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.dic = dict()  # Kan slettes?
        self.paths = []
        self.labels = []
        # Need to convert the labels to numbers
        # Create a dictionary of label to number
        dic = {}
        for folder in os.listdir(img_dir):
            dic[folder] = len(dic)
            for file in os.listdir(img_dir + folder):
                self.paths.append(Path(img_dir) / Path(folder) / Path(file))
                self.labels.append(folder)

        print("dic", dic)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # os.listdir(img_dir + folder)[3]  [5] [78] Denne kan vel også slettes

        img_path = self.paths[idx]
        image = read_image(str(img_path))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_catbreed_dataloader(img_dir, transform=None, target_transform=None):
    # TODO: Equally split such that there are equal number of each breed in each set
    dataset = CatbreedDataset(img_dir, transform, target_transform)
    train, val, test = random_split(dataset, [0.8, 0.1, 0.1])
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    val_dl = DataLoader(val, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=32, shuffle=True)
    return train_dl, val_dl, test_dl


def main():
    torch.manual_seed(42)
    train, val, test = create_catbreed_dataloader("./.data/")

    # image size torch.Size([3, 500, 333])
    # British shorthair
    # Find correct image (image 250 in the British Shorthair folder)


if __name__ == "__main__":
    main()
