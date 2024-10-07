from PIL import Image
import cv2 as cv
from torch.utils.data import Dataset
import numpy as np

class SegmentationDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, index):
        img_path = self.dataset.iloc[index, :]['image_path']
    

        mask_path = self.dataset.iloc[index, :]['mask_path']


        image = np.array(Image.open(img_path))[:,:, :3].copy()
        image = cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)

        mean_attention_map = np.array(Image.open(self.dataset.iloc[index, :]['mean_attention_map']).convert("L"))
        mean_attention_map = cv.normalize(mean_attention_map, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
     

        if image.shape[0] != 256:
          image = cv.resize(image, (256, 256),
                    interpolation=cv.INTER_LINEAR)


        mask = np.array(Image.open(mask_path))
        if mask.shape[0] != 256:
          mask = cv.resize(mask, (256, 256),
                    interpolation=cv.INTER_LINEAR)

        if len(mask.shape) == 3:
          mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        # Binarize the maske (0 or 1) as the original masks pixels contain values from 0 to 255
        mask = 1.*(mask != 0)
        data = np.concatenate((image, mean_attention_map), axis=2)
        if self.transform is not None:
            augmentations = self.transform(image=data, mask=mask)
            data = augmentations["image"]
            mask = augmentations["mask"]

        return data, mask