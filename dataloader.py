import imp
import glob
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.images = glob.glob(f"{img_dir}/*")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        pil_image = Image.open(img_path).convert("RGB")
        image = self.transform(pil_image)
        image_ = image.unsqueeze(0)
        image_x2 = F.interpolate(image_, scale_factor=0.5, recompute_scale_factor=True, mode='area')
        image_x4 = F.interpolate(image_, scale_factor=0.25, recompute_scale_factor=True, mode='area')
        image_x2 = image_x2.squeeze(0)
        image_x4 = image_x4.squeeze(0)
        pil_image.close()
        return image, image_x2, image_x4


def get_dataset(dataroot, image_size):

    transform = transforms.Compose([transforms.Resize(image_size, Image.LANCZOS),
                                    transforms.RandomCrop(image_size),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    # RandomResizedCrop,
                                    ])

    dataset = ImageDataset(dataroot, transform)
    return dataset