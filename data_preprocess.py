
import random
import numpy as np
from PIL import Image
from torchvision import transforms, utils

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0
        img[mask == 1] = 255
        img = Image.fromarray(img.astype('uint8'))
        return img


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, std=0, amp=1.0):
        self.mean = mean
        self.std = std
        self.amp = amp

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        noise = self.amp * np.random.normal(loc=self.mean, scale=self.std, size=(h, w, 1))
        noise = np.repeat(noise, c, axis=2)
        img = img + noise
        img[img > 255] = 255
        img[img < 0] = 0
        img = Image.fromarray(img.astype('unit8'))
        return img


class PCAJitter(object):
    def __init__(self, g_mean=0, g_std=0.1):
        self.g_mean = g_mean
        self.g_std = g_std

    def __call__(self, img):
        img = np.asarray(img, dtype='float32')
        img = img / 255.0
        img1 = img.reshape(-1, 3)
        img1 = np.transpose(img1)
        img_cov = np.cov([img1[0], img1[1], img1[2]])
        lamda, p = np.linalg.eig(img_cov)
        p = np.transpose(p)
        alpha1 = random.gauss(self.g_mean, self.g_std)
        alpha2 = random.gauss(self.g_mean, self.g_std)
        alpha3 = random.gauss(self.g_mean, self.g_std)
        v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))
        add_num = np.dot(p, v)
        img2 = np.array([img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])
        img2 = np.swapaxes(img2, 0, 2)
        img2 = np.swapaxes(img2, 0, 1)
        img = Image.fromarray(img2.astype('uint8'))
        return img