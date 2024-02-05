# encoding: utf-8
import torchvision.transforms as T
import math
import random

import  numpy as np
from PIL import Image

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


# from https://github.com/finger-monkey/Data-Augmentation
# This is the code of Local Grayscale Transfomation
class LGT(object):
    """ Randomly selects a rectangle region in an image and replaces the RGB
        pixels within it with their greyscale value.
        'Eliminate Deviation with Deviation for Data Augmentation and a General Multi-modal Data Learning Method' by Gong et al.
        See https://arxiv.org/pdf/2101.08533.pdf
    Args:
         probability: The probability that the Random Color Dropout operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
    """
    def __init__(self, probability=0.2, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        new = img.convert("L")   # Convert from here to the corresponding grayscale image
        np_img = np.array(new, dtype=np.uint8)
        img_gray = np.dstack([np_img, np_img, np_img])

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size[1] and h < img.size[0]:
                x1 = random.randint(0, img.size[0] - h)
                y1 = random.randint(0, img.size[1] - w)
                img = np.asarray(img).astype('float')

                img[y1:y1 + h, x1:x1 + w, 0] = img_gray[y1:y1 + h, x1:x1 + w, 0]
                img[y1:y1 + h, x1:x1 + w, 1] = img_gray[y1:y1 + h, x1:x1 + w, 1]
                img[y1:y1 + h, x1:x1 + w, 2] = img_gray[y1:y1 + h, x1:x1 + w, 2]

                img = Image.fromarray(img.astype('uint8'))

                return img

        return img


class RandomColorJitter(object):
    """ Randomly alters brightness and contrast of the image.
        Specifically, this is a 'T.ColorJitter' wrapper that makes
        its application random.
        See https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html
    Args:
         probability: The probability that the Random Color Dropout operation will be performed.
         brightness: How much to jitter brightness.
         contrast: How much to jitter contrast.
    """
    def __init__(self, probability=.2, brightness=.5, contrast=.2):
        self.probability = probability
        self.brightness = brightness
        self.contrast = contrast
        self.color_jitter = T.ColorJitter(brightness=self.brightness, contrast=self.contrast)

    def __call__(self, img):
        p = random.random()
        if p < self.probability:
            img = self.color_jitter(img)
        return img