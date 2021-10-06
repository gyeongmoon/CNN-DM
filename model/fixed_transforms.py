from __future__ import division
from PIL import Image, ImageOps
import numbers


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, random_samples):
        for t in self.transforms:
            if hasattr(t, 'name'):
                if t.name() is 'RandomCrop':
                    img = t(img, random_samples[:2])
                if t.name() is 'RandomHorizontalFlip':
                    img = t(img, random_samples[-1])
            else:
                img = t(img)
        return img


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, random_samples):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        # x1 = random.randint(0, w - tw)
        # y1 = random.randint(0, h - th)
        x1 = int(round(random_samples[0] * (w - tw)))
        y1 = int(round(random_samples[1] * (h - th)))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    @staticmethod
    def name():
        return 'RandomCrop'


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img, random_samples):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random_samples < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def name():
        return 'RandomHorizontalFlip'
