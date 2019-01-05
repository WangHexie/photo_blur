import os

import numpy as np
from skimage import io
from skimage.filters import rank
from skimage.morphology import disk


def get_file_list_in_dir(path):
    return [name for name in os.listdir(path) if not os.path.isdir(os.path.join(path, name))]


def blur_img_in_folder(path):
    if not os.path.exists(os.path.join(path, "B")):
        os.mkdir(os.path.join(path, "B"))
    for file_name in get_file_list_in_dir(path):
        image = io.imread(os.path.join(path, file_name))
        selem = disk(20)
        color_array = [rank.mean(image[:, :, dim], selem=selem).reshape(*image.shape[:2], 1) for
                       dim in range(3)]
        result = np.concatenate(color_array, axis=2)
        io.imsave(os.path.join(path, "B", file_name), result)


if __name__ == '__main__':
    blur_img_in_folder(os.path.join("face", "test"))
