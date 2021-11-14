import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


class LandCoverData():
    """Class to represent the S2GLC Land Cover Dataset for the challenge,
    with useful metadata and statistics.
    """
    # image size of the images and label masks
    IMG_SIZE = 256
    # the images are RGB+NIR (4 channels)
    N_CHANNELS = 4
    # we have 9 classes + a 'no_data' class for pixels with no labels (absent in the dataset)
    N_CLASSES = 10
    CLASSES = [
        'no_data',
        'clouds',
        'artificial',
        'cultivated',
        'broadleaf',
        'coniferous',
        'herbaceous',
        'natural',
        'snow',
        'water'
    ]
    # classes to ignore because they are not relevant. "no_data" refers to pixels without
    # a proper class, but it is absent in the dataset; "clouds" class is not relevant, it
    # is not a proper land cover type and images and masks do not exactly match in time.
    IGNORED_CLASSES_IDX = [0, 1]

    # The training dataset contains 18491 images and masks
    # The test dataset contains 5043 images and masks
    TRAINSET_SIZE = 18491
    TESTSET_SIZE = 5043

    # for visualization of the masks: classes indices and RGB colors
    CLASSES_COLORPALETTE = {
        0: [0,0,0],
        1: [255,25,236],
        2: [215,25,28],
        3: [211,154,92],
        4: [33,115,55],
        5: [21,75,35],
        6: [118,209,93],
        7: [130,130,130],
        8: [255,255,255],
        9: [43,61,255]
        }
    CLASSES_COLORPALETTE = {c: torch.tensor(color, dtype=torch.float) for (c, color) in CLASSES_COLORPALETTE.items()}

    # statistics
    # the pixel class counts in the training set
    TRAIN_CLASS_COUNTS = np.array(
        [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605]
    )
    # the minimum and maximum value of image pixels in the training set
    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def show_mask(mask_tensor):
    # masks_tensor: H, W
    show_mask = torch.zeros((*mask_tensor.size(), 3))
    for c, color in LandCoverData.CLASSES_COLORPALETTE.items():
        show_mask[mask_tensor == c, :] = color / 255
    return show_mask


def show_image(image_tensor):
    """ Show a image tensor loaded from dataloader
    :param image_tensor: C,H,W
    return a displayable tensor
    """
    # for image diaplay in tensorboard
    display_min = 50
    display_max = 3000
    image = image_tensor * LandCoverData.TRAIN_PIXELS_MAX
    image = image.clip(display_min, display_max)
    image = image / 1000  # divide by empirical mean
    image = torch.as_tensor(image.clip(0, 1), dtype=torch.float)
    return image


def ratio(mask):
    batch_size = mask.shape[0]
    size = mask.shape[1]
    res = torch.zeros((batch_size, 10))
    for i in range(10):
        res[:, i] = (mask == i).sum(dim=(1, 2))/(size*size)
    return res


def write_csv(index, matrix, path):
    #index must be a tensor with size = batch_size
    #matrix must be a tensor withe shape =batch_size * 256 * 256
    batch_size = matrix.shape[0]
    idx = index.reshape(batch_size, -1)
    myratio = ratio(matrix)
    data_pure = torch.cat((idx, myratio), dim=1)
    data = pd.DataFrame(data_pure.numpy() , columns = ['sample_id','no_data','clouds','artificial','cultivated','broadleaf','coniferous','herbaceous','natural','snow','water'])
    data = data.astype({'sample_id': 'int'})
    data.to_csv(path, index=False)
    print("make cvs done, with batch size = ", batch_size)
