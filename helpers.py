# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale


def my_imfilter(image: np.ndarray, filter: np.ndarray):
    """
  Your function should meet the requirements laid out on the project webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
    """
    filter_dim = filter.shape
    image_dim = image.shape

    if filter_dim[0] % 2 == 0 or filter_dim[1] % 2 == 0:
        raise ValueError('Filter must have odd dimensions')

    flipped_filter = np.flip(filter)
    flipped_filter_dim = flipped_filter.shape

    filtered_image = np.zeros(image_dim)

    if len(image_dim) == 2:  # Gray Scale Image
        padding_amount = (((filter_dim[0] - 1) // 2,) * 2, ((filter_dim[1] - 1) // 2,) * 2)
        padded_image = np.pad(image, padding_amount, constant_values=0)
        padded_image_dim = padded_image.shape

        for i in range(padded_image_dim[0] - flipped_filter_dim[0] + 1):
            for j in range(padded_image_dim[1] - flipped_filter_dim[1] + 1):
                image_part = padded_image[i:i + flipped_filter_dim[0], j:j + flipped_filter_dim[1]]
                conv_value = np.sum(np.multiply(image_part, flipped_filter), axis=(0, 1))
                filtered_image[i, j] = conv_value

    else:  # RGB
        padding_amount = (((filter_dim[0] - 1) // 2,) * 2, ((filter_dim[1] - 1) // 2,) * 2, (0, 0))
        padded_image = np.pad(image, padding_amount, constant_values=0)
        padded_image_dim = padded_image.shape

        for i in range(padded_image_dim[0] - flipped_filter_dim[0] + 1):
            for j in range(padded_image_dim[1] - flipped_filter_dim[1] + 1):
                image_part = padded_image[i:i + flipped_filter_dim[0], j:j + flipped_filter_dim[1]]
                for k in range(image_part.shape[2]):
                    conv_value = np.sum(np.multiply(image_part[:, :, k], flipped_filter))
                    filtered_image[i, j, k] = conv_value

    return filtered_image


def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
    """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

    assert image1.shape == image2.shape

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
    # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
    kernel = None

    # Your code here:
    low_frequencies = None  # Replace with your implementation

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    high_frequencies = None  # Replace with your implementation

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = None  # Replace with your implementation

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0,
    # and all values larger than 1.0 to 1.0.
    # (5) As a good software development practice you may add some checks (assertions) for the shapes
    # and ranges of your results. This can be performed as test for the code during development or even
    # at production!

    return low_frequencies, high_frequencies, hybrid_image


def vis_hybrid_image(hybrid_image: np.ndarray):
    """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales + 1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))
        # downsample image
        cur_image = rescale(cur_image, scale_factor, mode='reflect')
        # pad the top to append to the output
        pad = np.ones((original_height - cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))
    return output


def load_image(path):
    return img_as_float32(io.imread(path))


def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))
