import math
import numpy as np

# Taken from https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html

def bilinear(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  img_height, img_width = image.shape[:2]

  resized = np.empty([height, width])

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  for i in range(height):
    for j in range(width):

      x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
      x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

      x_weight = (x_ratio * j) - x_l
      y_weight = (y_ratio * i) - y_l

      a = image[y_l, x_l]
      b = image[y_l, x_h]
      c = image[y_h, x_l]
      d = image[y_h, x_h]

      pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight

      resized[i][j] = pixel

  return resized

def bilinear(image, dimension):
    '''Bilinear interpolation method to convert small image to original image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension
    Returns:
    numpy.ndarray: Resized image
    '''
    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width)/(dimension[1])
    scale_y = (height)/(dimension[0])

    new_image = np.zeros((dimension[0], dimension[1]))


    for i in range(dimension[0]):
        for j in range(dimension[1]):
            x = (j+0.5) * (scale_x) - 0.5
            y = (i+0.5) * (scale_y) - 0.5

            x_int = int(x)
            y_int = int(y)

            # Prevent crossing
            x_int = min(x_int, width-2)
            y_int = min(y_int, height-2)

            x_diff = x - x_int
            y_diff = y - y_int

            a = image[y_int, x_int]
            b = image[y_int, x_int+1]
            c = image[y_int+1, x_int]
            d = image[y_int+1, x_int+1]

            pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff

            new_image[i, j] = pixel.astype(np.uint8)

    return new_image

    
def downsampled(self, channel, factor):
  rows, cols = channel.shape
  downsampled_rows, downsapled_cols = rows//factor, cols//factor
  downsampled = np.empty((downsampled_rows, downsapled_cols))
  for i in range(downsampled_rows):
      for j in range(downsapled_cols):
          window = channel[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
          downsampled[i, j] = np.mean(window)
  return downsampled