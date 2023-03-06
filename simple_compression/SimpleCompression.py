import cv2
import numpy as np
import math

class Encoder:
    def __init__(self, img_path):
        self.image = cv2.imread(img_path, 1)
        self.ycbcr = ()

    def encode(self):
        '''
        Downsamples image in YCbCr colorspace
        Returns: Downsampled Y, Cb, Cr channels as tuple
        '''
        B, G, R = cv2.split(self.image)

        Y, Cb, Cr = self.rgb2ycbcr(R, G, B)

        # Downsample Y by a factor of 2
        Y = Y[::2, ::2]

        # Downsample Cb and Cr by a factor of 4
        Cb = Cb[::4, ::4]
        Cr = Cr[::4, ::4]

        self.ycbcr = (Y, Cb, Cr)
    
    def rgb2ycbcr(self, R, G, B):
        '''
        Convert RGB colorspace to YCbCr colorspace
        Parameters: R, G, B channels
        Returns: Y, Cb, Cr channels as tuple
        '''
        Y = 0.257*R + 0.504*G + 0.098*B + 16
        Cb = -0.148*R - 0.291*G + 0.439*B + 128
        Cr = 0.439*R - 0.368*G - 0.071*B + 128

        Y = np.clip(Y, 0, 255).astype(np.uint8)
        Cb = np.clip(Cb, 0, 255).astype(np.uint8)
        Cr = np.clip(Cr, 0, 255).astype(np.uint8)

        return (Y, Cb, Cr)
    
    def get_ycbcr(self):
        return self.ycbcr
    
class Decoder:
    def __init__(self, ycbcr, size):
        self.ycbcr = ycbcr
        self.size = size

    def decode(self):
        '''
        Upsamples image in YCbCr colorspace
        Parameters: Y, Cb, Cr channels
        Returns: Reconstructed image
        '''
        # Upsample Y by a factor of 2
        Y = self.__bilinear(self.ycbcr[0], self.size)

        # Upsample Cb and Cr by a factor of 4
        Cb = self.__bilinear(self.ycbcr[1], self.size)
        Cr = self.__bilinear(self.ycbcr[2], self.size)

        R, G, B = self.__ycbcr2rgb(Y, Cb, Cr)

        self.image = cv2.merge((B, G, R))
        return cv2.merge((B, G, R))

    def __ycbcr2rgb(self, Y, Cb, Cr):
        '''
        Convert YCbCr colorspace to RGB colorspace
        Parameters: Y, Cb, Cr channels
        Returns: R, G, B channels as tuple
        '''
        R = 1.164*(Y - 16) + 1.596*(Cr - 128)
        G = 1.164*(Y - 16) - 0.392*(Cb - 128) - 0.813*(Cr - 128)
        B = 1.164*(Y - 16) + 2.017*(Cb - 128)

        R = np.clip(R, 0, 255).astype(np.uint8)
        G = np.clip(G, 0, 255).astype(np.uint8)
        B = np.clip(B, 0, 255).astype(np.uint8)

        return (R, G, B)

    def __bilinear(self, image, dimension):
        '''
        Bilinear interpolation
        Parameters: Downsampled channel and dimension to upsample to
        Returns: Upsampled channel
        '''
        height, width = image.shape[:2]
        new_height, new_width = dimension

        upsampled = np.empty([new_height, new_width])

        # Find the ratio/interval between every new pixel
        # The -1 is to account for the already existing pixels
        x_ratio = float((width - 1) / (new_width - 1))
        y_ratio = float((height - 1) / (new_height - 1))

        for i in range(new_height):
            for j in range(new_width):

                # Find the coordinates of the 4 surrounding pixels
                x1 = math.floor(x_ratio * j)
                y1 = math.floor(y_ratio * i)
                x2 = math.ceil(x_ratio * j)
                y2 = math.ceil(y_ratio * i)

                # The 4 nearest points to the unknown pixel are P(y1, x1), P(y1, x2), P(y2, x1), P(y2, x2)
                # Note that indexing is [y, x] since the image dimensions are (height, width)
                p11 = image[y1, x1]
                p12 = image[y1, x2]
                p21 = image[y2, x1]
                p22 = image[y2, x2]

                # Find the weights for each pixel based on distance from the pixel to be interpolated
                x_weight = (x_ratio * j) - x1
                y_weight = (y_ratio * i) - y1

                pixel = p11 * (1 - x_weight) * (1 - y_weight) + \
                p12 * x_weight * (1 - y_weight) + \
                p21 * y_weight * (1 - x_weight) + \
                p22 * x_weight * y_weight

                upsampled[i, j] = pixel.astype(np.uint8)
        return upsampled

    def get_psnr(self, original_image):
        mse = np.mean((original_image - self.image) ** 2)
        return 10 * math.log10(255.0**2 / mse)

    def get_image(self):
        return self.image

def main():
    PATH = 'motorcycles.png'
    FILENAME = PATH.split('/')[-1].split('.')[0]
    EXTENSION = PATH.split('.')[-1]
    original_image = cv2.imread(PATH, 1)

    encoder = Encoder(PATH)
    encoder.encode()
    downsampled_image = encoder.get_ycbcr()

    decoder = Decoder(downsampled_image, original_image.shape[:2])
    decoder.decode()
    final_image = decoder.get_image()

    psnr = decoder.get_psnr(original_image)
    print('PSNR:', psnr)
    cv2.imwrite(f"{FILENAME}_upsampled.{EXTENSION}", final_image)
    cv2.imshow('Y', downsampled_image[0])
    cv2.imshow('Cb', downsampled_image[1])
    cv2.imshow('Cr', downsampled_image[2])
    cv2.imshow('Original', original_image)
    cv2.imshow('Upsampled', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
