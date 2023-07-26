import numpy
import matplotlib.pyplot as plt
import cv2
import data



def binData(SAMPLES):
    images, masks = data.load_data(SAMPLES)
    bin_img, bin_msk = [], []

    for gray_img in images:
        scaled_img = numpy.interp(gray_img, (-1, 1), (0, 255))

        # Convert the scaled image to 8-bit unsigned integer format
        uint8_img = numpy.uint8(scaled_img)

        # Threshold the image to get a binary mask
        threshold_value = 75
        max_value = 255
        _, binary_mask = cv2.threshold(uint8_img, threshold_value, max_value, cv2.THRESH_BINARY)

        # Scale the binary mask back to range from -1 to 1
        scaled_binary_mask = numpy.interp(binary_mask, (0, 255), (-1, 1))

        bin_img.append(scaled_binary_mask)

    for gray_img in masks:
        scaled_img = numpy.interp(gray_img, (-1, 1), (0, 255))

        # Convert the scaled image to 8-bit unsigned integer format
        uint8_img = numpy.uint8(scaled_img)

        # Threshold the image to get a binary mask
        threshold_value = 75
        max_value = 255
        _, binary_mask = cv2.threshold(uint8_img, threshold_value, max_value, cv2.THRESH_BINARY)

        # Scale the binary mask back to range from -1 to 1
        scaled_binary_mask = numpy.interp(binary_mask, (0, 255), (-1, 1))

        bin_msk.append(scaled_binary_mask)

    bin_img = numpy.asarray(bin_img)
    numpy.save('BINData.npy', bin_img)
    bin_msk = numpy.asarray(bin_msk)
    numpy.save('BINMasks.npy', bin_msk)


if __name__ == '__main__':
    binData(300)
