from PIL import Image
import pandas
import numpy


def load_data(n_samples):

    info = pandas.read_csv('./info_data.csv')
    info = info[info['shape'].str.contains(r"(512, 512)")]

    def imgArr(files):
        images = list()
        for file in files:
            im = Image.open(file).convert('L')
            arr = numpy.asarray(im)
            images.append(arr)
        return images

    images = (numpy.array(imgArr(list(info['fileLocation'].head(n_samples)))) - 127.5) / 127.5
    mask = (numpy.array(imgArr(list(info['maskLocation'].head(n_samples)))) - 127.5) / 127.5

    return (images, mask)
