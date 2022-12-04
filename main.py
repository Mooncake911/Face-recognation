# //Developing a face recognition system\\
from pyfftw.interfaces.numpy_fft import hfft
from matplotlib import pyplot as plt
from scipy.fftpack import dct
import numpy as np
import random
import time
import cv2

"""Some global parameters"""
im_count = 1        # Count of image of each person
scale_percent = 20  # The percent of image (20% it is 20/100 = 1/5 of image)
his_bin = 32        # Histogram bin
psz = 4             # The count of division (4x4 = 16 psz)

# The sizes of image
img_size = cv2.imread('archive\\s1\\1.pgm', 0)
(height, width) = img_size.shape

p = 550             # The count of random points
pxl = [[random.randrange(0, height), random.randrange(0, width)] for i in range(p)]  # The array of p random point

# Outputting
start_time = time.time()
fig, ((test, ax1, ax2, ax3, ax4, ax5), (etalon, ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(nrows=2, ncols=6, figsize=(9, 6), num='Face_Recognition', subplot_kw={'xticks': [], 'yticks': []})
ax_list = [(ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)]
title_list = ("LBPH", "SCALE", "DCT", "FFT", "RANDOM_PXL")

correct = cv2.resize(cv2.imread('green.jpg', cv2.IMREAD_COLOR), (width, height))
incorrect = cv2.resize(cv2.imread('red.jpg', cv2.IMREAD_COLOR), (width, height))


class Person:
    def __init__(self, i, j):
        """Vector of methods for image"""
        self.img = cv2.imread('archive\\s%d\\%d.pgm' % (i, j), 0)
        self.name = "s%d" % i
        self.lbp_img = get_lbph(self.img)
        self.scale, self.FFT, self.DCT = get_scale(self.img, width, height)
        self.lbph = get_histogram(self.lbp_img)
        self.random_pxl, self.pxl_img = get_random_pxl(self.img)

        # Form for calculating
        self.get_from = [self.lbph, self.scale, self.DCT, self.FFT, self.random_pxl]
        # Form for outputting
        self.out_form = [self.lbp_img, self.scale, self.DCT, self.FFT, self.pxl_img, self.img]


def get_lbph(img):
    """Create lbp_image"""
    img_lbp = np.zeros((height, width), dtype=np.uint8)
    for i in range(2, height-2):    # Delete uninformative frame of image
        for j in range(2, width-2):
            if i % 2 == 0 and j % 10 != 0:
                img_lbp[i, j] = lbp_calculated_pixel(img, i, j)
            else:
                img_lbp[i, j] = img[i, j]
    return img_lbp


def get_scale(img, w, h):
    # Scale
    new_h = int(h * scale_percent / 100)
    new_w = int(w * scale_percent / 100)
    output = np.array(cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR), np.uint8)
    # FFT + Scale(as optimization)
    f = np.array(hfft(output).real, dtype=np.float64)
    f_shift = np.fft.fftshift(f)
    size = [6, 5]  # the best constants
    output1 = np.zeros((2*size[0], 2*size[1]), dtype=np.float64)
    """Take the most informative part of fft (center)"""
    a, b = 0, 0
    for i in range(len(f_shift) // 2 - size[0], len(f_shift) // 2 + size[0]):
        for j in range(len(f_shift[0]) // 2 - size[1], len(f_shift[0]) // 2 + size[1]):
            output1[a][b] = f_shift[i][j]
            b += 1
        a += 1
        b = 0
    # DCT + Scale(as optimization)
    f = np.array(dct(output, type=2, n=new_h - new_h % 10), dtype=np.float64)
    output2 = np.zeros((len(f), len(f[0] // 2)), dtype=np.float64)
    """Take the most informative part of dct (first columns)"""
    for i in range(len(f)):
        for j in range(len(f[0]) // 2):
            output2[i][j] = f[i][j]
    return output, output1, output2


def get_random_pxl(img):
    """Take random points"""
    rgb_img = np.array(cv2.cvtColor(img, 3))
    arr = np.zeros(p, dtype=np.uint8)
    for i in range(p):
        arr[i] = img[pxl[i][0]][pxl[i][1]]
        rgb_img[pxl[i][0]][pxl[i][1]] = (255, 255, 0)
    return arr, rgb_img


def get_histogram(img):
    """Cut image and reckon histogram for each psz"""
    psz_height = height // psz
    psz_width = width // psz
    box_list = []
    for x in range(0, psz):
        for y in range(0, psz):
            cropped = img[x * psz_width:(x + 1) * psz_height, y * psz_height:(y + 1) * psz_width]
            box_list.append(cv2.calcHist([cropped], [0], None, [his_bin], [0, 256]))
    return box_list


def lbp_calculated_pixel(img, x, y):
    """Local Binary Pattern"""
    center = img[x][y]

    def get_pixel(i, j):
        """
             232 | 128 |   1            1 |  1->|   0
            ----------------         ----------------
             32  | 128 |   2   ->       0 |   X |   0   ->  10000101  ->  X = 133
            ----------------         ----------------
             165 |   8 |   4            1 |   0 |   0
        """
        new_value = 0
        if img[i][j] >= center:
            new_value = 1
        return new_value

    val_ar = [get_pixel(x - 1, y + 1), get_pixel(x, y + 1), get_pixel(x + 1, y + 1), get_pixel(x + 1, y),
              get_pixel(x + 1, y - 1), get_pixel(x, y - 1), get_pixel(x - 1, y - 1), get_pixel(x - 1, y)]

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for t in range(len(val_ar)):
        val += val_ar[t] * power_val[t]
    return val


def search(arr_names):
    """Search the best name"""
    the_best_etalon = "NO"
    largest = -1
    for b in range(len(arr_names)):
        idx = 1
        for c in range(len(arr_names)):
            if arr_names[b] == arr_names[c] and b != c:
                match b:
                    case 0: idx += 79.4
                    case 1: idx += 74.4
                    case 2: idx += 74.4
                    case 3: idx += 74.1
                    case 4: idx += 73
        if idx > largest:
            largest = idx
            the_best_etalon = arr_names[b]
    return the_best_etalon


def make_empty():
    """Clean window"""
    for j in range(2):
        for i in range(len(ax_list[j])):
            ax_list[j][i].imshow(np.zeros((height, width), dtype=int), cmap='gray')
            if j:
                ax_list[j][i].set_title("")


def main():
    global data
    not_count = 0
    count = 0
    i = 1
    while i <= 40:
        j = im_count + 1
        while j <= 10:
            who = Person(i, j)
            form = who.get_from
            out = who.out_form

            '''test.imshow(who.img, cmap='gray')
            test.set_title("Test_img %s" % who.name)'''

            # Find the best et_name for each method
            make_empty()
            arr_names = []
            et_name = "NO"
            for t in range(4, 5):
                '''ax_list[0][t].imshow(out[t], cmap='gray')
                ax_list[0][t].set_title(title_list[t])'''

                min_dif = np.inf
                a = 0
                found_a = "False"
                while a < im_count * 40:
                    et_form = data[a].get_from
                    et_out = data[a].out_form

                    '''etalon.imshow(et_out[len(et_out) - 1], cmap='gray')
                    etalon.set_title("Etalon_img %s" % data[a].name)

                    ax_list[1][t].imshow(et_out[t], cmap='gray')
                    ax_list[1][t].set_title("Searching...")
                    plt.pause(0.01)'''

                    difference = np.sum(np.absolute(np.subtract(form[t], et_form[t], dtype=float)))
                    if difference < min_dif:
                        min_dif = difference
                        found_a = a
                        et_name = data[a].name

                    a += 1
                    """If we compare all images -> get result green(Right) or red(Not Right)"""
                    '''if a == im_count * 40:

                        if et_name == who.name:
                            mix = cv2.addWeighted(cv2.cvtColor(data[found_a].img, 3), 0.5, correct, 0.5, 0.0)
                            ax_list[1][t].imshow(mix)
                            ax_list[1][t].set_title("Found %s" % et_name)
                            plt.pause(0.1)
                        else:
                            mix = cv2.addWeighted(cv2.cvtColor(data[found_a].img, 3), 0.5, incorrect, 0.5, 0.0)
                            ax_list[1][t].imshow(mix)
                            ax_list[1][t].set_title("NOT Found")
                            plt.pause(0.1)'''

                arr_names.append(et_name)

            et_name = search(arr_names)
            if et_name == who.name:
                count += 1
            else:
                not_count += 1
            j += 1
        i += 1

    print("Were found %d == %f" % (count, 100 / (400 - 40 * im_count) * count) + "%")
    print("Were not found %d == %f" % (not_count, 100 / (400 - 40 * im_count) * not_count) + "%")

    '''plt.tight_layout()
    plt.show()'''


# Programme start
if __name__ == '__main__':
    data = [Person(i, j) for i in range(1, 41) for j in range(1, im_count + 1)]  # Base of etalon images
    main()
    print("%.2f seconds" % (time.time() - start_time))
