import cv2
import matplotlib.pyplot as plt
import numpy as np

from tkinter import filedialog
from tkinter import *


def get_file():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    return root.filename  # returns file path


def pr(im, prog, chid):
    height, width = im.shape[:2]
    res = np.ones([height, width, 3])
    for x in range(height):
        for y in range(width):
            if(im[x, y, chid]>=prog):
                res[x, y] = [1.0, 1.0, 1.0]
            else:
                res[x, y] = [0.0, 0.0, 0.0]
    return res


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_path = get_file()
    img1 = cv2.cvtColor(cv2.imread(image_path, 1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(image_path, 1), cv2.COLOR_BGR2HSV)

    im1_blue = np.array([0 for col in range(256)])
    im1_red = np.array([0 for col in range(256)])
    im1_green = np.array([0 for col in range(256)])
    height1, width1 = img1.shape[:2]
    for x in range(height1):
        for y in range(width1):
            im1_red[img1[x, y, 0]] += 1
            im1_green[img1[x, y, 1]] += 1
            im1_blue[img1[x, y, 2]] += 1

    coords = []
    f = plt.figure()
    f.add_subplot(121)
    plt.imshow(img1)
    f.add_subplot(122)
    plt.plot(im1_red, color='red')
    plt.plot(im1_blue, color='blue')
    plt.plot(im1_green, color='green')

    def onclick(event):
        ix, iy = int(event.xdata), int(event.ydata)
        print('x = %d, y = %d' % (ix, iy))

        print(img1[iy, ix])
        lower = np.array(img2[iy, ix]-[20, 50, 50])
        upper = np.array(img2[iy, ix]+[20, 50, 50])
        print(upper)
        print(lower)

        color = np.array([[img1[iy, ix] for y in range(500)] for x in range(500)])

        mask = cv2.inRange(img2, lower, upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img2, img2, mask=mask)
        res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
        fig = plt.figure()

        fig.add_subplot(121)
        plt.imshow(color)
        fig.add_subplot(122)
        plt.imshow(res)

        plt.show()
        if len(coords) == 2:
            f.canvas.mpl_disconnect(cid)

    cid = f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()