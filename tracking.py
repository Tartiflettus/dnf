import cv2
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import ndimage

import vanilla_dnf as dnf


# TODO : Pour installer opencv:
# sudo apt-get install opencv* python3-opencv

# Si vous avez des problèmes de performances
#
# self.kernel = np.zeros([width * 2, height * 2], dtype=float)
# for i in range(width * 2):
#     for j in range(height * 2):
#         d = np.sqrt(((i / (width * 2) - 0.5) ** 2 + ((j / (height * 2) - 0.5) ** 2))) / np.sqrt(0.5)
#         self.kernel[i, j] = self.difference_of_gaussian(d)
#
#
# Le tableau de poids latéreaux est calculé de la façon suivante :
# self.lateral = signal.fftconvolve(self.potentials, self.kernel, mode='same')

size = (96, 128)


def reduce(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def selectByColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([90, 0, 0])
    upper_blue = np.array([140, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = reduce(mask, size)

    return res


def findCenter(modele):
    # on fait évoluer le modele en espérant qu'il se focalisera sur la tasse
    modele.update_map()

    # on extrait le centre de la boule ainsi générée
    """index = np.argmax(modele.potentials)
    y = index // modele.potentials.shape[0]
    x = index % modele.potentials.shape[0]"""
    x, y = ndimage.center_of_mass(modele.potentials)
    x = int(x)
    y = int(y)

    return (x, y)


def motorControl(center):
    x, y = center
    center_x, center_y = size[0] // 2, size[1] // 2
    gradient_x, gradient_y = x - center_x, y - center_y

    if gradient_x > 10:
        gradient_x = 10
    if gradient_y > 10:
        gradient_y = 10
    if gradient_x < -10:
        gradient_x = -10
    if gradient_y < -10:
        gradient_y = -10
    ct.move(gradient_x, gradient_y)


def track(frame, modele):
    input = selectByColor(frame)
    # tester selectByColor
    modele.input = input
    modele.update_map()
    cv2.imshow("Input", modele.input)
    cv2.imshow("Potentials", modele.potentials)
    center = findCenter(modele)
    motorControl(center)


if __name__ == '__main__':
    cv2.namedWindow("Camera")
    vc = cv2.VideoCapture(0) # 2 pour la caméra sur moteur, 0 pour tester sur la votre.

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        input = selectByColor(frame)
        # initialisez votre modele ici
        modele = dnf.DNF(size[0], size[1])
    else:
        rval = False


    while rval:
        cv2.imshow("Camera", frame)
        rval, frame = vc.read()
        track(frame, modele)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyAllWindows()
