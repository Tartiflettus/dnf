# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import math
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import uniform

size = 35


def init_array():
    # return [[0. for j in range(size)] for i in range(size)]
    return [[uniform(0, 1) for i in range(size)] for j in range(size)]


def normalize(array):
    min_array = min(map(min, array))
    max_array = max(map(max, array))
    #print("min: ", min_array)
    #print("max: ", max_array)

    for y in range(size):
        for x in range(size):
            array[y][x] = (array[y][x] - min_array) / (max_array - min_array)

    for y in range(size):
        for x in range(size):
            if array[y][x] < 0 or array[y][x] > 1:
                print("normalize ERROR")


# normalized euclidean distance
def euclidean_dist(x, y):
    (x1, y1) = x
    (x2, y2) = y
    x1 /= size
    x2 /= size
    y1 /= size
    y2 /= size
    ans = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    """if ans > math.sqrt(2):
        print("euclidean_dist ERROR")"""
    return ans


# variables globales
cexc = 1.25
sigmaexc = 0.1
cinh = 0.7
sigmainh = 10.
array = init_array()
dt = 0.1
t = 0
tau = 0.64


def difference_of_gaussian(distance):
    return cexc*math.exp(-distance*distance/(2.*sigmaexc*sigmaexc))
    - cinh*math.exp(-distance*distance / (2.*sigmainh*sigmainh))


def gaussian_activity(a, b, sigma):
    activity = [[0 for j in range(size)] for i in range(size)]
    (x1, y1) = a
    (x2, y2) = b
    x1 /= size
    x2 /= size
    y1 /= size
    y2 /= size
    # first gaussian
    for y in range(size):
        yn = y / size
        for x in range(size):
            xn = x / size
            first_term = (xn-x1)*(xn-x1) / (2.*sigma*sigma)
            second_term = (yn-y1)*(yn-y1) / (2.*sigma*sigma)
            activity[y][x] = math.exp(-(first_term + second_term))

    # second gaussian
    for y in range(size):
        yn = y / size
        for x in range(size):
            xn = x / size
            first_term = (xn-x2)*(xn-x2) / (2.*sigma*sigma)
            second_term = (yn-y2)*(yn-y2) / (2.*sigma*sigma)
            activity[y][x] += math.exp(-(first_term + second_term))

    normalize(activity)
    return activity


entry = gaussian_activity((10., 10.), (25., 25.), 2.)


def update_neuron(x):
    (x, y) = x
    # excitation term
    exc_term = 0.
    for yi in range(size):
        for xi in range(size):
            if xi != x or yi != y:
                exc_term += array[yi][xi]*difference_of_gaussian( euclidean_dist((x, y), (xi, yi)) )
    return array[y][x] + dt*(-array[y][x] + exc_term + entry[y][x]) / tau

    # avoid outliers
    """if array[y][x] > 1:
        array[y][x] = 1.
    elif array[y][x] < 0:
        array[y][x] = 0."""


def synchronous_run():
    global array
    new_array = [[None for i in range(size)] for j in range(size)]
    for y in range(size):
        for x in range(size):
            new_array[y][x] = update_neuron((x, y))
    array = new_array
    normalize(array)


"""class Asynchronous_run(threading.Thread):
    def __init__(self, elems):
        super.__init__(self)
        self._elems = elems

    def run(self):"""



if __name__ == "__main__":
    array = init_array()
    entry = gaussian_activity((10, 10), (25, 25), 0.1)
    diff = [[difference_of_gaussian(euclidean_dist((17, 17), (x, y)))
             for x in range(size)] for y in range(size)]
    normalize(diff)
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("neurons")
    im = plt.imshow(array, cmap='hot', interpolation='nearest', animated=True)
    plt.subplot(1, 3, 2)
    plt.title("input")
    im_entry = plt.imshow(entry, cmap='hot', interpolation='nearest', animated=True)
    plt.subplot(1, 3, 3)
    plt.title("gaussian diff")
    im_diff = plt.imshow(diff, cmap='hot', interpolation='nearest', animated=True)

    index = 0


    def updatefig(*args):
        global array
        global entry
        global index
        synchronous_run()
        #array = init_array()
        im.set_array(array)
        im_entry.set_array(entry)
        im_diff.set_array(diff)
        print(index)
        index += 1

        return im, im_entry, im_diff
    
    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    plt.show()
