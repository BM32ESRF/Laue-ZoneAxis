"""
Permet de représenter l'experience en 3d.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_cam(plt3d):
    x = y = [-1, 1]
    xx, yy = np.meshgrid(x, y)
    z = np.vectorize((lambda x, y: 1))(xx, yy)
    plt3d.plot_surface(xx, yy, z, alpha=0.6, color="green")
    plt3d.text(0, 1, 1, "plan de la caméra")

def plot_sample(plt3d):
    x = y = [-.2, .2]
    xx, yy = np.meshgrid(x, y)
    z = np.vectorize((lambda x, y: x))(xx, yy)
    plt3d.plot_surface(xx, yy, z, alpha=0.6, color="blue")
    plt3d.text(0, 0, 0, "échantillon")

def plot_ui(plt3d):
    plt3d.plot(xs=[-1, 0], ys=[0, 0], zs=[0, 0], color="red")
    plt3d.text(-1, 0, 0, "rayon incident")

def plot_ref(plt3d):
    for x, y in np.random.uniform(-1, 1, size=(10, 2)):
        plt3d.plot(xs=[0, x], ys=[0, y], zs=[0, 1], color="orange")

def main():
    plt3d = plt.figure().gca(projection="3d")
    plot_cam(plt3d)
    plot_sample(plt3d)
    plot_ui(plt3d)
    plot_ref(plt3d)

    plt3d.set_xlabel('X axis')
    plt3d.set_ylabel('Y axis')
    plt3d.set_zlabel('Z axis')
    plt.show()


if __name__ == "__main__":
    main()
