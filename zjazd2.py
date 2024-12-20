import tensorflow as tf

import numpy as np

import sys

# Zad 1 i 2.

# dane x, y, kąt alpha (zmieniana na macierz obrotu), chcemy obliczyć x' i y'

def calculate_points_after_rotation(x, y, angle=90):

    angle_in_radians = angle * np.pi / 180

    rotation_matrix = tf.constant([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],

                                   [np.sin(angle_in_radians), np.cos(angle_in_radians)]])

    original_points = tf.constant([[x], [y]], dtype=tf.double)

    points_after_rotation = tf.matmul(rotation_matrix, original_points)

    print(points_after_rotation)

# zad 3 i 4

def solve_linear_system(params, results):

    if params.size != results.size * results.size:

        print("Invalid size")

        return 1

    params = params.reshape(results.size, results.size)

    results = results.reshape(results.size, 1)

    mtx = tf.constant(params)

    vec = tf.constant(results)

    try:

        solution = tf.linalg.solve(mtx, vec)

        print(solution)

    except:

        print("Cannot solve the linear system")

if __name__ == "__main__":

    params = sys.argv[1]

    results = sys.argv[2]

    calculate_points_after_rotation(2.0, 3.0, 90)

    params = np.array(params.split(",")).astype(np.double)

    results = np.array(results.split(",")).astype(np.double)

    # 2x + y = 4, 3x + y = 6

    solve_linear_system(params, results)
