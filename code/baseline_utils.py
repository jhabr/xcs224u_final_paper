import colorsys
import cmath as cm
from itertools import product

__authors__ = "Anton Gochev, Jaro Habr, Yan Jiang, Samuel Kahn"
__version__ = "XCS224u, Stanford, Spring 2021"


def fourier_transform(hsv_color):
    """
    Fourier transform on hsv colors.

    Parameters
    ----------
    hsv_color: list
        A list with h, s, v color values

    Returns
    -------
    transformed_color: list
        54 dimensional list with real and imaginary values

    """
    real = []
    imaginary = []

    for j, k, l in product((0, 1, 2), repeat=3):
        f_jkl = cm.exp(-2j * cm.pi * (j * hsv_color[0] + k * hsv_color[1] + l * hsv_color[2]))
        real.append(f_jkl.real)
        imaginary.append(f_jkl.imag)

    transformed_color = real + imaginary

    assert len(transformed_color) == 54  # 54 dimensions as in paper

    return transformed_color


def hls_to_hsv(hls_color):
    """
    Helper function that transforms a color form HLS color space into a HSV color space.

    Parameters
    ----------
    hls_color: list
        A list with h, l, s color values

    Returns
    -------
    list
        A list with transformed h, s, v values

    """
    h, l, s = hls_color
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    return [h, s, v]
