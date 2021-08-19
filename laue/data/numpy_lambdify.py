#!/usr/bin/env python3

"""
This code was automatically generated.
"""

import numpy; from numpy import *; from numpy.linalg import *


def _cam_to_gnomonic_numpy(x_cam, y_cam, dd, xcen, ycen, beta, gamma, pixelsize):
    """Perform calculations in small float using the numpy module."""
    x0 = 0.0174532925199433*beta
    x1 = cos(x0)
    x2 = dd*x1
    x3 = sin(x0)
    del x0
    x4 = x_cam - xcen
    x5 = 0.0174532925199433*gamma
    x6 = sin(x5)
    x7 = x4*x6
    x8 = cos(x5)
    del x5
    x9 = y_cam - ycen
    x10 = x8*x9
    x11 = pixelsize*(-x10*x3 + x3*x7)
    x12 = x11 + x2
    x13 = x4*x8 + x6*x9
    del x9, x8, x6, x4
    x14 = dd*x3 - pixelsize*(-x1*x10 + x1*x7)
    del x7, x3, x10, x1
    x15 = x14 - sqrt(pixelsize**2*x13**2 + x12**2 + x14**2)
    del x14
    x16 = (-x11 + x15 - x2)**(-1.0)
    del x2, x11
    _0 = -x16*(x12 + x15)
    del x15, x12
    _1 = 1.4142135623731*pixelsize*x13*x16
    return [_0, _1]

def _gnomonic_to_cam_numpy(x_gnom, y_gnom, dd, xcen, ycen, beta, gamma, pixelsize):
    """Perform calculations in small float using the numpy module."""
    x0 = 0.0174532925199433*gamma
    x1 = dd*cos(x0)
    x2 = 0.707106781186548*x_gnom
    x3 = 2*x2 - 1.4142135623731
    x4 = x3*y_gnom
    x5 = 0.0174532925199433*beta
    x6 = cos(x5)
    x7 = x3*(x2 + 0.707106781186548)
    del x3, x2
    x8 = x6*x7
    x9 = pixelsize*x8
    x10 = dd*sin(x0)
    del x0
    x11 = sin(x5)
    del x5
    x12 = x11*x7
    del x7
    x13 = y_gnom**2 - 0.5*(x_gnom - 1)**2 + 0.5*(x_gnom + 1)**2
    x14 = x11*x13
    del x11
    x15 = pixelsize*x14
    x16 = x13*x6
    del x6, x13
    x17 = 1/(pixelsize*(x14 - x8))
    del x8, x14
    _0 = x17*(x1*x4 - x10*x12 - x10*x16 + x15*xcen - x9*xcen)
    _1 = x17*(x1*x12 + x1*x16 + x10*x4 + x15*ycen - x9*ycen)
    return [_0, _1]

def _cam_to_thetachi_numpy(x_cam, y_cam, dd, xcen, ycen, beta, gamma, pixelsize):
    """Perform calculations in small float using the numpy module."""
    x0 = 0.0174532925199433*beta
    x1 = sin(x0)
    x2 = cos(x0)
    del x0
    x3 = x_cam - xcen
    x4 = 0.0174532925199433*gamma
    x5 = sin(x4)
    x6 = x3*x5
    x7 = cos(x4)
    del x4
    x8 = y_cam - ycen
    x9 = x7*x8
    x10 = dd*x1 - pixelsize*(x2*x6 - x2*x9)
    x11 = x3*x7 + x5*x8
    del x8, x7, x5, x3
    x12 = pixelsize**2*x11**2 + (dd*x2 + pixelsize*(x1*x6 - x1*x9))**2
    del x9, x6, x2, x1
    _0 = 28.6478897565412*arccos(x10/sqrt(x10**2 + x12))
    del x10
    _1 = -57.2957795130823*arcsin(pixelsize*x11/sqrt(x12))
    return [_0, _1]

def _thetachi_to_cam_numpy(theta, chi, dd, xcen, ycen, beta, gamma, pixelsize):
    """Perform calculations in small float using the numpy module."""
    x0 = 0.0174532925199433*beta
    x1 = sin(x0)
    x2 = 0.0349065850398866*theta
    x3 = cos(x2)
    x4 = pixelsize*x1*x3
    x5 = cos(x0)
    del x0
    x6 = 0.0174532925199433*chi
    x7 = sin(x2)
    del x2
    x8 = x7*cos(x6)
    x9 = pixelsize*x5*x8
    x10 = (x4 + x9)**(-1.0)
    x11 = 0.0174532925199433*gamma
    x12 = dd*cos(x11)
    x13 = x7*sin(x6)
    del x7, x6
    x14 = dd*sin(x11)
    del x11
    x15 = x3*x5
    del x5, x3
    x16 = x1*x8
    del x8, x1
    _0 = x10*(-x12*x13 - x14*x15 + x14*x16 + x4*xcen + x9*xcen)
    _1 = x10*(x12*x15 - x12*x16 - x13*x14 + x4*ycen + x9*ycen)
    return [_0, _1]

def _thetachi_to_gnomonic_numpy(theta, chi):
    """Perform calculations in small float using the numpy module."""
    x0 = 0.0349065850398866*theta
    x1 = cos(x0)
    x2 = sin(x0)
    del x0
    x3 = 0.0174532925199433*chi
    x4 = x2*cos(x3)
    x5 = (-x1 + x4 + 1)**(-1.0)
    _0 = x5*(x1 + x4 - 1)
    del x4, x1
    _1 = 1.4142135623731*x2*x5*sin(x3)
    return [_0, _1]

def _gnomonic_to_thetachi_numpy(x_gnom, y_gnom):
    """Perform calculations in small float using the numpy module."""
    x0 = y_gnom**2
    x1 = x_gnom - 1
    _1 = -57.2957795130823*arcsin(x1*y_gnom/(sqrt(x0 + 0.5*(x_gnom + 1)**2)*abs(x1)))
    del x1
    _0 = 28.6478897565412*arccos((x0 + 2*x_gnom)/(x0 + x_gnom**2 + 1))
    return [_0, _1]

def _dist_cosine_numpy(theta_1, chi_1, theta_2, chi_2):
    """Perform calculations in small float using the numpy module."""
    x0 = 0.0349065850398866*theta_1
    x1 = cos(x0)
    x2 = 1 - x1
    x3 = 0.0349065850398866*theta_2
    x4 = cos(x3)
    x5 = -x4
    _0 = 57.2957795130823*arccos(0.5*(x1*x4 + x2 + x5 + sin(x0)*sin(x3)*cos(0.0174532925199433*chi_1 - 0.0174532925199433*chi_2))/(sqrt(x2)*sqrt(x5 + 1)))
    return _0

def _dist_euclidian_numpy(x1, y1, x2, y2):
    """Perform calculations in small float using the numpy module."""
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def _dist_line_numpy(phi, mu, x, y):
    """Perform calculations in small float using the numpy module."""
    x0 = 0.25*x**2
    x1 = 0.25*y**2
    x2 = 2*phi
    x3 = cos(x2)
    _0 = 1.4142135623731*sqrt(0.5*mu**2 - mu*x*cos(phi) - mu*y*sin(phi) + 0.5*x*y*sin(x2) + x0*x3 + x0 - x1*x3 + x1)
    return _0

def _hough_numpy(x_a, y_a, x_b, y_b):
    """Perform calculations in small float using the numpy module."""
    x0 = x_a - x_b
    x1 = y_a - y_b
    x2 = x0**2 + x1**2
    x3 = x0*x_a + x1*y_a
    x4 = -x0*x3 + x2*x_a
    del x0
    x5 = -x1*x3 + x2*y_a
    del x3, x2, x1
    x6 = 1/sqrt(x4**2 + x5**2)
    x7 = x_a*x_b
    x8 = y_a*y_b
    x9 = x_a**2
    x10 = x_b**2
    x11 = y_a**2
    x12 = y_b**2
    _0 = arccos(x4*x6)*sign(x5*x6)
    del x6, x5, x4
    _1 = 1.4142135623731*sqrt(0.5*x10 + 0.5*x11 + 0.5*x12 - x7 - x8 + 0.5*x9)*abs((x_a*y_b - x_b*y_a)/(x10 + x11 + x12 - 2*x7 - 2*x8 + x9))
    return [_0, _1]

def _inter_line_numpy(phi_1, mu_1, phi_2, mu_2):
    """Perform calculations in small float using the numpy module."""
    x0 = sin(phi_1 - phi_2)**(-1.0)
    _0 = x0*(-mu_1*sin(phi_2) + mu_2*sin(phi_1))
    _1 = x0*(mu_1*cos(phi_2) - mu_2*cos(phi_1))
    return [_0, _1]

