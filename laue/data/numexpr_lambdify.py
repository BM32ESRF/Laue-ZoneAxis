#!/usr/bin/env python3

"""
This code was automatically generated.
"""

from numexpr import evaluate


def _cam_to_gnomonic_numexpr(x_cam, y_cam, dd, xcen, ycen, beta, gamma, pixelsize):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('0.0174532925199433*beta', truediv=True)
    x1 = evaluate('cos(x0)', truediv=True)
    x2 = evaluate('dd*x1', truediv=True)
    x3 = evaluate('sin(x0)', truediv=True)
    del x0
    x4 = evaluate('x_cam - xcen', truediv=True)
    x5 = evaluate('0.0174532925199433*gamma', truediv=True)
    x6 = evaluate('sin(x5)', truediv=True)
    x7 = evaluate('x4*x6', truediv=True)
    x8 = evaluate('cos(x5)', truediv=True)
    del x5
    x9 = evaluate('y_cam - ycen', truediv=True)
    x10 = evaluate('x8*x9', truediv=True)
    x11 = evaluate('pixelsize*(-x10*x3 + x3*x7)', truediv=True)
    x12 = evaluate('x11 + x2', truediv=True)
    x13 = evaluate('x4*x8 + x6*x9', truediv=True)
    del x9, x8, x6, x4
    x14 = evaluate('dd*x3 - pixelsize*(-x1*x10 + x1*x7)', truediv=True)
    del x7, x3, x10, x1
    x15 = evaluate('x14 - (pixelsize**2*x13**2 + x12**2 + x14**2)**0.5', truediv=True)
    del x14
    x16 = evaluate('1/(-x11 + x15 - x2)', truediv=True)
    del x2, x11
    _0 = evaluate('-x16*(x12 + x15)', truediv=True)
    del x15, x12
    _1 = evaluate('1.4142135623731*pixelsize*x13*x16', truediv=True)
    return [_0, _1]

def _gnomonic_to_cam_numexpr(x_gnom, y_gnom, dd, xcen, ycen, beta, gamma, pixelsize):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('0.0174532925199433*gamma', truediv=True)
    x1 = evaluate('dd*cos(x0)', truediv=True)
    x2 = evaluate('0.707106781186548*x_gnom', truediv=True)
    x3 = evaluate('2.0*x2 - 1.4142135623731', truediv=True)
    x4 = evaluate('x3*y_gnom', truediv=True)
    x5 = evaluate('0.0174532925199433*beta', truediv=True)
    x6 = evaluate('cos(x5)', truediv=True)
    x7 = evaluate('x3*(x2 + 0.707106781186548)', truediv=True)
    del x3, x2
    x8 = evaluate('x6*x7', truediv=True)
    x9 = evaluate('pixelsize*x8', truediv=True)
    x10 = evaluate('dd*sin(x0)', truediv=True)
    del x0
    x11 = evaluate('sin(x5)', truediv=True)
    del x5
    x12 = evaluate('x11*x7', truediv=True)
    del x7
    x13 = evaluate('y_gnom**2 - 0.5*(x_gnom - 1.0)**2 + 0.5*(x_gnom + 1.0)**2', truediv=True)
    x14 = evaluate('x11*x13', truediv=True)
    del x11
    x15 = evaluate('pixelsize*x14', truediv=True)
    x16 = evaluate('x13*x6', truediv=True)
    del x6, x13
    x17 = evaluate('1/(pixelsize*(x14 - x8))', truediv=True)
    del x8, x14
    _0 = evaluate('x17*(x1*x4 - x10*x12 - x10*x16 + x15*xcen - x9*xcen)', truediv=True)
    _1 = evaluate('x17*(x1*x12 + x1*x16 + x10*x4 + x15*ycen - x9*ycen)', truediv=True)
    return [_0, _1]

def _cam_to_thetachi_numexpr(x_cam, y_cam, dd, xcen, ycen, beta, gamma, pixelsize):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('0.0174532925199433*beta', truediv=True)
    x1 = evaluate('sin(x0)', truediv=True)
    x2 = evaluate('cos(x0)', truediv=True)
    del x0
    x3 = evaluate('x_cam - xcen', truediv=True)
    x4 = evaluate('0.0174532925199433*gamma', truediv=True)
    x5 = evaluate('sin(x4)', truediv=True)
    x6 = evaluate('x3*x5', truediv=True)
    x7 = evaluate('cos(x4)', truediv=True)
    del x4
    x8 = evaluate('y_cam - ycen', truediv=True)
    x9 = evaluate('x7*x8', truediv=True)
    x10 = evaluate('dd*x1 - pixelsize*(x2*x6 - x2*x9)', truediv=True)
    x11 = evaluate('x3*x7 + x5*x8', truediv=True)
    del x8, x7, x5, x3
    x12 = evaluate('pixelsize**2*x11**2 + (dd*x2 + pixelsize*(x1*x6 - x1*x9))**2', truediv=True)
    del x9, x6, x2, x1
    _0 = evaluate('28.6478897565412*arccos(x10*(x10**2 + x12)**(-0.5))', truediv=True)
    del x10
    _1 = evaluate('-57.2957795130823*arcsin(pixelsize*x11*x12**(-0.5))', truediv=True)
    return [_0, _1]

def _thetachi_to_cam_numexpr(theta, chi, dd, xcen, ycen, beta, gamma, pixelsize):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('0.0174532925199433*beta', truediv=True)
    x1 = evaluate('sin(x0)', truediv=True)
    x2 = evaluate('0.0349065850398866*theta', truediv=True)
    x3 = evaluate('cos(x2)', truediv=True)
    x4 = evaluate('pixelsize*x1*x3', truediv=True)
    x5 = evaluate('cos(x0)', truediv=True)
    del x0
    x6 = evaluate('0.0174532925199433*chi', truediv=True)
    x7 = evaluate('sin(x2)', truediv=True)
    del x2
    x8 = evaluate('x7*cos(x6)', truediv=True)
    x9 = evaluate('pixelsize*x5*x8', truediv=True)
    x10 = evaluate('1/(x4 + x9)', truediv=True)
    x11 = evaluate('0.0174532925199433*gamma', truediv=True)
    x12 = evaluate('dd*cos(x11)', truediv=True)
    x13 = evaluate('x7*sin(x6)', truediv=True)
    del x7, x6
    x14 = evaluate('dd*sin(x11)', truediv=True)
    del x11
    x15 = evaluate('x3*x5', truediv=True)
    del x5, x3
    x16 = evaluate('x1*x8', truediv=True)
    del x8, x1
    _0 = evaluate('x10*(-x12*x13 - x14*x15 + x14*x16 + x4*xcen + x9*xcen)', truediv=True)
    _1 = evaluate('x10*(x12*x15 - x12*x16 - x13*x14 + x4*ycen + x9*ycen)', truediv=True)
    return [_0, _1]

def _thetachi_to_gnomonic_numexpr(theta, chi):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('0.0349065850398866*theta', truediv=True)
    x1 = evaluate('cos(x0)', truediv=True)
    x2 = evaluate('sin(x0)', truediv=True)
    del x0
    x3 = evaluate('0.0174532925199433*chi', truediv=True)
    x4 = evaluate('x2*cos(x3)', truediv=True)
    x5 = evaluate('1/(-x1 + x4 + 1.0)', truediv=True)
    _0 = evaluate('x5*(x1 + x4 - 1.0)', truediv=True)
    del x4, x1
    _1 = evaluate('1.4142135623731*x2*x5*sin(x3)', truediv=True)
    return [_0, _1]

def _gnomonic_to_thetachi_numexpr(x_gnom, y_gnom):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('y_gnom**2', truediv=True)
    x1 = evaluate('x_gnom - 1.0', truediv=True)
    _1 = evaluate('-57.2957795130823*arcsin(1.0*x1*y_gnom*(x0 + 0.5*(x_gnom + 1.0)**2)**(-0.5)/abs(x1))', truediv=True)
    del x1
    _0 = evaluate('28.6478897565412*arccos((x0 + 2.0*x_gnom)/(x0 + x_gnom**2 + 1.0))', truediv=True)
    return [_0, _1]

def _dist_cosine_numexpr(theta_1, chi_1, theta_2, chi_2):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('0.0349065850398866*theta_1', truediv=True)
    x1 = evaluate('cos(x0)', truediv=True)
    x2 = evaluate('1.0 - x1', truediv=True)
    x3 = evaluate('0.0349065850398866*theta_2', truediv=True)
    x4 = evaluate('cos(x3)', truediv=True)
    x5 = evaluate('-x4', truediv=True)
    _0 = evaluate('57.2957795130823*arccos(0.5*x2**(-0.5)*(x5 + 1.0)**(-0.5)*(x1*x4 + x2 + x5 + sin(x0)*sin(x3)*cos(0.0174532925199433*chi_1 - 0.0174532925199433*chi_2)))', truediv=True)
    return _0

def _dist_euclidian_numexpr(x1, y1, x2, y2):
    """Perform calculations in float64 using the numexpr module."""
    return evaluate('((x1 - x2)**2 + (y1 - y2)**2)**0.5', truediv=True)

def _dist_line_numexpr(phi, mu, x, y):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('0.25*x**2', truediv=True)
    x1 = evaluate('0.25*y**2', truediv=True)
    x2 = evaluate('2.0*phi', truediv=True)
    x3 = evaluate('cos(x2)', truediv=True)
    _0 = evaluate('1.4142135623731*(0.5*mu**2 - mu*x*cos(phi) - mu*y*sin(phi) + 0.5*x*y*sin(x2) + x0*x3 + x0 - x1*x3 + x1)**0.5', truediv=True)
    return _0

def _hough_numexpr(x_a, y_a, x_b, y_b):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('x_a - x_b', truediv=True)
    x1 = evaluate('y_a - y_b', truediv=True)
    x2 = evaluate('x0**2 + x1**2', truediv=True)
    x3 = evaluate('x0*x_a + x1*y_a', truediv=True)
    x4 = evaluate('-x0*x3 + x2*x_a', truediv=True)
    del x0
    x5 = evaluate('-x1*x3 + x2*y_a', truediv=True)
    del x3, x2, x1
    x6 = evaluate('(x4**2 + x5**2)**(-0.5)', truediv=True)
    x7 = evaluate('x_a*x_b', truediv=True)
    x8 = evaluate('y_a*y_b', truediv=True)
    x9 = evaluate('x_a**2', truediv=True)
    x10 = evaluate('x_b**2', truediv=True)
    x11 = evaluate('y_a**2', truediv=True)
    x12 = evaluate('y_b**2', truediv=True)
    _0 = evaluate('arccos(x4*x6)*(0.0 if x5*x6 == 0 else copysign(1, x5*x6))', truediv=True)
    del x6, x5, x4
    _1 = evaluate('1.4142135623731*(0.5*x10 + 0.5*x11 + 0.5*x12 - x7 - x8 + 0.5*x9)**0.5*abs((x_a*y_b - x_b*y_a)/(x10 + x11 + x12 - 2.0*x7 - 2.0*x8 + x9))', truediv=True)
    return [_0, _1]

def _inter_line_numexpr(phi_1, mu_1, phi_2, mu_2):
    """Perform calculations in float64 using the numexpr module."""
    x0 = evaluate('1/sin(phi_1 - phi_2)', truediv=True)
    _0 = evaluate('x0*(-mu_1*sin(phi_2) + mu_2*sin(phi_1))', truediv=True)
    _1 = evaluate('x0*(mu_1*cos(phi_2) - mu_2*cos(phi_1))', truediv=True)
    return [_0, _1]

