#!/usr/bin/env python3

"""
This code was automatically generated.
"""

from sympy.functions import *
from sympy.matrices import *
from sympy import Integral, pi, oo, nan, zoo, E, I
from sympy import symbols


def _cam_to_gnomonic_sympy():
    """Returns the tree of the sympy expression."""
    x_cam, y_cam, dd, xcen, ycen, beta, gamma, pixelsize = symbols('x_cam y_cam dd xcen ycen beta gamma pixelsize')
    return [-(dd*sin(pi*beta/180) + dd*cos(pi*beta/180) + pixelsize*((x_cam - xcen)*sin(pi*beta/180)*sin(pi*gamma/180) - (y_cam - ycen)*sin(pi*beta/180)*cos(pi*gamma/180)) - pixelsize*((x_cam - xcen)*sin(pi*gamma/180)*cos(pi*beta/180) - (y_cam - ycen)*cos(pi*beta/180)*cos(pi*gamma/180)) - sqrt(pixelsize**2*((x_cam - xcen)*cos(pi*gamma/180) + (y_cam - ycen)*sin(pi*gamma/180))**2 + (dd*sin(pi*beta/180) - pixelsize*((x_cam - xcen)*sin(pi*gamma/180)*cos(pi*beta/180) - (y_cam - ycen)*cos(pi*beta/180)*cos(pi*gamma/180)))**2 + (dd*cos(pi*beta/180) + pixelsize*((x_cam - xcen)*sin(pi*beta/180)*sin(pi*gamma/180) - (y_cam - ycen)*sin(pi*beta/180)*cos(pi*gamma/180)))**2))/(dd*sin(pi*beta/180) - dd*cos(pi*beta/180) - pixelsize*((x_cam - xcen)*sin(pi*beta/180)*sin(pi*gamma/180) - (y_cam - ycen)*sin(pi*beta/180)*cos(pi*gamma/180)) - pixelsize*((x_cam - xcen)*sin(pi*gamma/180)*cos(pi*beta/180) - (y_cam - ycen)*cos(pi*beta/180)*cos(pi*gamma/180)) - sqrt(pixelsize**2*((x_cam - xcen)*cos(pi*gamma/180) + (y_cam - ycen)*sin(pi*gamma/180))**2 + (dd*sin(pi*beta/180) - pixelsize*((x_cam - xcen)*sin(pi*gamma/180)*cos(pi*beta/180) - (y_cam - ycen)*cos(pi*beta/180)*cos(pi*gamma/180)))**2 + (dd*cos(pi*beta/180) + pixelsize*((x_cam - xcen)*sin(pi*beta/180)*sin(pi*gamma/180) - (y_cam - ycen)*sin(pi*beta/180)*cos(pi*gamma/180)))**2)), sqrt(2)*pixelsize*((x_cam - xcen)*cos(pi*gamma/180) + (y_cam - ycen)*sin(pi*gamma/180))/(dd*sin(pi*beta/180) - dd*cos(pi*beta/180) - pixelsize*((x_cam - xcen)*sin(pi*beta/180)*sin(pi*gamma/180) - (y_cam - ycen)*sin(pi*beta/180)*cos(pi*gamma/180)) - pixelsize*((x_cam - xcen)*sin(pi*gamma/180)*cos(pi*beta/180) - (y_cam - ycen)*cos(pi*beta/180)*cos(pi*gamma/180)) - sqrt(pixelsize**2*((x_cam - xcen)*cos(pi*gamma/180) + (y_cam - ycen)*sin(pi*gamma/180))**2 + (dd*sin(pi*beta/180) - pixelsize*((x_cam - xcen)*sin(pi*gamma/180)*cos(pi*beta/180) - (y_cam - ycen)*cos(pi*beta/180)*cos(pi*gamma/180)))**2 + (dd*cos(pi*beta/180) + pixelsize*((x_cam - xcen)*sin(pi*beta/180)*sin(pi*gamma/180) - (y_cam - ycen)*sin(pi*beta/180)*cos(pi*gamma/180)))**2))]

def _gnomonic_to_cam_sympy():
    """Returns the tree of the sympy expression."""
    x_gnom, y_gnom, dd, xcen, ycen, beta, gamma, pixelsize = symbols('x_gnom y_gnom dd xcen ycen beta gamma pixelsize')
    return [(2*dd*y_gnom*(sqrt(2)*x_gnom/2 - sqrt(2)/2)*cos(pi*gamma/180) - 2*dd*(sqrt(2)*x_gnom/2 - sqrt(2)/2)*(sqrt(2)*x_gnom/2 + sqrt(2)/2)*sin(pi*beta/180)*sin(pi*gamma/180) - dd*(y_gnom**2 - (sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + (sqrt(2)*x_gnom/2 + sqrt(2)/2)**2)*sin(pi*gamma/180)*cos(pi*beta/180) - 2*pixelsize*xcen*(sqrt(2)*x_gnom/2 - sqrt(2)/2)*(sqrt(2)*x_gnom/2 + sqrt(2)/2)*cos(pi*beta/180) + pixelsize*xcen*(y_gnom**2 - (sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + (sqrt(2)*x_gnom/2 + sqrt(2)/2)**2)*sin(pi*beta/180))/(pixelsize*(-2*(sqrt(2)*x_gnom/2 - sqrt(2)/2)*(sqrt(2)*x_gnom/2 + sqrt(2)/2)*cos(pi*beta/180) + (y_gnom**2 - (sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + (sqrt(2)*x_gnom/2 + sqrt(2)/2)**2)*sin(pi*beta/180))), (2*dd*y_gnom*(sqrt(2)*x_gnom/2 - sqrt(2)/2)*sin(pi*gamma/180) + 2*dd*(sqrt(2)*x_gnom/2 - sqrt(2)/2)*(sqrt(2)*x_gnom/2 + sqrt(2)/2)*sin(pi*beta/180)*cos(pi*gamma/180) + dd*(y_gnom**2 - (sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + (sqrt(2)*x_gnom/2 + sqrt(2)/2)**2)*cos(pi*beta/180)*cos(pi*gamma/180) - 2*pixelsize*ycen*(sqrt(2)*x_gnom/2 - sqrt(2)/2)*(sqrt(2)*x_gnom/2 + sqrt(2)/2)*cos(pi*beta/180) + pixelsize*ycen*(y_gnom**2 - (sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + (sqrt(2)*x_gnom/2 + sqrt(2)/2)**2)*sin(pi*beta/180))/(pixelsize*(-2*(sqrt(2)*x_gnom/2 - sqrt(2)/2)*(sqrt(2)*x_gnom/2 + sqrt(2)/2)*cos(pi*beta/180) + (y_gnom**2 - (sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + (sqrt(2)*x_gnom/2 + sqrt(2)/2)**2)*sin(pi*beta/180)))]

def _cam_to_thetachi_sympy():
    """Returns the tree of the sympy expression."""
    x_cam, y_cam, dd, xcen, ycen, beta, gamma, pixelsize = symbols('x_cam y_cam dd xcen ycen beta gamma pixelsize')
    return [28.6478897565412*acos((dd*sin(pi*beta/180) - pixelsize*((x_cam - xcen)*sin(pi*gamma/180)*cos(pi*beta/180) - (y_cam - ycen)*cos(pi*beta/180)*cos(pi*gamma/180)))/sqrt(pixelsize**2*((x_cam - xcen)*cos(pi*gamma/180) + (y_cam - ycen)*sin(pi*gamma/180))**2 + (dd*sin(pi*beta/180) - pixelsize*((x_cam - xcen)*sin(pi*gamma/180)*cos(pi*beta/180) - (y_cam - ycen)*cos(pi*beta/180)*cos(pi*gamma/180)))**2 + (dd*cos(pi*beta/180) + pixelsize*((x_cam - xcen)*sin(pi*beta/180)*sin(pi*gamma/180) - (y_cam - ycen)*sin(pi*beta/180)*cos(pi*gamma/180)))**2)), -57.2957795130823*asin(pixelsize*((x_cam - xcen)*cos(pi*gamma/180) + (y_cam - ycen)*sin(pi*gamma/180))/sqrt(pixelsize**2*((x_cam - xcen)*cos(pi*gamma/180) + (y_cam - ycen)*sin(pi*gamma/180))**2 + (dd*cos(pi*beta/180) + pixelsize*((x_cam - xcen)*sin(pi*beta/180)*sin(pi*gamma/180) - (y_cam - ycen)*sin(pi*beta/180)*cos(pi*gamma/180)))**2))]

def _thetachi_to_cam_sympy():
    """Returns the tree of the sympy expression."""
    theta, chi, dd, xcen, ycen, beta, gamma, pixelsize = symbols('theta chi dd xcen ycen beta gamma pixelsize')
    return [(-dd*sin(0.0174532925199433*chi)*sin(0.0349065850398866*theta)*cos(pi*gamma/180) + dd*sin(0.0349065850398866*theta)*sin(pi*beta/180)*sin(pi*gamma/180)*cos(0.0174532925199433*chi) - dd*sin(pi*gamma/180)*cos(0.0349065850398866*theta)*cos(pi*beta/180) + pixelsize*xcen*sin(0.0349065850398866*theta)*cos(0.0174532925199433*chi)*cos(pi*beta/180) + pixelsize*xcen*sin(pi*beta/180)*cos(0.0349065850398866*theta))/(pixelsize*(sin(0.0349065850398866*theta)*cos(0.0174532925199433*chi)*cos(pi*beta/180) + sin(pi*beta/180)*cos(0.0349065850398866*theta))), (-dd*sin(0.0174532925199433*chi)*sin(0.0349065850398866*theta)*sin(pi*gamma/180) - dd*sin(0.0349065850398866*theta)*sin(pi*beta/180)*cos(0.0174532925199433*chi)*cos(pi*gamma/180) + dd*cos(0.0349065850398866*theta)*cos(pi*beta/180)*cos(pi*gamma/180) + pixelsize*ycen*sin(0.0349065850398866*theta)*cos(0.0174532925199433*chi)*cos(pi*beta/180) + pixelsize*ycen*sin(pi*beta/180)*cos(0.0349065850398866*theta))/(pixelsize*(sin(0.0349065850398866*theta)*cos(0.0174532925199433*chi)*cos(pi*beta/180) + sin(pi*beta/180)*cos(0.0349065850398866*theta)))]

def _thetachi_to_gnomonic_sympy():
    """Returns the tree of the sympy expression."""
    theta, chi = symbols('theta chi')
    return [-(-sqrt(sin(0.0174532925199433*chi)**2*sin(0.0349065850398866*theta)**2 + sin(0.0349065850398866*theta)**2*cos(0.0174532925199433*chi)**2 + cos(0.0349065850398866*theta)**2) + sin(0.0349065850398866*theta)*cos(0.0174532925199433*chi) + cos(0.0349065850398866*theta))/(-sqrt(sin(0.0174532925199433*chi)**2*sin(0.0349065850398866*theta)**2 + sin(0.0349065850398866*theta)**2*cos(0.0174532925199433*chi)**2 + cos(0.0349065850398866*theta)**2) - sin(0.0349065850398866*theta)*cos(0.0174532925199433*chi) + cos(0.0349065850398866*theta)), -sqrt(2)*sin(0.0174532925199433*chi)*sin(0.0349065850398866*theta)/(-sqrt(sin(0.0174532925199433*chi)**2*sin(0.0349065850398866*theta)**2 + sin(0.0349065850398866*theta)**2*cos(0.0174532925199433*chi)**2 + cos(0.0349065850398866*theta)**2) - sin(0.0349065850398866*theta)*cos(0.0174532925199433*chi) + cos(0.0349065850398866*theta))]

def _gnomonic_to_thetachi_sympy():
    """Returns the tree of the sympy expression."""
    x_gnom, y_gnom = symbols('x_gnom y_gnom')
    return [28.6478897565412*acos((y_gnom**2 - (sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + (sqrt(2)*x_gnom/2 + sqrt(2)/2)**2)/sqrt(4*y_gnom**2*(sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + 4*(sqrt(2)*x_gnom/2 - sqrt(2)/2)**2*(sqrt(2)*x_gnom/2 + sqrt(2)/2)**2 + (y_gnom**2 - (sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + (sqrt(2)*x_gnom/2 + sqrt(2)/2)**2)**2)), -57.2957795130823*asin(2*y_gnom*(sqrt(2)*x_gnom/2 - sqrt(2)/2)/sqrt(4*y_gnom**2*(sqrt(2)*x_gnom/2 - sqrt(2)/2)**2 + 4*(sqrt(2)*x_gnom/2 - sqrt(2)/2)**2*(sqrt(2)*x_gnom/2 + sqrt(2)/2)**2))]

def _dist_cosine_sympy():
    """Returns the tree of the sympy expression."""
    theta_1, chi_1, theta_2, chi_2 = symbols('theta_1 chi_1 theta_2 chi_2')
    return 57.2957795130823*acos(((-sqrt(sin(0.0174532925199433*chi_1)**2*sin(0.0349065850398866*theta_1)**2 + sin(0.0349065850398866*theta_1)**2*cos(0.0174532925199433*chi_1)**2 + cos(0.0349065850398866*theta_1)**2) + cos(0.0349065850398866*theta_1))*(-sqrt(sin(0.0174532925199433*chi_2)**2*sin(0.0349065850398866*theta_2)**2 + sin(0.0349065850398866*theta_2)**2*cos(0.0174532925199433*chi_2)**2 + cos(0.0349065850398866*theta_2)**2) + cos(0.0349065850398866*theta_2)) + sin(0.0174532925199433*chi_1)*sin(0.0174532925199433*chi_2)*sin(0.0349065850398866*theta_1)*sin(0.0349065850398866*theta_2) + sin(0.0349065850398866*theta_1)*sin(0.0349065850398866*theta_2)*cos(0.0174532925199433*chi_1)*cos(0.0174532925199433*chi_2))/(sqrt((sqrt(sin(0.0174532925199433*chi_1)**2*sin(0.0349065850398866*theta_1)**2 + sin(0.0349065850398866*theta_1)**2*cos(0.0174532925199433*chi_1)**2 + cos(0.0349065850398866*theta_1)**2) - cos(0.0349065850398866*theta_1))**2 + sin(0.0174532925199433*chi_1)**2*sin(0.0349065850398866*theta_1)**2 + sin(0.0349065850398866*theta_1)**2*cos(0.0174532925199433*chi_1)**2)*sqrt((sqrt(sin(0.0174532925199433*chi_2)**2*sin(0.0349065850398866*theta_2)**2 + sin(0.0349065850398866*theta_2)**2*cos(0.0174532925199433*chi_2)**2 + cos(0.0349065850398866*theta_2)**2) - cos(0.0349065850398866*theta_2))**2 + sin(0.0174532925199433*chi_2)**2*sin(0.0349065850398866*theta_2)**2 + sin(0.0349065850398866*theta_2)**2*cos(0.0174532925199433*chi_2)**2)))

def _dist_euclidian_sympy():
    """Returns the tree of the sympy expression."""
    x1, y1, x2, y2 = symbols('x1 y1 x2 y2')
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def _dist_line_sympy():
    """Returns the tree of the sympy expression."""
    phi, mu, x, y = symbols('phi mu x y')
    return Abs(-mu + x*cos(phi) + y*sin(phi))

def _hough_sympy():
    """Returns the tree of the sympy expression."""
    x_a, y_a, x_b, y_b = symbols('x_a y_a x_b y_b')
    return [acos((x_a*((x_a - x_b)**2 + (y_a - y_b)**2) - (x_a - x_b)*(x_a*(x_a - x_b) + y_a*(y_a - y_b)))/sqrt((x_a*((x_a - x_b)**2 + (y_a - y_b)**2) - (x_a - x_b)*(x_a*(x_a - x_b) + y_a*(y_a - y_b)))**2 + (y_a*((x_a - x_b)**2 + (y_a - y_b)**2) - (y_a - y_b)*(x_a*(x_a - x_b) + y_a*(y_a - y_b)))**2))*sign((y_a*((x_a - x_b)**2 + (y_a - y_b)**2) - (y_a - y_b)*(x_a*(x_a - x_b) + y_a*(y_a - y_b)))/sqrt((x_a*((x_a - x_b)**2 + (y_a - y_b)**2) - (x_a - x_b)*(x_a*(x_a - x_b) + y_a*(y_a - y_b)))**2 + (y_a*((x_a - x_b)**2 + (y_a - y_b)**2) - (y_a - y_b)*(x_a*(x_a - x_b) + y_a*(y_a - y_b)))**2)), sqrt(x_a**4*y_b**2 - 2*x_a**3*x_b*y_a*y_b - 2*x_a**3*x_b*y_b**2 + x_a**2*x_b**2*y_a**2 + 4*x_a**2*x_b**2*y_a*y_b + x_a**2*x_b**2*y_b**2 + x_a**2*y_a**2*y_b**2 - 2*x_a**2*y_a*y_b**3 + x_a**2*y_b**4 - 2*x_a*x_b**3*y_a**2 - 2*x_a*x_b**3*y_a*y_b - 2*x_a*x_b*y_a**3*y_b + 4*x_a*x_b*y_a**2*y_b**2 - 2*x_a*x_b*y_a*y_b**3 + x_b**4*y_a**2 + x_b**2*y_a**4 - 2*x_b**2*y_a**3*y_b + x_b**2*y_a**2*y_b**2)/Abs(x_a**2 - 2*x_a*x_b + x_b**2 + y_a**2 - 2*y_a*y_b + y_b**2)]

def _inter_line_sympy():
    """Returns the tree of the sympy expression."""
    phi_1, mu_1, phi_2, mu_2 = symbols('phi_1 mu_1 phi_2 mu_2')
    return [(-mu_1*sin(phi_2) + mu_2*sin(phi_1))/sin(phi_1 - phi_2), (mu_1*cos(phi_2) - mu_2*cos(phi_1))/sin(phi_1 - phi_2)]
