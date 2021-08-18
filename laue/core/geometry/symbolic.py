#!/usr/bin/env python3

"""
Recherche les equations symboliques qui expriment
les transformations geometriques.
"""

import hashlib
import importlib.util
import inspect
import numbers
import os
import pickle

import numpy as np
import sympy

import laue
import laue.utilities.lambdify as lambdify


class Equations:
    """
    Exprime des petites transformations elementaires.

    C'est une interface de la classe ``laue.core.geometry.transformer.Transformer``.
    """
    def __init__(self):
        # Les constantes.
        self.dd = sympy.Symbol("dd", real=True, positive=True) # Distance entre l'origine et le plan de la camera en mm.
        self.xcen, self.ycen = sympy.symbols("xcen ycen", real=True) # Position du point d'incidence normale en pxl par rapport au repere de la camera.
        self.xbet, self.xgam = sympy.symbols("beta gamma", real=True) # Rotation autour x camera, Rotation autour axe incidence normale.
        self.pixelsize = sympy.Symbol("pixelsize", real=True, positive=True) # Taille des pixels en mm/pxl.

        # Les variables.
        self.x_cam, self.y_cam = sympy.symbols("x_cam y_cam", real=True, positive=True) # Position du pxl dans le repere du plan de la camera.
        self.x_gnom, self.y_gnom = sympy.symbols("x_gnom y_gnom", real=True) # Position des points dans le plan gnomonic.
        self.theta, self.chi = sympy.symbols("theta chi", real=True) # Les angles decrivant le rayon reflechi.

        # Expression des elements du model.
        self.rx = sympy.Matrix([1, 0, 0])
        self.ry = sympy.Matrix([0, 1, 0])
        self.rz = sympy.Matrix([0, 0, 1])

        self.u_i = self.rx # Le rayon de lumiere incident norme parallele a l'axe X dans le repere du cristal.

        self.rot_camera = sympy.rot_axis2(-self.xbet*sympy.pi/180) @ sympy.rot_axis3(self.xgam*sympy.pi/180) # Rotation globale de la camera par rapport au cristal.
        self.ci = self.rot_camera @ -self.ry # Vecteur Xcamera.
        self.cj = self.rot_camera @ self.rx # Vecteur Ycamera.
        self.ck = self.rot_camera @ self.rz # Vecteur Zcamera normal au plan de la camera.

        self.rot_gnom = sympy.rot_axis2(-sympy.pi/4) # Rotation du repere de plan gnomonic par rapport au repere du cristal.
        self.gi = self.rot_gnom @ self.rz # Vecteur Xgnomonic.
        self.gj = self.rot_gnom @ self.ry # Vecteur Ygnomonic.
        self.gk = self.rot_gnom @ -self.rx # Vecteur Zgnomonic normal au plan gnomonic.

    def get_expr_cam_to_uf(self, x_cam, y_cam):
        """
        ** Equation permetant de passer de la camera a uf. **

        Notes
        -----
        Le vecteur de sortie (uf) n'est pas normalise.

        Parameters
        ----------
        x_cam
            La position x de la camera.
        y_cam
            La position y de la camera.

        Returns
        -------
        sympy.Matrix
            La matrice sympy de taille 3 representant
            le vecteur uf dans le repere principal.

        Examples
        --------
        >>> from sympy import symbols
        >>> from laue import Transformer
        >>> transformer = Transformer()
        >>> x_cam, y_cam = symbols("x y")
        >>> transformer.get_expr_cam_to_uf(x_cam, y_cam)
        Matrix([
        [dd*sin(pi*beta/180) - pixelsize*((x - xcen)*sin(pi*gamma/180)*cos(pi*beta/180) - (y - ycen)*cos(pi*beta/180)*cos(pi*gamma/180))],
        [                                                       -pixelsize*((x - xcen)*cos(pi*gamma/180) + (y - ycen)*sin(pi*gamma/180))],
        [dd*cos(pi*beta/180) + pixelsize*((x - xcen)*sin(pi*beta/180)*sin(pi*gamma/180) - (y - ycen)*sin(pi*beta/180)*cos(pi*gamma/180))]])
        >>>
        """
        x_cam_atomic, y_cam_atomic = sympy.symbols("x_cam y_cam", real=True)
        o_op = self.dd * self.ck # Vecteur OO'.
        op_p = self.pixelsize * ((x_cam_atomic-self.xcen)*self.ci + (y_cam_atomic-self.ycen)*self.cj) # Vecteur O'P
        o_p = o_op + op_p # Relation de Chasles.
        o_p = sympy.signsimp(o_p)
        return o_p.subs({x_cam_atomic: x_cam, y_cam_atomic: y_cam})

    def get_expr_uf_to_cam(self, uf_x, uf_y, uf_z):
        """
        ** Equation permettant de passer de uf a la camera. **

        Parameters
        ----------
        uf_x, uf_y, uf_z
            Les 3 coordonnees du vecteur uf exprimees dans le repere principale.

        Returns
        -------
        x_camera : sympy.Basic
            Expression sympy de la position de la tache dans le repere
            de la camera. (selon l'axe x ou Ci)
        y_camera : sympy.Basic
            Comme ``x_camera`` selon l'axe y ou Cj.

        Examples
        --------
        >>> from sympy import symbols
        >>> from laue import Transformer
        >>> transformer = Transformer()
        >>> uf_x, uf_y, uf_z = symbols("uf_x, uf_y, uf_z")
        >>> x_cam, y_cam = transformer.get_expr_uf_to_cam(uf_x, uf_y, uf_z)
        >>> x_cam
        (-dd*uf_x*sin(pi*gamma/180)*cos(pi*beta/180) - dd*uf_y*cos(pi*gamma/180) + dd*uf_z*sin(pi*beta/180)*sin(pi*gamma/180) + pixelsize*uf_x*xcen*sin(pi*beta/180) + pixelsize*uf_z*xcen*cos(pi*beta/180))/(pixelsize*(uf_x*sin(pi*beta/180) + uf_z*cos(pi*beta/180)))
        >>> y_cam
        (dd*uf_x*cos(pi*beta/180)*cos(pi*gamma/180) - dd*uf_y*sin(pi*gamma/180) - dd*uf_z*sin(pi*beta/180)*cos(pi*gamma/180) + pixelsize*uf_x*ycen*sin(pi*beta/180) + pixelsize*uf_z*ycen*cos(pi*beta/180))/(pixelsize*(uf_x*sin(pi*beta/180) + uf_z*cos(pi*beta/180)))
        >>>
        """
        uf_x_atomic, uf_y_atomic, uf_z_atomic = sympy.symbols("uf_x uf_y uf_z", real=True)
        u_f = sympy.Matrix([uf_x_atomic, uf_y_atomic, uf_z_atomic])

        # Expression du vecteur O''P.
        opp_op = self.pixelsize * (self.xcen*self.ci + self.ycen*self.cj) # Vecteur O''O'.
        o_op = self.dd * self.ck # Vecteur OO'.
        op_o = -o_op # Vecteur O'O.
        camera_plane = sympy.Plane(o_op, normal_vector=self.ck) # Plan de la camera.
        refl_ray = sympy.Line([0, 0, 0], u_f) # Rayon reflechi.
        o_p = sympy.Matrix(camera_plane.intersection(refl_ray).pop())
        opp_p = opp_op + op_o + o_p # Relation de Chasles.

        # Projection dans le plan de la camera pour remonter a x_c, y_c
        x_cam = opp_p.dot(self.ci) / self.pixelsize # Coordonnees en pxl axe x de la camera.
        y_cam = opp_p.dot(self.cj) / self.pixelsize # Coordonnees en pxl axe y de la camera.
        x_cam, y_cam = sympy.trigsimp(x_cam), sympy.trigsimp(y_cam) # Longueur reduite par 2.2 .

        return (x_cam.subs({uf_x_atomic: uf_x, uf_y_atomic: uf_y, uf_z_atomic: uf_z}),
                y_cam.subs({uf_x_atomic: uf_x, uf_y_atomic: uf_y, uf_z_atomic: uf_z}))

    def get_expr_uf_to_uq(self, uf_x, uf_y, uf_z):
        """
        ** Equation permettant de passer de uf a uq. **

        Parameters
        ----------
        uf_x, uf_y, uf_z
            Les 3 coordonnees du vecteur uf exprimees dans le repere principal.

        Returns
        -------
        sympy.Matrix
            La matrice sympy de taille 3 representant
            le vecteur uq dans le repere principal.

        Examples
        --------
        >>> from sympy import symbols
        >>> from laue import Transformer
        >>> transformer = Transformer()
        >>> uf_x, uf_y, uf_z = symbols("uf_x, uf_y, uf_z")
        >>> transformer.get_expr_uf_to_uq(uf_x, uf_y, uf_z)
        Matrix([
        [uf_x - sqrt(uf_x**2 + uf_y**2 + uf_z**2)],
        [                                    uf_y],
        [                                    uf_z]])
        >>>
        """
        uf_x_atomic, uf_y_atomic, uf_z_atomic = sympy.symbols("uf_x uf_y uf_z", real=True)
        u_f = sympy.Matrix([uf_x_atomic, uf_y_atomic, uf_z_atomic])
        u_q = u_f - self.u_i*u_f.norm() # Relation de reflexion.
        return u_q.subs({uf_x_atomic: uf_x, uf_y_atomic: uf_y, uf_z_atomic: uf_z})

    def get_expr_uq_to_uf(self, uq_x, uq_y, uq_z):
        """
        ** Equation permettant de passer de uq a uf. **

        Parameters
        ----------
        uq_x, uq_y, uq_z
            Les 3 coordonnees du vecteur uq exprimees dans le repere principal.

        Returns
        -------
        sympy.Matrix
            La matrice sympy de taille 3 representant
            le vecteur uf dans le repere principal.

        Examples
        --------
        >>> from sympy import symbols
        >>> from laue import Transformer
        >>> transformer = Transformer()
        >>> uq_x, uq_y, uq_z = symbols("uq_x, uq_y, uq_z")
        >>> transformer.get_expr_uq_to_uf(uq_x, uq_y, uq_z)
        Matrix([
        [-uq_x**2 + uq_y**2 + uq_z**2],
        [                -2*uq_x*uq_y],
        [                -2*uq_x*uq_z]])
        >>>
        """
        uq_x_atomic, uq_y_atomic, uq_z_atomic = sympy.symbols("uq_x uq_y uq_z", real=True)
        u_q = sympy.Matrix([uq_x_atomic, uq_y_atomic, uq_z_atomic])
        u_f = self.u_i*u_q.norm()**2 - 2*u_q.dot(self.u_i)*u_q # Vecteur unitaire reflechi.
        return u_f.subs({uq_x_atomic: uq_x, uq_y_atomic: uq_y, uq_z_atomic: uq_z})

    def get_expr_uq_to_gnomonic(self, uq_x, uq_y, uq_z):
        """
        ** Equation permettant de passer de uq a gnomonic. **

        Parameters
        ----------
        uq_x, uq_y, uq_z
            Les 3 coordonnees du vecteur uq exprimees dans le repere principal.

        Returns
        -------
        x_gnomonic : sympy.Basic
            Expression sympy de la position de la tache dans le repere
            du plan gnomonic. (selon l'axe x ou Gi)
        y_gnomonic : sympy.Basic
            Comme ``x_gnomonic`` selon l'axe y ou Gj.

        Examples
        --------
        >>> from sympy import symbols
        >>> from laue import Transformer
        >>> transformer = Transformer()
        >>> uq_x, uq_y, uq_z = symbols("uq_x, uq_y, uq_z")
        >>> x_gnom, y_gnom = transformer.get_expr_uq_to_gnomonic(uq_x, uq_y, uq_z)
        >>> x_gnom
        -(uq_x + uq_z)/(uq_x - uq_z)
        >>> y_gnom
        -sqrt(2)*uq_y/(uq_x - uq_z)
        >>>
        """
        uq_x_atomic, uq_y_atomic, uq_z_atomic = sympy.symbols("uq_x uq_y uq_z", real=True)
        u_q = sympy.Matrix([uq_x_atomic, uq_y_atomic, uq_z_atomic])
        o_oppp = 1*self.gk # Car sphere unitaire de rayon 1.

        gnom_plane = sympy.Plane(o_oppp, normal_vector=self.gk) # Plan gnomonic.
        normal_ray = sympy.Line([0, 0, 0], u_q) # Droite portee par la normal au plan christalin.
        o_pp = sympy.Matrix(gnom_plane.intersection(normal_ray).pop())
        oppp_pp = -o_oppp + o_pp

        # Projection dans le plan gnomonic pour remonter a x_g, y_g.
        x_gnom = oppp_pp.dot(self.gi) # Coordonnees en mm axe x du plan gnomonic.
        y_gnom = oppp_pp.dot(self.gj) # Coordonnees en mm axe y du plan gnomonic.

        x_gnom = sympy.signsimp(sympy.cancel(x_gnom))

        return (x_gnom.subs({uq_x_atomic: uq_x, uq_y_atomic: uq_y, uq_z_atomic: uq_z}),
                y_gnom.subs({uq_x_atomic: uq_x, uq_y_atomic: uq_y, uq_z_atomic: uq_z}))

    def get_expr_gnomonic_to_uq(self, x_gnom, y_gnom):
        """
        ** Equation permettant de passer du plan gnomonique a uq. **

        Parameters
        ----------
        x_gnom
            La position x du plan gnomonic.
        y_gnom
            La position y du plan gnomonic.

        Returns
        -------
        sympy.Matrix
            La matrice sympy de taille 3 representant
            le vecteur uq dans le repere principal.

        Examples
        --------
        >>> from sympy import symbols
        >>> from laue import Transformer
        >>> transformer = Transformer()
        >>> x_gnom, y_gnom = symbols("x y")
        >>> transformer.get_expr_gnomonic_to_uq(x_gnom, y_gnom)
        Matrix([
        [sqrt(2)*x/2 - sqrt(2)/2],
        [                      y],
        [sqrt(2)*x/2 + sqrt(2)/2]])
        >>>
        """
        x_gnom_atomic, y_gnom_atomic = sympy.symbols("x_gnom y_gnom", real=True)

        o_oppp = 1*self.gk # Vecteur OO''' == gk car le plan gnomonic est tangent a la shere unitaire.
        u_q = o_oppp + (x_gnom_atomic*self.gi + y_gnom_atomic*self.gj) # Relation de chasle.

        return u_q.subs({x_gnom_atomic: x_gnom, y_gnom_atomic: y_gnom})

    def get_expr_uf_to_thetachi(self, uf_x, uf_y, uf_z):
        """
        ** Equation permetant de passer de uf a thetachi. **

        Notes
        -----
        Les resultats sont en radian. Il ne sont pas converti
        en degres a ce stade car ca gene la simplification.

        Parameters
        ----------
        uf_x, uf_y, uf_z
            Les 3 coordonnees du vecteur uf exprimees dans le repere principal.

        Returns
        -------
        theta : sympy.Basic
            Angle de rotation du plan christalin autour de -x.
        chi : sympy.Basic
            La moitier de l'angle de rotation du plan christalin autour de -y.

        >>> from sympy import symbols
        >>> from laue import Transformer
        >>> transformer = Transformer()
        >>> uf_x, uf_y, uf_z = symbols("uf_x, uf_y, uf_z", real=True)
        >>> theta, chi = transformer.get_expr_uf_to_thetachi(uf_x, uf_y, uf_z)
        >>> theta
        acos(uf_x/sqrt(uf_x**2 + uf_y**2 + uf_z**2))/2
        >>> chi
        asin(uf_y/sqrt(uf_y**2 + uf_z**2))
        >>>
        """
        uf_x_atomic, uf_y_atomic, uf_z_atomic = sympy.symbols("uf_x uf_y uf_z", real=True)

        theta_rad = sympy.acos(uf_x_atomic/sympy.sqrt(uf_x_atomic**2 + uf_y_atomic**2 + uf_z_atomic**2))/2
        chi_rad = sympy.asin(uf_y_atomic/sympy.sqrt(uf_y_atomic**2 + uf_z_atomic**2))

        return (theta_rad.subs({uf_x_atomic: uf_x, uf_y_atomic: uf_y, uf_z_atomic: uf_z}),
                chi_rad.subs({uf_x_atomic: uf_x, uf_y_atomic: uf_y, uf_z_atomic: uf_z}))

    def get_expr_thetachi_to_uf(self, theta, chi):
        """
        ** Equation permetant de passer de thetachi a uf. **

        Notes
        -----
        A ce stade, les angles sont exprimes en radian pour des 
        raisons de simplifications symbolique.

        Parameters
        ----------
        theta
            Angle de rotation du plan christalin autour de -x. (en rad)
        chi
            La moitier de l'angle de rotation du plan christalin autour de -y. (en rad)

        Returns
        -------
        sympy.Matrix
            La matrice sympy de taille 3 representant
            le vecteur uf dans le repere principal.

        Examples
        --------
        >>> from sympy import symbols
        >>> from laue import Transformer
        >>> transformer = Transformer()
        >>> theta, chi = symbols("theta chi")
        >>> transformer.get_expr_thetachi_to_uf(theta, chi)
        Matrix([
        [         cos(2*theta)],
        [sin(chi)*sin(2*theta)],
        [sin(2*theta)*cos(chi)]])
        >>>
        """
        theta_atomic, chi_atomic = sympy.symbols("theta chi", real=True)

        # Expresion du rayon reflechit en fonction des angles.
        rot_refl = sympy.rot_axis1(chi_atomic) @ sympy.rot_axis2(2*theta_atomic)
        u_f = rot_refl @ self.u_i

        return u_f.subs({theta_atomic: theta, chi_atomic: chi})

class Compilator(Equations):
    """
    Extrait et enregistre les equations brutes.
    Combine les blocs elementaires.

    Notes
    -----
    Les equations sont enregistrees de facon globales
    de sorte a eviter la recompilation entre chaque objet,
    et permet aussi d'alleger la serialisation de ``Transformer``.
    """
    def __init__(self):
        """
        Genere le dictionaire a protee globale.
        """
        Equations.__init__(self)

        self.compiled_expressions = {}
        self.names = [
            "cam_to_gnomonic",
            "gnomonic_to_cam",
            "cam_to_thetachi",
            "thetachi_to_cam",
            "thetachi_to_gnomonic",
            "gnomonic_to_thetachi",
            "dist_cosine",
            "dist_euclidian",
            "dist_line",
            "hough",
            "inter_line"]
        self.load()
        self._compile()

    def _compile(self):
        """
        ** Precalcule toutes les equations. **
        """
        names = [n for n in self.names if n not in self.compiled_expressions]
        for name in names:
            getattr(self, f"get_fct_{name}")()

        if names:
            self.save() # On enregistre pour gagner du temps les prochaines fois.

    def get_fct_cam_to_gnomonic(self):
        """
        ** Equation permetant de passer de la camera au plan gnomonic. **
        """
        if "cam_to_gnomonic" in self.compiled_expressions:
            return self.compiled_expressions["cam_to_gnomonic"]

        u_f = self.get_expr_cam_to_uf(self.x_cam, self.y_cam)
        u_q = self.get_expr_uf_to_uq(*u_f)
        x_gnom, y_gnom = self.get_expr_uq_to_gnomonic(*u_q)

        self.compiled_expressions["cam_to_gnomonic"] = lambdify.Lambdify(
            args=[self.x_cam, self.y_cam, self.dd, self.xcen, self.ycen, self.xbet, self.xgam, self.pixelsize],
            expr=[x_gnom, y_gnom]) # On l'enregistre une bonne fois pour toutes.
        return self.compiled_expressions["cam_to_gnomonic"]

    def get_fct_cam_to_thetachi(self):
        """
        ** Equation permetant de passer de la camera a la representation theta chi. **

        theta et chi sont exprimes en degres
        """
        if "cam_to_thetachi" in self.compiled_expressions:
            return self.compiled_expressions["cam_to_thetachi"]

        u_f = self.get_expr_cam_to_uf(self.x_cam, self.y_cam)
        theta_rad, chi_rad = self.get_expr_uf_to_thetachi(*u_f)
        theta_deg, chi_deg = theta_rad*180/np.pi, chi_rad*180/np.pi

        self.compiled_expressions["cam_to_thetachi"] = lambdify.Lambdify(
            args=[self.x_cam, self.y_cam, self.dd, self.xcen, self.ycen, self.xbet, self.xgam, self.pixelsize],
            expr=[theta_deg, chi_deg]) # On l'enregistre une bonne fois pour toutes.
        return self.compiled_expressions["cam_to_thetachi"]

    def get_fct_dist_cosine(self):
        """
        ** Equation de la cosine distance des vecteurs uq. **

        theta et chi sont exprimes en degres

        def calculdist_from_thetachi(listpoints1, listpoints2):
            data1 = np.array(listpoints1)
            data2 = np.array(listpoints2)
            # print "data1",data1
            # print "data2",data2
            longdata1 = data1[:, 0] * DEG  # theta
            latdata1 = data1[:, 1] * DEG  # chi

            longdata2 = data2[:, 0] * DEG  # theta
            latdata2 = data2[:, 1] * DEG  # chi

            deltalat = latdata1 - np.reshape(latdata2, (len(latdata2), 1))
            longdata2new = np.reshape(longdata2, (len(longdata2), 1))
            prodcos = np.cos(longdata1) * np.cos(longdata2new)
            prodsin = np.sin(longdata1) * np.sin(longdata2new)

            arccos_arg = np.around(prodsin + prodcos * np.cos(deltalat), decimals=9)

            tab_angulardist = (1.0 / DEG) * np.arccos(arccos_arg)

            return tab_angulardist
        """
        if "dist_cosine" in self.compiled_expressions:
            return self.compiled_expressions["dist_cosine"]

        theta1, chi1 = sympy.symbols("theta_1 chi_1", real=True)
        theta2, chi2 = sympy.symbols("theta_2 chi_2", real=True)
        uq_1 = self.get_expr_uf_to_uq(*self.get_expr_thetachi_to_uf(theta1*np.pi/180, chi1*np.pi/180))
        uq_2 = self.get_expr_uf_to_uq(*self.get_expr_thetachi_to_uf(theta2*np.pi/180, chi2*np.pi/180))
        dist_expr = sympy.acos(uq_1.dot(uq_2)/(uq_1.norm()*uq_2.norm()))*180/np.pi # Cosine distance.

        self.compiled_expressions["dist_cosine"] = lambdify.Lambdify(
            [theta1, chi1, theta2, chi2], dist_expr)
        return self.compiled_expressions["dist_cosine"]

    def get_fct_dist_euclidian(self):
        """
        ** Simple norme euclidiene en 2d. **
        """
        if "dist_euclidian" in self.compiled_expressions:
            return self.compiled_expressions["dist_euclidian"]

        x1, y1, x2, y2 = sympy.symbols("x1 y1, x2, y2", real=True)
        dist_expr = sympy.sqrt((x1-x2)**2 + (y1-y2)**2)

        self.compiled_expressions["dist_euclidian"] = lambdify.Lambdify(
            [x1, y1, x2, y2], dist_expr)
        return self.compiled_expressions["dist_euclidian"]

    def get_fct_dist_line(self):
        """
        ** Equation de projection de points sur une droite. **
        """
        if "dist_line" in self.compiled_expressions:
            return self.compiled_expressions["dist_line"]

        # Creation de la droite.
        phi, mu, x, y = sympy.symbols("phi mu x y", real=True)
        p = sympy.Point(mu*sympy.cos(phi), mu*sympy.sin(phi)) # Point appartenant a la droite.
        op = sympy.Line(sympy.Point(0, 0), p) # Droite normale a la droite principale.
        line = op.perpendicular_line(p) # C'est la droite principale.

        # Projection des points.
        distance = line.distance(sympy.Point(x, y)) # La distance entre la droite et un point.

        # Optimisation.
        distance = sympy.trigsimp(distance) # Permet un gain de 2.90

        # Vectorisation de l'expression.
        self.compiled_expressions["dist_line"] = lambdify.Lambdify([phi, mu, x, y], distance)
        return self.compiled_expressions["dist_line"]

    def get_fct_gnomonic_to_cam(self):
        """
        ** Equation permetant de passer de l'espace gnomonic a celui de la camera. **
        """
        if "gnomonic_to_cam" in self.compiled_expressions:
            return self.compiled_expressions["gnomonic_to_cam"]

        u_q = self.get_expr_gnomonic_to_uq(self.x_gnom, self.y_gnom)
        u_f = self.get_expr_uq_to_uf(*u_q)
        x_c, y_c = self.get_expr_uf_to_cam(*u_f)

        self.compiled_expressions["gnomonic_to_cam"] = lambdify.Lambdify(
            args=[self.x_gnom, self.y_gnom, self.dd, self.xcen, self.ycen, self.xbet, self.xgam, self.pixelsize],
            expr=[x_c, y_c]) # On l'enregistre une bonne fois pour toutes.
        return self.compiled_expressions["gnomonic_to_cam"]

    def get_fct_gnomonic_to_thetachi(self):
        """
        ** Equation permetant de passer du plan gnomonic a la representation theta chi. **

        theta et chi sont exprimes en degres
        """
        if "gnomonic_to_thetachi" in self.compiled_expressions:
            return self.compiled_expressions["gnomonic_to_thetachi"]

        u_q = self.get_expr_gnomonic_to_uq(self.x_gnom, self.y_gnom)
        u_f = self.get_expr_uq_to_uf(*u_q)
        theta_rad, chi_rad = self.get_expr_uf_to_thetachi(*u_f)
        theta_deg, chi_deg = theta_rad*180/np.pi, chi_rad*180/np.pi

        self.compiled_expressions["gnomonic_to_thetachi"] = lambdify.Lambdify(
            args=[self.x_gnom, self.y_gnom],
            expr=[theta_deg, chi_deg]) # On l'enregistre une bonne fois pour toutes.
        return self.compiled_expressions["gnomonic_to_thetachi"]

    def get_fct_hough(self):
        """
        ** Equation pour la transformee de hough. **
        """
        if "hough" in self.compiled_expressions:
            return self.compiled_expressions["hough"]

        xa, ya, xb, yb = sympy.symbols("x_a y_a x_b y_b", real=True)
        u = sympy.Matrix([xa-xb, ya-yb]).normalized()
        x = sympy.Matrix([1, 0])

        # Calcul de la distance entre la droite et l'origine.
        d1 = sympy.Line(sympy.Point(xa, ya), sympy.Point(xb, yb)) # C'est la droite passant par les 2 points.
        mu = d1.distance(sympy.Point(0, 0)) # La distance separant l'origine de la droite.

        # Calcul de l'angle entre l'axe horizontal et la droite.
        p = d1.projection(sympy.Point(0, 0)) # Le point ou la distance entre ce point de la droite et l'origine est minimale.
        n = p / sympy.sqrt(p.x**2 + p.y**2) # On normalise le point.
        phi_abs = sympy.acos(n.x) # La valeur absolue de phi.
        phi_sign = sympy.sign(n.y) # Si il est negatif c'est que phi < 0, si il est positif alors phi > 0
        phi = phi_abs * phi_sign # Compris entre -pi et +pi

        # Optimisation.
        mu = sympy.trigsimp(sympy.cancel(mu)) # Permet un gain de 1.40

        # Vectorisation des expressions.
        self.compiled_expressions["hough"] = lambdify.Lambdify([xa, ya, xb, yb], [phi, mu])
        return self.compiled_expressions["hough"]

    def get_fct_inter_line(self):
        """
        ** Equation d'intersection entre 2 droites. **
        """
        if "inter_line" in self.compiled_expressions:
            return self.compiled_expressions["inter_line"]

        # Creation des 2 droites.
        phi_1, mu_1, phi_2, mu_2 = sympy.symbols("phi_1, mu_1, phi_2, mu_2", real=True)
        p1 = sympy.Point(mu_1*sympy.cos(phi_1), mu_1*sympy.sin(phi_1)) # Point appartenant a la premiere droite.
        p2 = sympy.Point(mu_2*sympy.cos(phi_2), mu_2*sympy.sin(phi_2)) # Point appartenant a la seconde droite.
        op1 = sympy.Line(sympy.Point(0, 0), p1) # Droite normale a la premiere droite.
        op2 = sympy.Line(sympy.Point(0, 0), p2) # Droite normale a la deuxieme droite.
        line1 = op1.perpendicular_line(p1) # La premiere droite.
        line2 = op2.perpendicular_line(p2) # La seconde droite.

        # Calcul des coordonnes du point d'intersection.
        point = line1.intersection(line2)[0]
        inter_x = point.x
        inter_y = point.y

        # Optimisation.
        # Il n'y en a pas car les expressions sont deja tres simples.

        # Vectorisation des expressions.
        self.compiled_expressions["inter_line"] = lambdify.Lambdify(
            [phi_1, mu_1, phi_2, mu_2], [inter_x, inter_y])
        return self.compiled_expressions["inter_line"]

    def get_fct_thetachi_to_gnomonic(self):
        """
        ** Equation permetant de passer de theta chi au plan gnomonic. **

        theta et chi doivent etre donnes en degres.
        """
        if "thetachi_to_gnomonic" in self.compiled_expressions:
            return self.compiled_expressions["thetachi_to_gnomonic"]

        u_f = self.get_expr_thetachi_to_uf(self.theta*np.pi/180, self.chi*np.pi/180)
        u_q = self.get_expr_uf_to_uq(*u_f)
        x_gnom, y_gnom = self.get_expr_uq_to_gnomonic(*u_q)

        self.compiled_expressions["thetachi_to_gnomonic"] = lambdify.Lambdify(
            args=[self.theta, self.chi],
            expr=[x_gnom, y_gnom]) # On l'enregistre une bonne fois pour toutes.
        return self.compiled_expressions["thetachi_to_gnomonic"]

    def get_fct_thetachi_to_cam(self):
        """
        ** Equation permetant de passer de theta chi a la camera. **

        theta et chi doivent etre donnes en degres.
        """
        if "thetachi_to_cam" in self.compiled_expressions:
            return self.compiled_expressions["thetachi_to_cam"]

        u_f = self.get_expr_thetachi_to_uf(self.theta*np.pi/180, self.chi*np.pi/180)
        x_c, y_c = self.get_expr_uf_to_cam(*u_f)

        self.compiled_expressions["thetachi_to_cam"] = lambdify.Lambdify(
            args=[self.theta, self.chi, self.dd, self.xcen, self.ycen, self.xbet, self.xgam, self.pixelsize],
            expr=[x_c, y_c]) # On l'enregistre une bonne fois pour toutes.
        return self.compiled_expressions["thetachi_to_cam"]

    def _hash(self):
        """
        ** Retourne le hash de ce code. **
        """
        return hashlib.md5(
            inspect.getsource(Compilator).encode(encoding="utf-8")
          + inspect.getsource(Equations).encode(encoding="utf-8")
          + inspect.getsource(lambdify).encode(encoding="utf-8")
            ).hexdigest()

    def load(self):
        """
        ** Charge si il existe, le fichier contenant les expressions. **

        Deverse les expressions dans le dictionaire: ``self.compiled_expressions``.
        """
        def path_import(absolute_path):
            spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        dirname = os.path.dirname(os.path.abspath(laue.__file__))
        file = os.path.join(dirname, "data", "lambdifygenerated.py")
        try:
            lambdifygenerated = path_import(file)
        except FileNotFoundError:
            return self.compiled_expressions
        else:
            if lambdifygenerated.HASH != self._hash():
                return self.compiled_expressions
            for name in self.names:
                if name in self.compiled_expressions:
                    continue
                self.compiled_expressions[name] = getattr(lambdifygenerated, name)
            return self.compiled_expressions

    def save(self):
        """
        ** Enregistre un module contenant les expressions. **

        Enregistre seulement ce qui est present dans ``self.compiled_expressions``.
        N'ecrase pas l'ancien contenu.
        """
        import time
        dirname = os.path.join(os.path.dirname(os.path.abspath(laue.__file__)), "data")
        self.load() # Recuperation du contenu du fichier.

        # Ecriture du module principal.
        with open(os.path.join(dirname, "lambdifygenerated.py"), "w", encoding="utf-8") as f:
            f.write( "#!/usr/bin/env python3\n")
            f.write( "\n")
            f.write( '"""\n')
            f.write(f"This code was automatically generated on {time.ctime()}.\n")
            f.write( '"""\n')
            f.write( "\n")
            f.write( "import sympy\n")
            f.write( "import numpy as np\n")
            f.write( "\n")
            f.write(f"HASH = {repr(self._hash())}\n")
            f.write( "\n")
            for func_name, lamb in self.compiled_expressions.items():
                if not isinstance(lamb, lambdify.Lambdify): # C'est juste de la prevention,
                    del self.compiled_expressions[func_name] # ce n'est pas cence servir.
                    getattr(self, f"get_fct_{func_name}")()
                f.write(lamb.__str__(name=func_name, bloc="main"))
                f.write("\n")

        # Ecriture des modules secondaires.
        from sympy.utilities.lambdify import MODULES
        for mod in ["numpy", "numpy128", "numexpr", "sympy"]:
            header = (
                "\n".join(MODULES[mod if mod != "numpy128" else "numpy"][-1])
                if mod != "numexpr" else "from numexpr import evaluate")
            header += "\n"
            if mod == "sympy":
                header += "from sympy import symbols\n"

            with open(os.path.join(dirname, f"{mod}_lambdify.py"), "w", encoding="utf-8") as f:
                f.write("#!/usr/bin/env python3\n")
                f.write("\n")
                f.write('"""\n')
                f.write("This code was automatically generated.\n")
                f.write('"""\n')
                f.write("\n")
                f.write(header)
                f.write("\n\n")
                for func_name, lamb in self.compiled_expressions.items():
                    f.write(lamb.__str__(name=func_name, bloc=mod))
