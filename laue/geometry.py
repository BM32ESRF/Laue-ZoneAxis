#!/usr/bin/env python3

"""
** Permet d'appliquer des transformations geometriques. **
----------------------------------------------------------

Notes
-----
Le module ``numexpr`` permet d'accelerer les calculs si il est installe.
"""

import collections
import hashlib
import inspect
import math
import multiprocessing
import os

import cloudpickle
import numpy as np
try:
    import numexpr
except ImportError:
    numexpr = None
import sympy


__all__ = ["Transformer", "comb2ind", "ind2comb"]


class Compilator:
    """
    Extrait et enregistre les equations brutes.

    Notes
    -----
    Les equations sont enregistrees de facon globale
    de sorte a eviter la recompilation entre chaque objet,
    et permet aussi d'alleger la serialisation de ``Transformer``.
    """
    def __init__(self):
        """
        Genere le dictionaire a protee globale.
        """
        if "compiled_expressions" not in globals():
            globals()["compiled_expressions"] = {}

        self.load()

    def compile(self, parameters=None):
        """
        ** Precalcul toutes les equations. **

        Parameters
        ----------
        parameters : dict, optional
            Les parametres donnes par la fonction ``laue.tools.parsing.extract_parameters``.
            Si ils sont fourni, l'expression est encore un peu
            plus optimisee.
        """
        names = [
            "expr_cam_to_gnomonic",
            "expr_gnomonic_to_cam",
            "expr_thetachi_to_gnomonic",
            "expr_gnomonic_to_thetachi",
            "fct_dist_line",
            "fct_hough",
            "fct_inter_line"]
        names = [n for n in names if n not in globals()["compiled_expressions"]]

        for name in names:
            getattr(self, f"get_{name}")()

        self.save() # On enregistre les grandes equations.

        if parameters is not None:
            assert isinstance(parameters, dict), ("Les parametres doivent founis "
                f"dans un dictionaire, pas dans un {type(parameters).__name__}")
            assert set(parameters) == {"dd", "xbet", "xgam", "xcen", "ycen", "pixelsize"}, \
                ("Les clefs doivent etres 'dd', 'xbet', 'xgam', 'xcen', 'ycen' et 'pixelsize'. "
                f"Or les clefs sont {set(parameters)}.")
            assert all(isinstance(v, (float, int)) for v in parameters.values()), \
                "La valeurs des parametres doivent toutes etre des nombres."

            hash_param = self._hash_parameters(parameters)
            contants = {self.dd: parameters["dd"], # C'est qu'il est tant de faire de l'optimisation.
                        self.xcen: parameters["xcen"],
                        self.ycen: parameters["ycen"],
                        self.xbet: parameters["xbet"],
                        self.xgam: parameters["xgam"],
                        self.pixelsize: parameters["pixelsize"]}
            self._fcts_cam_to_gnomonic[hash_param] = _Lambdify(
                    args=[self.x_cam, self.y_cam],
                    expr=self.get_expr_cam_to_gnomonic().get_expr().subs(contants),
                    modules="numexpr")
            self._fcts_gnomonic_to_cam[hash_param] = _Lambdify(
                    args=[self.x_gnom, self.y_gnom],
                    expr=self.get_expr_gnomonic_to_cam().get_expr().subs(contants),
                    modules="numexpr")

    def get_expr_cam_to_gnomonic(self):
        """
        ** Equation permetant de passer de la camera au plan gnomonic. **
        """
        if "expr_cam_to_gnomonic" in globals()["compiled_expressions"]:
            return globals()["compiled_expressions"]["expr_cam_to_gnomonic"]

        # Expresion du vecteur OP.
        o_op = self.dd * self.ck # Vecteur OO'.
        op_p = self.pixelsize * ((self.x_cam-self.xcen)*self.ci + (self.y_cam-self.ycen)*self.cj) # Vecteur O'P
        o_p = o_op + op_p # Relation de Chasles.

        # Recherche des droites normales au plan christalin.
        u_f = o_p.normalized() # Vecteur norme du rayon diffracte u_f.
        u_q = u_f - self.u_i # Relation de reflexion.

        # Recherche du vecteur O'''P'.
        oppp_pp = u_q / u_q.dot(self.gk) # Point d'intersection avec le plan gnomonic. (Normalisation de l'axe gk.)

        # Projection dans le plan gnomonic pour remonter a x_g, y_g
        oppp_pp = sympy.simplify(oppp_pp)
        x_g = oppp_pp.dot(self.gi) # Coordonnees en mm axe x du plan gnomonic.
        y_g = oppp_pp.dot(self.gj) # Coordonnees en mm axe y du plan gnomonic.

        globals()["compiled_expressions"]["expr_cam_to_gnomonic"] = _Lambdify(
            args=[self.x_cam, self.y_cam, self.dd, self.xcen, self.ycen, self.xbet, self.xgam, self.pixelsize],
            expr=[x_g, y_g], # Les imprecisions de calculs donnent parfois des complexes!
            modules="numexpr") # On l'enregistre une bonne fois pour toutes.
        return globals()["compiled_expressions"]["expr_cam_to_gnomonic"]

    def get_expr_gnomonic_to_cam(self):
        """
        ** Equation permetant de passer de l'espace gnomonic a celui de la camera. **
        """
        if "expr_gnomonic_to_cam" in globals()["compiled_expressions"]:
            return globals()["compiled_expressions"]["expr_gnomonic_to_cam"]

        # Recherche du vecteur u_q.
        o_oppp = self.gk # Vecteur OO''' == gk car le plan gnomonic est tangent a la shere unitaire.
        u_q = o_oppp + (self.x_gnom*self.gi + self.y_gnom*self.gj) # Vecteur non normalise de la normale au plan christalin.
        u_q = u_q.normalized() # Normale unitaire au plan christalin.

        # Lois de la reflexion.
        u_f = self.u_i - 2*u_q.dot(self.u_i)*u_q # Vecteur unitaire reflechi.
        # u_f = sympy.simplify(u_f) # Permet d'accelerer les calculs par la suite.

        # Expression du vecteur O''P.
        opp_op = self.pixelsize * (self.xcen*self.ci + self.ycen*self.cj) # Vecteur O''O'.
        o_op = self.dd * self.ck # Vecteur OO'.
        op_o = -o_op # Vecteur O'O.
        camera_plane = sympy.Plane(o_op, normal_vector=self.ck) # Plan de la camera.
        refl_ray = sympy.Line([0, 0, 0], u_f) # Rayon reflechi.
        o_p = sympy.Matrix(camera_plane.intersection(refl_ray).pop())
        opp_p = opp_op + op_o + o_p # Relation de Chasles.

        # Projection dans le plan de la camera pour remonter a x_c, y_c
        opp_p = sympy.simplify(opp_p)
        x_c = opp_p.dot(self.ci) / self.pixelsize # Coordonnees en pxl axe x de la camera.
        y_c = opp_p.dot(self.cj) / self.pixelsize # Coordonnees en pxl axe y de la camera.

        globals()["compiled_expressions"]["expr_gnomonic_to_cam"] = _Lambdify(
            args=[self.x_gnom, self.y_gnom, self.dd, self.xcen, self.ycen, self.xbet, self.xgam, self.pixelsize],
            expr=[x_c, y_c],
            modules="numexpr") # On l'enregistre une bonne fois pour toutes.
        return globals()["compiled_expressions"]["expr_gnomonic_to_cam"]

    def get_expr_thetachi_to_gnomonic(self):
        """
        ** Equation permetant de passer de theta chi au plan gnomonic. **
        """
        if "expr_thetachi_to_gnomonic" in globals()["compiled_expressions"]:
            return globals()["compiled_expressions"]["expr_thetachi_to_gnomonic"]

        # Expresion du rayon reflechit en fonction des angles.
        rot_refl = sympy.rot_axis1(self.chi) @ sympy.rot_axis2(2*self.theta)

        # Recherche des droites normales au plan christalin.
        u_f = rot_refl @ self.u_i # Vecteur norme du rayon diffracte u_f.
        u_q = u_f - self.u_i # Relation de reflexion.

        # Recherche du vecteur O'''P'.
        oppp_pp = u_q / u_q.dot(self.gk) # Point d'intersection avec le plan gnomonic. (Normalisation de l'axe gk.)

        # Projection dans le plan gnomonic pour remonter a x_g, y_g
        oppp_pp = sympy.simplify(oppp_pp)
        x_g = oppp_pp.dot(self.gi) # Coordonnees en mm axe x du plan gnomonic.
        y_g = oppp_pp.dot(self.gj) # Coordonnees en mm axe y du plan gnomonic.

        globals()["compiled_expressions"]["expr_thetachi_to_gnomonic"] = _Lambdify(
            args=[self.theta, self.chi],
            expr=[x_g, y_g], # Les imprecisions de calculs donnent parfois des complexes!
            modules="numexpr") # On l'enregistre une bonne fois pour toutes.
        return globals()["compiled_expressions"]["expr_thetachi_to_gnomonic"]

    def get_expr_gnomonic_to_thetachi(self):
        """
        ** Equation permetant de passer du plan gnomonic a la representation theta chi. **
        """
        if "expr_gnomonic_to_thetachi" in globals()["compiled_expressions"]:
            return globals()["compiled_expressions"]["expr_gnomonic_to_thetachi"]

        # Recherche du vecteur u_q.
        o_oppp = self.gk # Vecteur OO''' == gk car le plan gnomonic est tangent a la shere unitaire.
        u_q = o_oppp + (self.x_gnom*self.gi + self.y_gnom*self.gj) # Vecteur non normalise de la normale au plan christalin.
        u_q = u_q.normalized() # Normale unitaire au plan christalin.

        # Lois de la reflexion.
        u_f = self.u_i - 2*u_q.dot(self.u_i)*u_q # Vecteur unitaire reflechi.

        # Projection et normalisation dans le plan normal a x pour acceder a chi.
        # Projection et normalisation dans le plan normal a y pour acceder a theta.
        u_f = sympy.simplify(u_f) # Permet d'accelerer les calculs par la suite.
        chi = sympy.asin(u_f.dot(self.ry) / (u_f.dot(self.rz)**2 + u_f.dot(self.ry)**2))
        theta = sympy.acos(u_f.dot(self.rx) / (u_f.dot(self.rx)**2 + u_f.dot(self.rz)**2)) / 2

        globals()["compiled_expressions"]["expr_gnomonic_to_thetachi"] = _Lambdify(
            args=[self.x_gnom, self.y_gnom],
            expr=[theta, chi], # Les imprecisions de calculs donnent parfois des complexes!
            modules="numexpr") # On l'enregistre une bonne fois pour toutes.
        return globals()["compiled_expressions"]["expr_gnomonic_to_thetachi"]

    def get_fct_dist_line(self):
        """
        ** Equation de projection de points sur une droite. **
        """
        if "fct_dist_line" in globals()["compiled_expressions"]:
            return globals()["compiled_expressions"]["fct_dist_line"]

        # Creation de la droite.
        theta, dist, x, y = sympy.symbols("theta alpha x y", real=True)
        p = sympy.Point(dist*sympy.cos(theta), dist*sympy.sin(theta)) # Point appartenant a la droite.
        op = sympy.Line(sympy.Point(0, 0), p) # Droite normale a la droite principale.
        line = op.perpendicular_line(p) # C'est la droite principale.

        # Projection des points.
        distance = line.distance(sympy.Point(x, y)) # La distance entre la droite et un point.
        distance = sympy.simplify(distance)

        # Vectorisation de l'expression.
        globals()["compiled_expressions"]["fct_dist_line"] = _Lambdify([theta, dist, x, y], distance, modules="numexpr")
        return globals()["compiled_expressions"]["fct_dist_line"]

    def get_fct_hough(self):
        """
        ** Equation pour la transformee de hough. **
        """
        if "fct_hough" in globals()["compiled_expressions"]:
            return globals()["compiled_expressions"]["fct_hough"]

        xa, ya, xb, yb = sympy.symbols("x_a y_a x_b y_b", real=True)
        u = sympy.Matrix([xa-xb, ya-yb]).normalized()
        x = sympy.Matrix([1, 0])

        # Calcul de la distance entre la droite et l'origine.
        d1 = sympy.Line(sympy.Point(xa, ya), sympy.Point(xb, yb)) # C'est la droite passant par les 2 points.
        dist = d1.distance(sympy.Point(0, 0)) # La distance separant l'origine de la droite.
        dist = sympy.simplify(dist)

        # Calcul de l'angle entre l'axe horizontal et la droite.
        p = d1.projection(sympy.Point(0, 0)) # Le point ou la distance entre ce point de la droite et l'origine est minimale.
        n = p / sympy.sqrt(p.x**2 + p.y**2) # On normalise le point.
        theta_abs = sympy.acos(n.x) # La valeur absolue de theta.
        theta_sign = sympy.sign(n.y) # Si il est negatif c'est que theta < 0, si il est positif alors theta > 0
        theta = theta_abs * theta_sign # Compris entre -pi et +pi
        theta = sympy.simplify(theta)

        # Vectorisation des expressions.
        globals()["compiled_expressions"]["fct_hough"] = _Lambdify([xa, ya, xb, yb], [theta, dist], modules="numexpr")
        return globals()["compiled_expressions"]["fct_hough"]

    def get_fct_inter_line(self):
        """
        ** Equation d'intersection entre 2 droites. **
        """
        if "fct_inter_line" in globals()["compiled_expressions"]:
            return globals()["compiled_expressions"]["fct_inter_line"]

        # Creation des 2 droites.
        theta_1, dist_1, theta_2, dist_2 = sympy.symbols("theta_1, dist_1, theta_2, dist_2", real=True)
        p1 = sympy.Point(dist_1*sympy.cos(theta_1), dist_1*sympy.sin(theta_1)) # Point appartenant a la premiere droite.
        p2 = sympy.Point(dist_2*sympy.cos(theta_2), dist_2*sympy.sin(theta_2)) # Point appartenant a la seconde droite.
        op1 = sympy.Line(sympy.Point(0, 0), p1) # Droite normale a la premiere droite.
        op2 = sympy.Line(sympy.Point(0, 0), p2) # Droite normale a la deuxieme droite.
        line1 = op1.perpendicular_line(p1) # La premiere droite.
        line2 = op2.perpendicular_line(p2) # La seconde droite.

        # Calcul des coordonnes du point d'intersection.
        point = line1.intersection(line2)[0]
        inter_x = sympy.simplify(point.x)
        inter_y = sympy.simplify(point.y)

        # Vectorisation des expressions.
        globals()["compiled_expressions"]["fct_inter_line"] = _Lambdify(
            [theta_1, dist_1, theta_2, dist_2], [inter_x, inter_y], modules="numexpr")
        return globals()["compiled_expressions"]["fct_inter_line"]

    def _hash(self):
        """
        ** Retourne le hash de ce code. **
        """
        return hashlib.md5(
            inspect.getsource(Compilator).encode(encoding="utf-8")
            ).hexdigest()

    def save(self):
        """
        ** Enregistre un fichier contenant les expressions. **

        Enregistre seulement ce qui est present dans ``globals()["compiled_expressions"]``.
        N'ecrase pas l'ancien contenu.
        """
        dirname = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(dirname, "geometry.data")
        self.load() # Recuperation du contenu du fichier.
        content = {
            "hash": self._hash(),
            "expr": globals()["compiled_expressions"]
            }
        with open(file, "wb") as f:
            cloudpickle.dump(content, f)

    def load(self):
        """
        ** Charge si il existe, le fichier contenant les expressions. **

        Deverse les expressions dans le dictionaire: ``globals()["compiled_expressions"]``.
        """
        dirname = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(dirname, "geometry.data")
        
        # if os.path.exists(file):
        #     with open(file, "rb") as f:
        #         try:
        #             content = cloudpickle.load(f)
        #         except ValueError: # Si c'est pas le bon protocol
        #             content = {"hash": None}
        #     if content["hash"] == self._hash(): # Si les donnees sont a jour.
        #         globals()["compiled_expressions"] = {**globals()["compiled_expressions"], **content["expr"]}
        return globals()["compiled_expressions"]


class Transformer(Compilator):
    """
    Permet d'effectuer des transformations geometrique comme jongler
    entre l'espace de la camera et l'espace gnomonique ou encore
    s'ammuser avec la transformee de Hough.
    """
    def __init__(self):
        # Les constantes.
        self.dd = sympy.Symbol("dd", real=True, positive=True) # Distance entre l'origine et le plan de la camera en mm.
        self.xcen, self.ycen = sympy.symbols("xcen ycen", real=True) # Position du point d'incidence normale en pxl par rapport au repere de la camera.
        self.xbet, self.xgam = sympy.symbols("beta gamma", real=True) # Rotation autour x camera, Rotation autour axe incidence normale.
        self.pixelsize = sympy.Symbol("alpha", real=True, positive=True) # Taille des pixels en mm/pxl.

        # Les variables.
        self.x_cam, self.y_cam = sympy.symbols("x_cam y_cam", real=True, positive=True) # Position du pxl dans le repere du plan de la camera.
        self.x_gnom, self.y_gnom = sympy.symbols("x_gnom y_gnom", real=True) # Position des points dans le plan gnomonic.
        self.theta, self.chi = sympy.symbols("theta chi", real=True) # Les angles decrivant le rayon reflechit.

        # Expression des elements du model.
        self.rx = sympy.Matrix([1, 0, 0])
        self.ry = sympy.Matrix([0, 1, 0])
        self.rz = sympy.Matrix([0, 0, 1])

        self.u_i = self.rx # Le rayon de lumiere incident norme parallele a l'axe X dans le repere du cristal.

        self.rot_camera = sympy.rot_axis2(-self.xbet) @ sympy.rot_axis3(self.xgam) # Rotation globale de la camera par rapport au cristal.
        self.ci = self.rot_camera @ -self.ry # Vecteur Xcamera.
        self.cj = self.rot_camera @ self.rx # Vecteur Ycamera.
        self.ck = self.rot_camera @ self.rz # Vecteur Zcamera normal au plan de la camera.

        self.rot_gnom = sympy.rot_axis2(-sympy.pi/4) # Rotation du repere de plan gnomonic par rapport au repere du cristal.
        self.gi = self.rot_gnom @ self.rz # Vecteur Xgnomonic.
        self.gj = self.rot_gnom @ self.ry # Vecteur Ygnomonic.
        self.gk = self.rot_gnom @ -self.rx # Vecteur Zgnomonic normal au plan gnomonic.

        Compilator.__init__(self) # Globalisation des expressions.

        # Les memoires tampon.
        self._fcts_cam_to_gnomonic = collections.defaultdict(lambda: 0) # Fonctions vectorisees avec seulement f(x_cam, y_cam), les parametres sont deja remplaces.
        self._fcts_gnomonic_to_cam = collections.defaultdict(lambda: 0) # Fonctions vectorisees avec seulement f(x_gnom, y_gnom), les parametres sont deja remplaces.
        self._parameters_memory = {} # Permet d'eviter de relire le dictionaire des parametres a chaque fois.

    def cam_to_gnomonic(self, pxl_x, pxl_y, parameters):
        """
        ** Passe des points de la camera dans un plan gnomonic. **

        Notes
        -----
        * Les calculs sont effectues avec des float32 pour des histoire de performance.
        * Cette methode va 1.8 a 3.3 fois plus vite que celle de LaueTools.
        * Contrairement a LaueTools, cette methode prend en charge les vecteurs a n dimenssions.

        Parameters
        ----------
        pxl_x : float, int ou np.ndarray
            Coordonnee.s du.des pxl.s selon l'axe x dans le repere de la camera. (en pxl)
        pxl_y : float, int ou np.ndarray
            Coordonnee.s du.des pxl.s selon l'axe y dans le repere de la camera. (en pxl)
        parameters : dict
            Le dictionaire issue de la fonction ``laue.tools.parsing.extract_parameters``.

        Returns
        -------
        float ou np.ndarray
            Le.s coordonnee.s x puis y du.des point.s dans le plan gnomonic eprimee.s en mm.
            Les dimenssions du tableau de sortie sont les memes que celle du tableau d'entree.
            shape de sortie = (2, *shape_d_entree)

        Examples
        -------
        >>> import numpy as np
        >>> from laue.geometry import Transformer
        >>> from laue.tools.parsing import extract_parameters
        >>> parameters = extract_parameters(dd=70, bet=.0, gam=.0, pixelsize=.08, x0=1024, y0=1024)
        >>> transformer = Transformer()
        >>> x_cam, y_cam = np.linspace(0, 2048, 5), np.linspace(0, 2048, 5)
        >>>
        >>> transformer.cam_to_gnomonic(x_cam, y_cam, parameters).astype(np.float16)
        array([[-0.5127, -0.3064,  0.    ,  0.1676,  0.1342],
               [ 0.4033,  0.287 , -0.    , -0.4832, -0.9385]], dtype=float16)
        >>>
        """
        assert isinstance(pxl_x, (float, int, np.ndarray)), \
            f"'pxl_x' can not be of type {type(pxl_x).__name__}."
        assert isinstance(pxl_y, (float, int, np.ndarray)), \
            f"'pxl_y' can not be of type {type(pxl_y).__name__}."
        assert type(pxl_x) == type(pxl_y), \
            f"Les 2 types sont differents: {type(pxl_x).__name__} vs {type(pxl_y).__name__}."
        if isinstance(pxl_x, np.ndarray):
            assert pxl_x.shape == pxl_y.shape, \
                f"Ils n'ont pas le meme taille: {pxl_x.shape} vs {pxl_y.shape}."
        assert isinstance(parameters, dict), ("Les parametres doivent founis "
            f"dans un dictionaire, pas dans un {type(parameters).__name__}")
        assert set(parameters) == {"dd", "xbet", "xgam", "xcen", "ycen", "pixelsize"}, \
            ("Les clefs doivent etres 'dd', 'xbet', 'xgam', 'xcen', 'ycen' et 'pixelsize'. "
            f"Or les clefs sont {set(parameters)}.")
        assert all(isinstance(v, (float, int)) for v in parameters.values()), \
            "La valeurs des parametres doivent toutes etre des nombres."

        if isinstance(pxl_x, np.ndarray):
            pxl_x, pxl_y = pxl_x.astype(np.float32, copy=False), pxl_y.astype(np.float32, copy=False)
        else:
            pxl_x, pxl_y = np.float32(pxl_x), np.float32(pxl_y)

        hash_param = self._hash_parameters(parameters) # Recuperation de la 'signature' des parametres.
        optimized_func = self._fcts_cam_to_gnomonic[hash_param] # On regarde si il y a une fonction deja optimisee.

        if isinstance(optimized_func, int): # Si il n'y a pas de fonction optimisee.
            nbr_access = optimized_func # Ce qui est enregistre et le nombre de fois que l'on a chercher a y acceder.
            self._fcts_cam_to_gnomonic[hash_param] += 1 # Comme on cherche a y acceder actuelement, on peut incrementer le compteur.
            if nbr_access + 1 == 4: # Si c'est la 4 eme fois qu'on accede a la fonction.
                self.compile(parameters) # On optimise la fonction.
            else: # Si ce n'est pas encore le moment de perdre du temps a optimiser.
                return self.get_expr_cam_to_gnomonic()(pxl_x, pxl_y,
                    parameters["dd"], parameters["xcen"], parameters["ycen"],
                    parameters["xbet"], parameters["xgam"], parameters["pixelsize"])

        return self._fcts_cam_to_gnomonic[hash_param](pxl_x, pxl_y)

    def dist_line(self, theta_vect, dist_vect, x_vect, y_vect):
        """
        ** Calcul les distances projetees des points sur une droite. **

        Notes
        -----
        Pour des raisons de performances, travail avec des float32.

        Parameters
        ----------
        theta_vect : np.ndarray
            Les angles des droites normales aux droites principales.
            shape = ``(*nbr_droites)``
        dist_vect : np.ndarray
            Les distances entre les droites et l'origine.
            shape = ``(*nbr_droites)``
        x_vect : np.ndarray
            L'ensemble des coordonnees x des points.
            shape = ``(*nbr_points)``
        y_vect : np.ndarray
            L'ensemble des coordonnees y des points.
            shape = ``(*nbr_points)``

        Returns
        -------
        np.ndarray
            Les distances des projetees des points sur chacunes des droites.
            shape = ``(*nbr_droites, *nbr_points)``

        Examples
        --------
        >>> import numpy as np
        >>> from laue.geometry import Transformer
        >>> from laue.tools.parsing import extract_parameters
        >>> parameters = extract_parameters(dd=70, bet=.0, gam=.0, pixelsize=.08, x0=1024, y0=1024)
        >>> transformer = Transformer()
        >>>
        >>> lines = (np.array([0, np.pi/2]), np.array([1, 1])) # Horizontale et verticale passant par (1, 1)
        >>> points = (np.array([0, 1, 3, 0]), np.array([0, 1, 3, 1]))
        >>> np.round(transformer.dist_line(*lines, *points))
        array([[1., 0., 2., 1.],
               [1., 0., 2., 0.]], dtype=float32)
        >>>
        """
        assert isinstance(theta_vect, np.ndarray), \
            f"'theta_vect' has to be of type np.ndarray, not {type(theta_vect).__name__}."
        assert isinstance(dist_vect, np.ndarray), \
            f"'dist_vect' has to be of type np.ndarray, not {type(dist_vect).__name__}."
        assert theta_vect.shape == dist_vect.shape, \
            f"Les 2 parametres de droite doivent avoir la meme taille: {theta_vect.shape} vs {dist_vect.shape}."
        assert isinstance(x_vect, np.ndarray), \
            f"'x_vect' has to be of type np.ndarray, not {type(x_vect).__name__}."
        assert isinstance(y_vect, np.ndarray), \
            f"'y_vect' has to be of type np.ndarray, not {type(y_vect).__name__}."
        assert x_vect.shape == y_vect.shape, \
            f"Les 2 coordonnees des points doivent avoir la meme shape: {x_vect.shape} vs {y_vect.shape}."

        theta_vect, dist_vect = theta_vect.astype(np.float32, copy=False), dist_vect.astype(np.float32, copy=False)
        x_vect, y_vect = x_vect.astype(np.float32, copy=False), y_vect.astype(np.float32, copy=False)

        nbr_droites = theta_vect.shape
        nbr_points = x_vect.shape

        # Ca ne vaut pas le coup de paralleliser car c'est tres rapide.
        func = self.get_fct_dist_line()
        result = np.array([func(theta, dist, x_vect, y_vect)
                           for theta, dist
                           in zip(theta_vect.ravel(), dist_vect.ravel())
                          ], dtype=np.float32).reshape((*nbr_droites, *nbr_points))
        return result

    def gnomonic_to_cam(self, gnom_x, gnom_y, parameters):
        """
        ** Passe des points du plan gnomonic vers la camera. **

        Parameters
        ----------
        gnom_x : float ou np.ndarray
            Coordonnee.s du.des point.s selon l'axe x du repere du plan gnomonic. (en mm)
        gnom_y : float ou np.ndarray
            Coordonnee.s du.des point.s selon l'axe y du repere du plan gnomonic. (en mm)
        parameters : dict
            Le dictionaire issue de la fonction ``laue.tools.parsing.extract_parameters``.

        Returns
        -------
        coords : np.ndarray
            * Le.s coordonnee.s x puis y du.des point.s dans le plan de la camera. (en pxl)
            * shape = (2, ...)

        Examples
        -------
        >>> import numpy as np
        >>> from laue.geometry import Transformer
        >>> from laue.tools.parsing import extract_parameters
        >>> parameters = extract_parameters(dd=70, bet=.0, gam=.0, pixelsize=.08, x0=1024, y0=1024)
        >>> transformer = Transformer()
        >>> gnom_x, gnom_y = np.random.normal(size=(2, 10, 4))
        >>>
        >>> transformer.gnomonic_to_cam(gnom_x, gnom_y, parameters).shape
        (2, 10, 4)
        >>> transformer.gnomonic_to_cam(.0, .0, parameters)
        array([1024., 1024.])
        >>>
        """
        assert isinstance(gnom_x, (float, int, np.ndarray)), \
            f"'gnom_x' can not be of type {type(gnom_x).__name__}."
        assert isinstance(gnom_y, (float, int, np.ndarray)), \
            f"'gnom_y' can not be of type {type(gnom_y).__name__}."
        assert type(gnom_x) == type(gnom_y), \
            f"Les 2 types sont differents: {type(gnom_x).__name__} vs {type(gnom_y).__name__}."
        if isinstance(gnom_x, np.ndarray):
            assert gnom_x.shape == gnom_y.shape, \
                f"Ils n'ont pas la meme taille: {gnom_x.shape} vs {gnom_y.shape}."
        assert isinstance(parameters, dict), ("Les parametres doivent founis "
            f"dans un dictionaire, pas dans un {type(parameters).__name__}")
        assert set(parameters) == {"dd", "xbet", "xgam", "xcen", "ycen", "pixelsize"}, \
            ("Les clefs doivent etres 'dd', 'xbet', 'xgam', 'xcen', 'ycen' et 'pixelsize'. "
            f"Or les clefs sont {set(parameters)}.")
        assert all(isinstance(v, (float, int)) for v in parameters.values()), \
            "La valeurs des parametres doivent toutes etre des nombres."

        if isinstance(gnom_x, np.ndarray):
            gnom_x, gnom_y = gnom_x.astype(np.float32, copy=False), gnom_y.astype(np.float32, copy=False)
        else:
            gnom_x, gnom_y = np.float32(gnom_x), np.float32(gnom_y)

        hash_param = self._hash_parameters(parameters) # Recuperation de la 'signature' des parametres.
        optimized_func = self._fcts_gnomonic_to_cam[hash_param] # On regarde si il y a une fonction deja optimisee.

        if isinstance(optimized_func, int): # Si il n'y a pas de fonction optimisee.
            nbr_access = optimized_func # Ce qui est enregistre et le nombre de fois que l'on a chercher a y acceder.
            self._fcts_gnomonic_to_cam[hash_param] += 1 # Comme on cherche a y acceder actuelement, on peut incrementer le compteur.
            if nbr_access + 1 == 4: # Si c'est la 4 eme fois qu'on accede a la fonction.
                self.compile(parameters) # On optimise la fonction.
            else: # Si ce n'est pas encore le moment de perdre du temps a optimiser.
                return self.get_expr_gnomonic_to_cam()(gnom_x, gnom_y,
                    parameters["dd"], parameters["xcen"], parameters["ycen"],
                    parameters["xbet"], parameters["xgam"], parameters["pixelsize"])

        return self._fcts_gnomonic_to_cam[hash_param](gnom_x, gnom_y)

    def hough(self, x_vect, y_vect):
        r"""
        ** Transformee de hough avec des droites. **

        Note
        ----
        * Pour des raisons de performances, les calculs se font sur des float32.
        * Les indices sont agences selon l'ordre defini par la fonction ``comb2ind``.

        Parameters
        ----------
        x_vect : np.ndarray
            L'ensemble des coordonnees x des points de shape: (*over_dims, nbr_points)
        y_vect : np.ndarray
            L'ensemble des coordonnees y des points de shape: (*over_dims, nbr_points)

        Returns
        -------
        theta : np.ndarray
            * Les angles au sens trigomometrique des vecteurs reliant l'origine
            ``O`` (0, 0) au point ``P`` appartenant a la droite tel que ``||OP||``
            soit la plus petite possible.
            * theta € [-pi, pi]
            * shape = ``(*over_dims, n*(n-1)/2)``
        dist : np.ndarray
            * Ce sont les normes des vecteur ``OP``.
            * dist € [0, +oo].
            * shape = ``(*over_dims, n*(n-1)/2)``
        * Ces 2 grandeurs sont concatenees dans une seule array de
        shape = ``(2, *over_dims, n*(n-1)/2)``

        Examples
        --------
        >>> import numpy as np
        >>> from laue import geometry
        >>> transformer = geometry.Transformer()
        >>> x, y = np.random.normal(size=(2, 6))
        >>> transformer.hough(x, y).shape
        (2, 15)
        >>>
        >>> x, y = np.random.normal(size=(2, 4, 5, 6))
        >>> transformer.hough(x, y).shape
        (2, 4, 5, 15)
        >>> 
        """
        assert isinstance(x_vect, np.ndarray), \
            f"'x_vect' has to be of type np.ndarray, not {type(x_vect).__name__}."
        assert isinstance(y_vect, np.ndarray), \
            f"'y_vect' has to be of type np.ndarray, not {type(y_vect).__name__}."
        assert x_vect.shape == y_vect.shape, \
            f"Les 2 entrees doivent avoir la meme taille: {x_vect.shape} vs {y_vect.shape}."

        n = x_vect.shape[-1]
        x_vect, y_vect = x_vect.astype(np.float32, copy=False), y_vect.astype(np.float32, copy=False)

        xa = np.concatenate([np.repeat(x_vect[..., i, np.newaxis], n-i-1, axis=-1) for i in range(n-1)], axis=-1)
        ya = np.concatenate([np.repeat(y_vect[..., i, np.newaxis], n-i-1, axis=-1) for i in range(n-1)], axis=-1)
        xb = np.concatenate([x_vect[..., i+1:] for i in range(n-1)], axis=-1)
        yb = np.concatenate([y_vect[..., i+1:] for i in range(n-1)], axis=-1)

        return np.nan_to_num(
            self.get_fct_hough()(xa, ya, xb, yb),
            copy=False,
            nan=0.0)

    def hough_reduce(self, theta_vect, dist_vect, *, nbr=4, tol=0.018):
        """
        ** Regroupe des droites ressemblantes. **

        Notes
        -----
        * Cette methode est concue pour traiter les donnees issues de ``laue.geometry.Transformer.hough``.
        * La metrique utilise est la distance euclidiene sur un cylindre ferme sur theta.
        * En raison de performance et de memoire, les calculs se font sur des float32.

        Parameters
        ----------
        theta_vect : np.ndarray
            * Vecteur des angles compris entre [-pi, pi].
            * shape = ``(*over_dims, nbr_inter)``
        dist_vect : np.ndarray
            * Vecteur des distances des droites a l'origine comprises [0, +oo].
            * shape = ``(*over_dims, nbr_inter)``
        tol : float
            La distance maximal separant 2 points dans l'espace de hough reduit,
            (ie la difference entre 2 droites dans l'espace spacial) tel que les points
            se retrouvent dans le meme cluster. Plus ce nombre est petit, plus les points
            doivent etre bien alignes. C'est une sorte de tolerance sur l'alignement.
        nbr : int
            C'est le nombre minimum de points presque alignes pour que
            l'on puisse considerer la droite qui passe par ces points.
            Par defaut, les droites qui ne passent que par 4 points et plus sont retenues.

        Returns
        -------
        np.ndarray(dtype=float32), np.ndarray(dtype=object)
            * Ce sont les centres des clusters pour chaque 'nuages de points'. Cela correspond
            aux angles et aux distances qui caracterisent chaque droites.
            * Si les parametres d'entres sont des vecteurs 1d, le resultat sera une array
            numpy contenant les **angles** puis les **distances**. Donc de shape = ``(2, nbr_clusters)``
            * Si les parametres d'entres sont en plusieur dimensions, (representes plusieur
            nuages de points indepandant), alors le resultat sera une array d'objet de
            shape = ``(*over_dims)``. Chaque objet est lui meme un array, resultat recursif
            de l'appel de cette fonction sur le nuage de points unique correspondant.

        Examples
        --------
        >>> import numpy as np
        >>> from laue import geometry
        >>> transformer = geometry.Transformer()

        Type de retour ``np.float32`` vs ``object``.
        >>> x, y = (np.array([ 1.,  2.,  3.,  0., -1.]),
        ...         np.array([ 0.,  1.,  1., -1.,  1.]))
        >>> theta, dist = transformer.hough(x, y)
        >>> transformer.hough_reduce(theta, dist, nbr=3)
        array([[-0.7853982 ,  1.5707964 ],
               [ 0.70710677,  1.        ]], dtype=float32)
        >>> transformer.hough_reduce(theta.reshape((1, -1)), dist.reshape((1, -1)), nbr=3)
        array([array([[-0.7853982 ,  1.5707964 ],
                      [ 0.70710677,  1.        ]], dtype=float32)], dtype=object)
        >>>

        Les dimensions de retour.
        >>> x, y = (np.random.normal(size=(6, 5, 4)),
        ...         np.random.normal(size=(6, 5, 4)))
        >>> theta, dist = transformer.hough(x, y)
        >>> transformer.hough_reduce(theta, dist).shape
        (6, 5)
        >>> 
        """
        assert isinstance(theta_vect, np.ndarray), \
            f"'theta_vect' has to be of type np.ndarray, not {type(theta_vect).__name__}."
        assert isinstance(dist_vect, np.ndarray), \
            f"'dist_vect' has to be of type np.ndarray, not {type(dist_vect).__name__}."
        assert theta_vect.shape == dist_vect.shape, \
            f"Les 2 entrees doivent avoir la meme taille: {theta_vect.shape} vs {dist_vect.shape}."
        assert theta_vect.ndim >= 1, "La matrice ne doit pas etre vide."
        assert isinstance(tol, float), f"'tol' has to be a float, not a {type(tol).__name__}."
        assert 0.0 < tol <= 0.5, ("Les valeurs coherentes de 'tol' se trouvent entre "
            f"]0, 1/2], or tol vaut {tol}, ce qui sort de cet intervalle.")
        assert isinstance(nbr, int), f"'nbr' has to be an integer, not a {type(nbr).__name__}."
        assert 2 < nbr, f"2 points sont toujours alignes! Vous ne pouvez pas choisir nbr={nbr}."

        # On fait la conversion des le debut pour un gain de temps.
        theta_vect, dist_vect = theta_vect.astype(np.float32, copy=False), dist_vect.astype(np.float32, copy=False)

        *over_dims, nbr_inter = theta_vect.shape # Recuperation des dimensions.
        nbr = (nbr*(nbr-1))/2 # On converti le nombre de points alignes en nbr de segments.

        # On commence par travailler avec les donnees reduites.
        theta_theo_std = math.pi / math.sqrt(3) # Variance theorique = (math.pi - -math.pi)**2 / 12
        dist_std = np.nanstd(dist_vect, axis=-1) # Ecart type non biaise (sum(*over_dims)/N), shape: (*over_dims)
        dist_vect = (dist_vect * theta_theo_std
            / np.repeat(dist_std[..., np.newaxis], nbr_inter, axis=-1)) # Les distances quasi reduites.
        
        # Extraction des clusters.
        if not len(over_dims): # Cas des tableaux 1d.
            return self._clustering_1d(theta_vect, dist_vect, dist_std, tol, nbr)

        clusters = np.empty(np.prod(over_dims, dtype=int), dtype=object) # On doit d'abord creer un tableau d'objet 1d.
        if multiprocessing.current_process().name == "MainProcess" and np.prod(over_dims) >= os.cpu_count(): # Si ca vaut le coup de parraleliser:
            ser_self = cloudpickle.dumps(self) # Strategie car 'pickle' ne sais pas faire ca.
            from laue.tools.multi_core import pickleable_method
            with multiprocessing.Pool() as pool:
                clusters[:] = pool.map(
                    pickleable_method, # Car si il y a autant de cluster dans chaque image,
                    (                   # numpy aurait envi de faire un tableau 2d plutot qu'un vecteur de listes.
                        (
                            Transformer._clustering_1d,
                            ser_self,
                            {"theta_vect_1d":theta, "dist_vect_1d":dist, "std":std, "tol":tol, "nbr":nbr}
                        )
                        for theta, dist, std
                        in zip(
                            theta_vect.reshape((-1, nbr_inter)),
                            dist_vect.reshape((-1, nbr_inter)),
                            np.nditer(dist_std)
                        )
                    )
                )
        else:
            clusters[:] = [self._clustering_1d(theta, dist, std, tol, nbr)
                           for theta, dist, std in zip(
                                    theta_vect.reshape((-1, nbr_inter)),
                                    dist_vect.reshape((-1, nbr_inter)),
                                    np.nditer(dist_std))] 
        clusters = clusters.reshape(over_dims) # On redimensione a la fin de sorte a garentir les dimensions.

        return clusters

    def inter_lines(self, theta_vect, dist_vect):
        r"""
        ** Calcul les points d'intersection entre les droites. **

        Notes
        -----
        * Cette methode est concue pour traiter les donnees issues de ``laue.geometry.Transformer.hough``.
        * En raison de performance et de memoire, les calculs se font sur des float32.
        * Les indices sont agences selon l'ordre defini par la fonction ``comb2ind``.

        Parameters
        ----------
        theta_vect : np.ndarray
            * Vecteur des angles compris entre [-pi, pi].
            * shape = (*over_dims, nbr_droites)
        dist_vect : np.ndarray
            * Vecteur des distances des droites a l'origine comprises [0, +oo].
            * shape = (*over_dims, nbr_droites)

        Returns
        -------
        np.ndarray(dtype=float32)
            * Dans le dommaine spatial et non pas le domaine de hough, cherche
            les intersections des droites. Il y a ``n*(n-1)/2`` intersections, n etant
            le nombre de droites. donc la complexite de cette methode est en ``o(n**2)``.
            * Si les vecteurs d'entre sont des vecteurs 1d (ie ``*over_dims == ()``), 
            Seront retournes le vecteur d'intersection selon l'axe x et le vecteur
            des intersections selon l'axe y. Ces 2 vecteurs de meme taille sont concatenes
            sous la forme d'une matrice de shape = ``(2, n*(n-1)/2)``.
            * Si les vecteurs d'entre sont en plusieurs dimensions, seul les droites de la
            derniere dimensions se retrouvent dans la meme famille. Tous comme pour les
            vecteurs 1d, on trouve d'abord les intersections selon x puis en suite selon y.
            La shape du tenseur final est donc: ** shape = ``(2, *over_dims, n*(n-1)/2)`` **.

        Examples
        -------
        >>> import numpy as np
        >>> from laue.geometry import Transformer
        >>> transformer = Transformer()
        >>> np.random.seed(0)
        >>> x, y = np.random.normal(size=(2, 4, 5, 6))
        >>> theta, dist = transformer.hough(x, y)
        >>> theta.shape
        (4, 5, 15)
        >>> transformer.inter_lines(theta, dist).shape
        (2, 4, 5, 105)
        >>>
        """
        assert isinstance(theta_vect, np.ndarray), \
            f"'theta_vect' has to be of type np.ndarray, not {type(theta_vect).__name__}."
        assert isinstance(dist_vect, np.ndarray), \
            f"'dist_vect' has to be of type np.ndarray, not {type(dist_vect).__name__}."
        assert theta_vect.shape == dist_vect.shape, \
            f"Les 2 entrees doivent avoir la meme taille: {theta_vect.shape} vs {dist_vect.shape}."
        assert theta_vect.ndim >= 1, "La matrice ne doit pas etre vide."
        assert theta_vect.shape[-1] >= 2, \
            f"Il doit y avoir au moins 2 droites par famille, pas {theta_vect.shape[-1]}."

        theta_vect, dist_vect = theta_vect.astype(np.float32, copy=False), dist_vect.astype(np.float32, copy=False)
        n = theta_vect.shape[-1]

        theta_1 = np.concatenate([np.repeat(theta_vect[..., i, np.newaxis], n-i-1, axis=-1) for i in range(n-1)], axis=-1)
        dist_1 = np.concatenate([np.repeat(dist_vect[..., i, np.newaxis], n-i-1, axis=-1) for i in range(n-1)], axis=-1)
        theta_2 = np.concatenate([theta_vect[..., i+1:] for i in range(n-1)], axis=-1)
        dist_2 = np.concatenate([dist_vect[..., i+1:] for i in range(n-1)], axis=-1)

        return self.get_fct_inter_line()(theta_1, dist_1, theta_2, dist_2)

    def _clustering_1d(self, theta_vect_1d, dist_vect_1d, std, tol, nbr):
        """
        ** Help for hough_reduce. **

        * Permet de trouver les clusters d'un nuage de points.
        * La projection 3d, bien que moins realiste, est 20% plus rapide que la distance reele.
        """
        from sklearn.cluster import DBSCAN

        THETA_STD = np.float32(math.pi / math.sqrt(3))
        WEIGHT = 0.65 # 0 => tres souple sur les angles, 1=> tres souple sur les distances.

        # On retire les droites aberantes.
        mask_to_keep = np.isfinite(theta_vect_1d) & np.isfinite(dist_vect_1d)
        if not mask_to_keep.any(): # Si il ne reste plus rien.
            return np.array([], dtype=np.float32)
        theta_vect_1d, dist_vect_1d = theta_vect_1d[mask_to_keep], dist_vect_1d[mask_to_keep]

        # On passe dans un autre repere de facon a ce que -pi et pi se retrouvent a cote.
        if numexpr is not None:
            theta_x = numexpr.evaluate("2*WEIGHT*cos(theta_vect_1d)")
            theta_y = numexpr.evaluate("2*WEIGHT*sin(theta_vect_1d)")
        else:
            theta_x, theta_y = 2*WEIGHT*np.cos(theta_vect_1d), 2*WEIGHT*np.sin(theta_vect_1d)

        # Recherche des clusters.
        n_jobs = -1 if multiprocessing.current_process().name == "MainProcess" else 1
        db_res = DBSCAN(eps=tol, min_samples=nbr, n_jobs=n_jobs).fit(
            np.vstack((theta_x, theta_y, 2*(1-WEIGHT)*dist_vect_1d)).transpose())

        # Mise en forme des clusters.
        clusters_dict = collections.defaultdict(lambda: [])
        keep = db_res.labels_ != -1 # Les indices des clusters a garder.
        for x_cyl, y_cyl, dist, group in zip(
                theta_x[keep], theta_y[keep], dist_vect_1d[keep], db_res.labels_[keep]):
            clusters_dict[group].append((x_cyl, y_cyl, dist))

        theta = np.array([np.arccos(cluster[:, 0].mean()/(2*WEIGHT))*np.sign(cluster[:, 1].sum())
                    for cluster in map(np.array, clusters_dict.values())],
                    dtype=np.float32)
        dist = np.array([cluster[:, 2].mean()
                        for cluster in map(np.array, clusters_dict.values())],
                    dtype=np.float32) * std / THETA_STD
        return np.array([theta, dist], dtype=np.float32)

    def _hash_parameters(self, parameters):
        """
        ** Hache le dictionaire des parametres. **

        * Il n'y a pas de verification pour des histoires de performances.

        Parameters
        ----------
        parameters : dict
            Dictionaire des parametres issues de ``laue.tools.parsing.extract_parameters``.

        Returns
        -------
        int
            Un identifiant tq 2 dictionaires identiques renvoient le meme id.
        """
        return hash(( # Il faut imperativement garantir l'ordre.
            parameters["dd"],
            parameters["xcen"],
            parameters["ycen"],
            parameters["xbet"],
            parameters["xgam"],
            parameters["pixelsize"]))


def comb2ind(ind1, ind2, n):
    """
    ** Transforme 2 indices en un seul. **

    Note
    ----
    * Bijection de ``ind2comb``.
    * Peut etre utile pour les methodes ``laue.geometry.Transformer.hough``
    et ``laue.geometry.Transformer.inter_lines``.

    Parameters
    ----------
    ind1 : int ou np.ndarray(dtype=int)
        L'indice du premier element: ``0 <= ind1``.
    ind2 : int ou np.ndarray(dtype=int)
        L'indice du second element: ``ind1 < ind2``.
    n : int
        Le nombre de symboles : ``2 <= n and ind2 < n``.

    Returns
    -------
    int, np.ndarray(dtype=int)
        Le nombre de mots contenant exactement 2 symboles dans un
        alphabet de cardinal ``n``. Sachant que le deuxieme symbole
        est stricement superieur au premier, et que les mots sont
        generes avec le comptage naturel (representation des nombres
        en base n).

    Examples
    -------
    >>> from laue.geometry import comb2ind
    >>> comb2ind(0, 1, n=6)
    0
    >>> comb2ind(0, 2, n=6)
    1
    >>> comb2ind(0, 5, n=6)
    4
    >>> comb2ind(1, 2, n=6)
    5
    >>> comb2ind(4, 5, n=6)
    14
    """
    assert isinstance(ind1, (int, np.ndarray)), \
        f"'ind1' can not being of type {type(ind1).__name__}."
    assert isinstance(ind2, (int, np.ndarray)), \
        f"'ind2' can not being of type {type(ind2).__name__}."
    assert isinstance(n, int), f"'n' has to ba an integer, not a {type(n).__name__}."
    if isinstance(ind1, np.ndarray):
        assert ind1.dtype == int, f"'ind1' must be integer, not {str(ind1.dtype)}."
        assert (ind1 >= 0).all(), "Tous les indices doivent etres positifs."
    else:
        assert ind1 >= 0, "Les indices doivent etre positifs."
    if isinstance(ind2, np.ndarray):
        assert ind2.dtype == int, f"'ind2' must be integer, not {str(ind2.dtype)}."
        assert ind1.shape == ind2.shape, ("Si les indices sont des arrays, elles doivent "
            f"toutes 2 avoir les memes dimensions. {ind1.shape} vs {ind2.shape}.")
        assert (ind2 > ind1).all(), ("Les 2ieme indices doivent "
            "etres strictement superieur aux premiers.")
        assert (ind2 < n).all(), "Vous aimez un peu trop les 'index out of range'."
    else:
        assert ind2 > ind1, ("Le 2ieme indice doit etre strictement superieur au premier. "
            f"{ind1} vs {ind2}.")
        assert ind2 < n, "Vous aimez un peu trop les 'index out of range'."

    return n*ind1 - (ind1**2 + 3*ind1)//2 + ind2 - 1

def ind2comb(comb, n):
    """
    ** Eclate un rang en 2 indices **

    Notes
    -----
    * Bijection de ``comb2ind``.
    * Peut etre utile pour les methodes ``laue.geometry.Transformer.hough``
    et ``laue.geometry.Transformer.inter_lines``.
    * Risque de donner de faux resultats pour n trop grand.

    Parameters
    ----------
    comb : int ou np.ndarray(dtype=int)
        L'indice, ie le (comb)ieme mot constitue 2 de symbols
        ajence comme decrit dans ``comb2ind``.
    n : int
        Le nombre de symboles.

    Returns
    -------
    int ou np.ndarray(dtype=int)
        Le premier des 2 indicices (``ind1``).
    int ou np.ndarray(dtype=int)
        Le second des 2 indicices (``ind2``). De sorte que ``comb2ind(ind1, ind2) == comb``.

    Examples
    --------
    >>> from laue.geometry import ind2comb
    >>> ind2comb(0, n=6)
    (0, 1)
    >>> ind2comb(1, n=6)
    (0, 2)
    >>> ind2comb(4, n=6)
    (0, 5)
    >>> ind2comb(5, n=6)
    (1, 2)
    >>> ind2comb(14, n=6)
    (4, 5)
    >>>
    """
    assert isinstance(comb, (int, np.ndarray)), \
        f"'comb' can not being of type {type(comb).__name__}."
    assert isinstance(n, int), f"'n' has to ba an integer, not a {type(n).__name__}."
    if isinstance(comb, np.ndarray):
        assert comb.dtype == int, f"'comb' must be integer, not {str(comb.dtype)}."
        assert (comb >= 0).all(), "Tous les indices doivent etres positifs."
        assert (comb < n*(n-1)/2).all(), (f"Dans un alphabet a {n} symboles, il ne peut y a voir "
            f"que {n*(n-1)/2} mots. Le mot d'indice {comb.max()} n'existe donc pas!")
    else:
        assert comb >= 0, "Les indices doivent etre positifs."
        assert comb < n*(n-1)/2, (f"Dans un alphabet a {n} symboles, il ne peut y a voir "
            f"que {n*(n-1)/2} mots. Le mot d'indice {comb} n'existe donc pas!")

    homogeneous_int = lambda x: x.astype(int) if isinstance(x, np.ndarray) else int(x)
    ind1 = homogeneous_int(np.ceil(n - np.sqrt(-8*comb + 4*n**2 - 4*n - 7)/2 - 3/2))
    ind2 = comb + (ind1**2 + 3*ind1)//2 - ind1*n + 1
    return ind1, ind2


class _Lambdify:
    """
    Vectorize une expression sympy.
    """
    def __init__(self, args, expr, modules=None):
        """
        Notes
        -----
        * Est plus efficace si le module ``numexpr`` est installe.
        * Force a convertir le type de retour en float.

        Parameters
        ----------
        :param parameters: Les parametres constant a remplacer.
        For all other parameters, they are exactly the same as  ``sympy.lambdify``.
        """
        self.sub_var_cmp = 0 # Compteur qui permet de creer plein de variables sympy.

        self.args = args
        self.modules = modules
        self.expr = self._expr2sym(expr)
        self.operations = collections.OrderedDict() # Contient la suite d'operations.
        self._gather(args, self.expr) # Remplissage des etapes de calcul.
        # self.operations["final_result"] = (sympy.lambdify(args, self.expr), self.expr, [])

    def __call__(self, *args):
        """
        Evalue la fonction.
        """
        intermediate = collections.OrderedDict(zip(self.args, args)) # Ce sont les resultats intermediaires.
        for var, (func, _, useless_vars) in self.operations.items():
            intermediate[var] = func(*intermediate.values())
            for var_u in useless_vars:
                del intermediate[var_u]
        return np.real(intermediate["final_result"])

    def __repr__(self):
        """
        Donne une representation evaluable de soi.
        """
        return f"_Lambdify({self.args}, {self.expr})"

    def __str__(self):
        """
        Retourne le code deplie.
        """
        code = f"def _lambdifygenerated({', '.join(map(str, self.args))}):\n"
        for var, (_, expr, useless_vars) in self.operations.items():
            code += f"\t{var} = {expr}\n"
            if useless_vars:
                code += f"\tdel {', '.join(map(str, useless_vars))}\n"
        code += "\treturn final_result\n"
        return code

    def get_expr(self):
        """
        Recupere l'expression brute sympy.

        :returns: L'expression sympy liee au parametre 'expr' de self.__init__.
        :rtype: sympy.core.basic.Basic
        """
        return self.expr

    def _expr2sym(self, expr):
        """
        Transforme 'expr' en expression sympy.

        :param expr: Expression sympy, tuple, liste, str
        :returns: Une expression sympy complete.
        """
        assert isinstance(expr, (sympy.core.basic.Basic, list, tuple, str)), \
            f"'expression' has to be list, tuple or sympy expr, not {type(expression).__name__}."

        if isinstance(expr, sympy.core.basic.Basic):
            return expr

        if isinstance(expr, str):
            standard_transformations = sympy.parsing.sympy_parser.standard_transformations
            implicit_multiplication_application = sympy.parsing.sympy_parser.implicit_multiplication_application
            transformations = (standard_transformations + (implicit_multiplication_application,))
            return self._expr2sym(sympy.parse_expr(expr, transformations=transformations))

        return sympy.Tuple(*[self._expr2sym(child) for child in expr])

    def _gather(self, variables, expr):
        """
        Supprime la redondance.
        """
        if isinstance(expr, sympy.core.basic.Atom): # Si il n'y a rien a symplifier.
            self.operations["final_result"] = (sympy.lambdify(variables, expr), expr, [])
            return

        complete_hist = self._hist_sub_expr(expr) # Histograme complet des descendances.
        max_red = max(complete_hist.values()) # Redondance maximale.
        if max_red == 1: # Si il n'y a pas de redondance.
            self.operations["final_result"] = (sympy.lambdify(variables, expr), expr, [])
            return

        max_red_expr = [expr for expr, red in complete_hist.items() if red == max_red] # Seul les elements les plus redondants.

        dephs = [self._len(expr) for expr in max_red_expr] # 3 La profondeur respective de chaque arbre.
        sub_expr = max_red_expr[np.argmax(dephs)] # L'expression de l'element a remplacer.
        self.sub_var_cmp += 1 # On incremente le conteur pour creer une nouvelle variable unique.
        sub_var = sympy.Symbol(f"subvar_{self.sub_var_cmp}") # Nom de la variable intermediaire.
        expr = expr.xreplace({sub_expr: sub_var})

        useless_vars = set(variables) - expr.free_symbols # Se sont les variables inutiles par la suite.
        try:
            self.operations[sub_var] = (sympy.lambdify(variables, sub_expr, modules=self.modules), sub_expr, useless_vars) # ajoute de l'etape de calcul.
        except (ImportError, TypeError):
            self.operations[sub_var] = (sympy.lambdify(variables, sub_expr), sub_expr, useless_vars)
        variables = [var for var in variables if var not in useless_vars] # On vire les variables inutiles.

        return self._gather(variables+[sub_var], expr) # On simplifie, puis on recommence.

    def _hist_sub_expr(self, expr):
        """
        Fait l'histograme des enfants.

        * Ne fait pas de verification pour plus de permormances.
        * Ne compte pas les symbols et les nombres.

        :param expr: Expression sympy derivee de sympy.core.basic.Basic
        :returns: Le dictionaire qui a chaque enfant et descendant en general,
            associ son nombre d'apparition.
        """
        hist = collections.defaultdict(lambda: 0)
        for child in expr.args:
            if child.is_number or child.is_symbol:
                continue
            hist[child] += 1
            for little_child, occur in self._hist_sub_expr(child).items():
                hist[little_child] += occur
        return hist

    def _len(self, expr):
        """
        Cherche le nombres d'elements.

        * Pas de verification pour une question de performance
        """
        return 1 + sum(self._len(child) for child in expr.args)
