#!/usr/bin/env python3

"""
** Permet de separer les differents grains d'un diagramme. **
------------------------------------------------------------

Pour separer les grains d'un diagramme, les etapes sont les suivantes:

1. Chercher les **axes de zone**.
2. Chercher les spots aux **intersections** des axes de zones.
    (Pour la suite on ne considere plus que ces spots.)
3. Construire un **graphe** qui relie les spots qui partagent un meme axe de zone.
4. Supprimer les aretes non pertinantes.
    1. On parcours chaque arete du graphe.
    2. Si il n'y a pas de voisins communs parmis les voisins des 2 somet de cette arete.
    3. On supprime cette arete.
5. Extraire les **composantes connexes** du graphe.
6. Associer les axes de zone aux sous-graphes conexes.
"""

import math


class Splitable:
    """
    Interface pour la classe ``laue.diagram.LaueDiagram``.
    """
    def find_subsets(self, *, angle_max=math.pi/24, spots_max=20, distance_max=.08,
        _atomic_subsets_res=None, _get_args=False, **_):
        """
        ** Recherche des spots qui appartiennent a un meme grain. **

        L'algorithme est le suivant:

        - Creation d'un graphe dont les somets sont des spots.
            - Selection des spots interressants.
                - Selection des spots qui sont a des intersection d'axes de zone.
                - Trie des spots par nombre d'axes de zone decroissant.
                - Selection des ``spots_max`` spots qui intersectent suffisement d'axes.
            - Creation des aretes.
                - Pour chaque paire de somets (paire de spots).
                    - Si il sont lies par un axe de zone.
                        - Alors on les relie avec une arete dont le poid est la
                        qualite du meilleur axe de zone qui passe par ces 2 spots.
        - Supression de certaines aretes.
            - Reperage des paires de spots qui apartiennent a des grains differents.
                - Pour chaque paire de somets (paire de spots).
                    - Si les 2 spots sont trop proche (distance < ``distance_max``).
                        - Alors on repere cette paire de spots.
                    - Si par ces 2 spots passe 2 axes de zone angulairement < ``angle_max``.
                        - Alors on repere cette paire de spots.
            - Tant qu'il existe au moin un chemin permetant de relier une paire de spots marque.
                - On cherche l'arete qui, si on l'a retire, permet de reduire le plus fortement
                le nombre de paire de somet marque qui sont encore relies.
                - Parmis les aretes quandidates, on supprime celle qui est lie a l'axe de
                zone de plus mauvaise qualite.
        - Creation des clusters.
            - On extrait toutes les composantes conexes du graphe.
            - On elimine les composantes qui ne contienent qu'un seul sommet.
            - **return** les composantes conexes restantes.

        Notes
        -----
        Les spots trouves sont souvent des spots ayant un indice de Miller
        relativement faible.

        Returns
        -------
        list
            Chaque element est un 'cluster' de spots qui ont une forte probabilite
            d'appartenir a un meme grain. Par contre, il n'y a aucune garantie
            que 2 clusters appartienent chacuns a 2 grains differents.
            Les cluster sont des ``set`` de spot de type ``laue.spot.Spot``.

        Parameters
        ----------
        angle_max : float
            L'angle maximal entre 2 axes de zones. (en radian)
            Si par un spot, il passe 2 axes de zonnes angulairement plus seres
            que cet angle limite, alors ces 2 axes ne sont comptes que comme 1 seul.
            en consequence, plus cet angle est grand, moins on est selectif sur les spots.
            Ce meme angle intervient ailleur. Si il existe 2 axes de zones passants chacuns
            par un spots differents, que ces 2 axes sont angulairement plus sere
            que ``angle_max``, et que ces 2 spots sont relies par un autre axe de zone,
            alors ces 2 spots se retrouveront dans des clusters differents.
        spots_max : int
            Le nombre maximum total de spots consideres. Plus ce nombre est grand
            plus il y a des chance qu'il y ai une solution mais moins cette solution
            aura des chances d'etre pertinante. Le temps de calcul est aussi quadratiquement
            lie au nombre de sommets.
        distance_max : float
            2 spots qui sont separes d'une distance inferieure a ce parametre se retrouveront
            dans des clusters differents. Cette distance est la distance euclidiene du plan
            gnomonic (en mm).

        Examples
        --------
        >>> import laue
        >>> image = "laue/examples/ge_blanc.mccd"
        >>> diag = next(iter(laue.experiment.base_experiment.Experiment(image, config_file="laue/examples/ge_blanc.det")))
        >>> type(diag.find_subsets())
        <class 'list'>
        >>>
        """
        assert isinstance(angle_max, float), \
            f"'angle_max' has to be a float, not a {type(angle_max).__name__}."
        assert isinstance(spots_max, int), \
            f"'spots_max' has to be an integer, not a {type(spots_max).__name__}."
        assert isinstance(distance_max, float), \
            f"'distance_max' has to be of type float, not {type(distance_max).__name__}."
        assert 0 < angle_max <= math.pi/4, ("L'angle doit etre exprime en radian. "
            f"Il doit etre dans l'intervalle ]0, pi/4]. Or il vaut {angle_max}.")
        assert 2 <= spots_max, \
            f"Il faut au moins considerer 2 spots, {spots_max} c'est pas possible."
        assert 0 < distance_max < .3, ("La distance de sepration des grains est exprimee "
            "en mm dans le plan gnomonic. Elle doit etre comprise entre ]0, .3]. "
            f"Or elle vaut {distance_max}, ce qui sort de cet intervalle.")


        if _get_args: # Si il faut seulement preparer le travail.
            if (angle_max, spots_max, distance_max) in self._subsets:
                return None, None, angle_max, spots_max, distance_max
            spots_dict, axes_dict = _get_spots_axes(self)
            return spots_dict, axes_dict, angle_max, spots_max, distance_max

        if (angle_max, spots_max, distance_max) in self._subsets: # Si on a deja la solution.
            return self._subsets[(angle_max, spots_max, distance_max)]

        if _atomic_subsets_res is None: # Si il faut faire les calculs.
            if self.experiment.verbose:
                print(f"Recherche des sous ensembles de {self.get_id()}...")
            _atomic_subsets_res = atomic_find_subsets(
                *self.find_subsets(angle_max=angle_max, spots_max=spots_max,
                    distance_max=distance_max, _get_args=True))
            if self.experiment.verbose:
                print(f"    OK: {len(_atomic_subsets_res)} sous-ensembles trouves.")

        # Creation des sous ensembles.
        subsets = [{self[spot_id] for spot_id in subset} for subset in _atomic_subsets_res]
        return subsets

def _get_spots_axes(diag):
    """
    Recupere et met en forme les spots et les axes.
    Retourne 'spots' et 'axes' compatible avec ``find_subset``.
    """
    spots = {
        i: {
            "gnom": spot.get_gnomonic(),
            "axes": {axis.get_id() for axis in spot.find_zone_axes()}
        }
        for i, spot in enumerate(diag)
    }
    axes = {
        axes.get_id(): {
            "polar": axes.get_polar_coords(),
            "quality": axes.get_quality(),
            "spots": {spot.get_id() for spot in axes}
        }
        for axes in diag.find_zone_axes()
    }
    return spots, axes

def atomic_find_subsets(spots_dict, axes_dict, angle_max, spots_max, distance_max):
    """
    ** Fonction 'bas niveau' de separation de grains. **

    Notes
    -----
    * Cette fonction n'est pas faite pour etre utilisee directement,
    il vaut mieux s'en servir a travers ``laue.experiment.base_experiment.Experiment.find_subsets``
    ou encore via ``laue.diagram.LaueDiagram.find_subsets`` car le context
    est mieu gere, les entrees sont plus simples et les sorties aussi.
    * Il n'y a pas de verifications sur les entrees car elles sont faite
    dans les methodes de plus haut niveau.
    * Cette fonction n'est pas parallelisee. Par contre la methode
    ``laue.experiment.base_experiment.Experiment.find_subsets`` gere nativement le parallelisme.

    Parameters
    ----------
    spots_dict : dict
        Une representation des spots et des information minimales
        qui y sont raccrochees. Il doit prendre la forme suivante:
        ``{spot_ind: {"gnom": (x_gnom, y_gnom), "axes": {1, 3, ...}}, ...}``
    axes_dict : dict
        Une representations des axes de zone. Doit etre de la forme:
        ``{axe_ind: {"polar": (theta, dist), "quality": .7, "spots": {1, 4, 5, 6}}, ...}``
    Pour les autres arguments, se referer a ``laue.core.subsets.Splitable.find_subsets``.

    Returns
    -------
    list
        Chaque element est un ensemble d'indice de spot appartenant au meme grain.

    Examples
    --------
    >>> import laue
    >>> spots_dict = {0: {'gnom': (0.3136510725564478, -0.44091934214920697), 'axes': set()},
    ...               1: {'gnom': (0.3092269223801162, -0.3703963974694111), 'axes': {0}},
    ...               2: {'gnom': (0.29464954430652696, 0.39670741889986955), 'axes': {1}},
    ...               3: {'gnom': (0.30191318963902786, 0.011759364157261544), 'axes': {2}},
    ...               4: {'gnom': (0.2656586060012433, -0.29892567761860783), 'axes': {0}},
    ...               5: {'gnom': (0.2537441830952802, 0.32256757073867776), 'axes': {1}},
    ...               6: {'gnom': (0.259687191990358, 0.011094867051531301), 'axes': {2}},
    ...               7: {'gnom': (0.21447480004378858, -0.2150686716681864), 'axes': {0}},
    ...               8: {'gnom': (0.2057015738443993, 0.2356420984106664), 'axes': {1}},
    ...               9: {'gnom': (0.09095514707222462, -0.5356714822419564), 'axes': {3, 4}},
    ...               10: {'gnom': (0.06846289051965272, 0.5548594993575858), 'axes': {3, 5}},
    ...               11: {'gnom': (0.16685972592987006, -0.1362785647833782), 'axes': {0}},
    ...               12: {'gnom': (0.16092652407854113, 0.15424696872260854), 'axes': {1}},
    ...               13: {'gnom': (0.0880179013812851, -0.40043047804756454), 'axes': {3}},
    ...               14: {'gnom': (0.07130573600458438, 0.4174843621463065), 'axes': {3}},
    ...               15: {'gnom': (0.08639048300887514, -0.31911672107326594), 'axes': {3}},
    ...               16: {'gnom': (0.07308383568548608, 0.3351760458933546), 'axes': {3}},
    ...               17: {'gnom': (-0.016567440130016563, -0.5541568143935384), 'axes': {6}},
    ...               18: {'gnom': (-0.04084781468303007, 0.5689448068249144), 'axes': {7}},
    ...               19: {'gnom': (0.0011982188252073992, -0.45016007120047424), 'axes': set()},
    ...               20: {'gnom': (-0.018154025485441625, 0.46402526754308027), 'axes': set()},
    ...               21: {'gnom': (-0.07286049562681773, -0.5640133440412382), 'axes': set()},
    ...               22: {'gnom': (0.0798366324753105, 0.007023938379407561), 'axes': {0, 1, 2, 3}},
    ...               23: {'gnom': (-0.0026903259324252246, 0.3921879524265071), 'axes': {5}},
    ...               24: {'gnom': (-0.09784778756977644, 0.5766271192972883), 'axes': set()},
    ...               25: {'gnom': (0.008002458469782677, -0.12293574282766238), 'axes': {1}},
    ...               26: {'gnom': (0.0027460490484708115, 0.13394781189060836), 'axes': {0}},
    ...               27: {'gnom': (-0.00523750349077013, -0.1470394198008183), 'axes': {1}},
    ...               28: {'gnom': (-0.011614516626281228, 0.15777148638160254), 'axes': {0}},
    ...               29: {'gnom': (-0.02477598980567221, -0.1823836791494654), 'axes': {1}},
    ...               30: {'gnom': (-0.03264505249777272, 0.19229497060446735), 'axes': {0}},
    ...               31: {'gnom': (-0.05554717740227801, -0.23832176191125093), 'axes': {1, 4}},
    ...               32: {'gnom': (-0.06606796372263056, 0.24757266647939288), 'axes': {0, 5}},
    ...               33: {'gnom': (-0.07907768658950604, -0.28096918114364355), 'axes': {1}},
    ...               34: {'gnom': (-0.09156428823859053, 0.2893970244445073), 'axes': {0}},
    ...               35: {'gnom': (-0.1126289846318788, -0.3416241560303849), 'axes': {1, 6}},
    ...               36: {'gnom': (-0.060637577826655534, 0.0040391083212410655), 'axes': {2}},
    ...               37: {'gnom': (-0.12787900102974828, 0.3491863530719953), 'axes': {0, 7}},
    ...               38: {'gnom': (-0.1638208798822865, -0.4347286698438957), 'axes': {1}},
    ...               39: {'gnom': (-0.18763951100210385, -0.4776573606384503), 'axes': {1}},
    ...               40: {'gnom': (-0.1835787305562983, 0.4411739247300923), 'axes': {0}},
    ...               41: {'gnom': (-0.08524649743749545, 0.003436523504335945), 'axes': {2}},
    ...               42: {'gnom': (-0.2523542667003644, -0.5949096601292776), 'axes': {8, 1}},
    ...               43: {'gnom': (-0.20958039716029048, 0.4837711867805186), 'axes': {0}},
    ...               44: {'gnom': (-0.11758167206566646, -0.1123768702642382), 'axes': {4}},
    ...               45: {'gnom': (-0.12266805447587614, 0.11817207059217111), 'axes': {5}},
    ...               46: {'gnom': (-0.17376685735949005, 0.0016472083664571353), 'axes': {2, 4, 5}},
    ...               47: {'gnom': (-0.20724297206742168, -0.25150647771157364), 'axes': set()},
    ...               48: {'gnom': (-0.19501092265693473, -0.15910834995174863), 'axes': {6}},
    ...               49: {'gnom': (-0.21870028043714163, 0.25404867230414246), 'axes': set()},
    ...               50: {'gnom': (-0.2022799134764107, 0.16163477331619278), 'axes': {7}},
    ...               51: {'gnom': (-0.2581018757728966, -0.3579924435989167), 'axes': {8}},
    ...               52: {'gnom': (-0.2748170617964089, 0.3596407013965223), 'axes': {8}},
    ...               53: {'gnom': (-0.21290266125204668, 0.0008346770568725252), 'axes': {2}},
    ...               54: {'gnom': (-0.21986841578944072, -0.10369811043407293), 'axes': {5, 6}},
    ...               55: {'gnom': (-0.224622744015843, 0.10503684334781563), 'axes': {4, 7}},
    ...               56: {'gnom': (-0.25955409478645797, -0.29848728294016463), 'axes': {8}},
    ...               57: {'gnom': (-0.27338619051240226, 0.299389601118816), 'axes': {8}},
    ...               58: {'gnom': (-0.3156632165624574, -0.4202358686963255), 'axes': set()},
    ...               59: {'gnom': (-0.2618304221414466, -0.19929059589121062), 'axes': {8, 5}},
    ...               60: {'gnom': (-0.2711073666720387, 0.1994839074523256), 'axes': {8, 4}},
    ...               61: {'gnom': (-0.2637120677768278, -0.11984479170021081), 'axes': {8}},
    ...               62: {'gnom': (-0.26934469418071083, 0.11943958616023759), 'axes': {8}},
    ...               63: {'gnom': (-0.26653796331705015, -0.00040349411462381246), 'axes': {8, 2, 6, 7}},
    ...               64: {'gnom': (-0.3351088299984176, -0.36661822106768144), 'axes': {5}},
    ...               65: {'gnom': (-0.30919293226499917, -0.22782988778598667), 'axes': set()},
    ...               66: {'gnom': (-0.3524825023650732, 0.3647641920521141), 'axes': {4}},
    ...               67: {'gnom': (-0.31990969642322137, 0.22611752961083748), 'axes': set()},
    ...               68: {'gnom': (-0.3144023520521512, -0.0013698968940222644), 'axes': {2}},
    ...               69: {'gnom': (-0.3223772542169661, -0.14153279317889217), 'axes': {7}},
    ...               70: {'gnom': (-0.3290872223583307, 0.13867490378055763), 'axes': {6}},
    ...               71: {'gnom': (-0.34380505280556134, -0.0019232844125064402), 'axes': {2}},
    ...               72: {'gnom': (-0.3716527272451695, -0.2655541496297943), 'axes': {7}},
    ...               73: {'gnom': (-0.384382394518522, 0.26137948985364423), 'axes': {6}},
    ...               74: {'gnom': (-0.4150393355485816, -0.37556396956837046), 'axes': {7}},
    ...               75: {'gnom': (-0.4213116259525938, -0.1277995546595711), 'axes': set()},
    ...               76: {'gnom': (-0.42743697409652126, 0.12091462467337603), 'axes': set()},
    ...               77: {'gnom': (-0.4305672928685682, -0.10207732996277492), 'axes': set()}}
    >>> axes_dict = {0: {'polar': (0.5456325, 0.07189146), 'quality': 0.7786729549943984, 'spots': {32, 1, 34, 4, 37, 7, 40, 11, 43, 22, 26, 28, 30}},
    ...              1: {'polar': (-0.50404394, 0.066435024), 'quality': 0.8495679649229442, 'spots': {33, 2, 35, 5, 38, 39, 8, 42, 12, 22, 25, 27, 29, 31}},
    ...              2: {'polar': (1.5920126, 0.0053436677), 'quality': 0.5048333500772214, 'spots': {3, 36, 68, 6, 71, 41, 46, 53, 22, 63}},
    ...              3: {'polar': (0.020630987, 0.07989738), 'quality': 0.32446216058045907, 'spots': {9, 10, 13, 14, 15, 16, 22}},
    ...              4: {'polar': (-2.6840417, 0.15513226), 'quality': 0.32470155528405464, 'spots': {66, 9, 44, 46, 55, 60, 31}},
    ...              5: {'polar': (2.7287471, 0.15980783), 'quality': 0.36693478788694234, 'spots': {32, 64, 10, 45, 46, 54, 23, 59}},
    ...              6: {'polar': (-2.7180853, 0.24310948), 'quality': 0.324204821510315, 'spots': {35, 70, 73, 48, 17, 54, 63}},
    ...              7: {'polar': (2.7643242, 0.24759501), 'quality': 0.3668147624332471, 'spots': {69, 37, 72, 74, 18, 50, 55, 63}},
    ...              8: {'polar': (-3.118213, 0.26645306), 'quality': 0.5046664108258551, 'spots': {42, 51, 52, 56, 57, 59, 60, 61, 62, 63}}}
    >>> kwargs = {'angle_max': 0.1308996938995747, 'spots_max': 20, 'distance_max': 0.08}
    >>> laue.atomic_find_subsets(spots_dict, axes_dict, **kwargs)
    [{9, 10, 22}]
    >>>
    """
    from laue.zone_axis import distance as distance_axis
    from laue.spot import distance as distance_pic
    import networkx

    def count_variant_axis(spot_id):
        """
        Compte le nombre d'axes de zone qui different beaucoup.
        (ie considere 2 axes proche comme un seul axe)
        """
        axes_id = spots_dict[spot_id]["axes"]
        if len(axes_id) <= 1:
            return len(axes_id)
        axes = [axes_dict[axis_id]["polar"] for axis_id in axes_id]
        nbr_near = (distance_axis(axes, axes, weight=1) < angle_max).sum()
        nbr = len(axes) - (nbr_near-len(axes))//2
        return nbr

    def simul_remove_axis(graph, axis, excluded):
        """
        Retire les aretes liees a cet axe, puis regarde ce que ca donne.
        """
        graph_bis = graph.copy()
        for spot1, spot2, axis_found in graph.edges.data("axis"):
            if axis_found is axis:
                graph_bis.remove_edge(spot1, spot2)
        excluded_bis = [edge for edge in excluded if networkx.algorithms.has_path(graph_bis, *edge)]
        return excluded_bis, graph_bis

    # Extraction des spots.
    spots_at_cross = sorted(spots_dict, key=count_variant_axis, reverse=True)[:spots_max]
    max_cross = count_variant_axis(spots_at_cross[0])
    limit = math.sqrt(max(2**2, max_cross)) # Permet d'eviter 'ValueError: math domain error'
    spots_at_cross = [spot_id for spot_id in spots_at_cross if count_variant_axis(spot_id) >= limit]

    # Creation des noeuds du graphe.
    graph = networkx.Graph()
    graph.add_nodes_from(spots_at_cross)

    # Ajout grossier de certaine aretes.
    excluded = [] # La liste des noeuds appartenant a des grains differents.
    candidate_axes = set() # L'ensemble des axes de zone consideres.
    for i, spot1 in enumerate(spots_at_cross[:-1]): # On faite toutes les combinaisons
        for spot2 in spots_at_cross[i+1:]: # de 2 sommets possibles.

            ## Exclusion des spots trop proches.
            if distance_pic(
                    spots_dict[spot1]["gnom"],
                    spots_dict[spot2]["gnom"],
                    space="gnomonic"
                    ) < distance_max:
                excluded.append((spot1, spot2))
                continue

            ## On ne relie pas les spots qui n'ont pas d'axe commun.
            common_axes = spots_dict[spot1]["axes"] & spots_dict[spot2]["axes"]
            if not common_axes:
                continue

            ## Exclusion des spots ayant 2 axes de zone paralleles.
            axes1, axes2 = spots_dict[spot1]["axes"]-common_axes, spots_dict[spot2]["axes"]-common_axes
            if axes1 and axes2 and distance_axis(
                    [axes_dict[axis_id]["polar"] for axis_id in axes1],
                    [axes_dict[axis_id]["polar"] for axis_id in axes2],
                    weight=1).min() < angle_max: # tolerance angulaire de pi/32
                excluded.append((spot1, spot2))
                continue

            ## Ajout dans le graphe.
            best_axis = sorted(common_axes, key=lambda axis_id: axes_dict[axis_id]["quality"])[-1]
            candidate_axes.add(best_axis)
            graph.add_edge(spot1, spot2, quality=axes_dict[best_axis]["quality"], axis=best_axis)

    # Suppression des aretes en trop afin de discosier les grains.
    excluded = [edge for edge in excluded if networkx.algorithms.has_path(graph, *edge)]
    while excluded:
        predictions = [(*simul_remove_axis(graph, axis_id, excluded), axis_id) for axis_id in candidate_axes]
        best_len = min(len(excluded_bis) for excluded_bis, _, _ in predictions)
        predictions = [p for p in predictions if len(p[0]) == best_len]
        costs = [axes_dict[axis_id]["quality"] for _, _, axis_id in predictions]
        min_quality = min(costs)
        predictions = [p for p in predictions if axes_dict[p[2]]["quality"] == min_quality]
        excluded, graph, axis_id = predictions.pop()
        candidate_axes.remove(axis_id)

    # Creation des clusters.
    subsets = list(networkx.algorithms.connected_components(graph))
    subsets = sorted(subsets, key=lambda con: len(con), reverse=True)
    subsets = [con for con in subsets if len(con) >= 2]
    return subsets

def _jump_find_subsets(args):
    """
    ** Help for ``LaueDiagram.find_subsets``. **

    Etale les arguments de ``atomic_find_subsets`` et saute la fonction si besoin.
    """
    spots_dict, axes_dict, angle_max, spots_max, distance_max = args
    if spots_dict is None: # Si il ne faut pas refaire les calculs
        return {"angle_max": angle_max, "spots_max": spots_max, "distance_max": distance_max}
    return atomic_find_subsets(spots_dict, axes_dict, angle_max, spots_max, distance_max)
