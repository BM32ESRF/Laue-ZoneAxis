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
        _atomic_subsets_res=None, _get_args=False):
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
        >>> diag = next(iter(laue.Experiment(image, config_file="laue/examples/ge_blanc.det")))
        >>> type(diag.find_subsets())
        <class 'list'>
        >>> len(diag.find_subsets()) # nbr de clusters
        1
        >>> type(diag.find_subsets().pop())
        <class 'set'>
        >>> len(diag.find_subsets().pop()) # nbr de spot dans le premier cluster
        3
        >>> type(diag.find_subsets().pop().pop())
        <class 'laue.spot.Spot'>
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

        args = _get_spots_axes(self)
        config = dict(angle_max=angle_max,
                      spots_max=spots_max,
                      distance_max=distance_max)
        if _get_args: # Si il faut seulement preparer le travail.
            return args, config

        if _atomic_subsets_res is None: # Si il faut faire les calculs.
            _atomic_subsets_res = atomic_find_subsets(*args, **config)
        
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


def atomic_find_subsets(spots_dict, axes_dict, **kwargs):
    """
    Help for pickelable ``find_subset``.

    Parameters
    ----------
    spots_dict : dict
        {spot_ind: {"gnom": (x, y), "axes": {1, 2, 3}}, ...}
    axes_dict : dict
        {axe_ind: {"polar": (theta, dist), "quality": .7, "spots": {1, 4, 5, 6}}, ...}

    Returns
    -------
    list
        Chaque element est un ensemble de spot appartenant au meme grain.
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
        nbr_near = (distance_axis(axes, axes, weight=1) < kwargs["angle_max"]).sum()
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
    spots_at_cross = sorted(spots_dict, key=count_variant_axis, reverse=True)[:kwargs["spots_max"]]
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
                    ) < kwargs["distance_max"]:
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
                    weight=1).min() < kwargs["angle_max"]: # tolerance angulaire de pi/32
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

def _pickelable(args):
    """
    args = ((spots_dict, axes_dict), kwargs)
    """
    return atomic_find_subsets(*args[0], **args[1])
