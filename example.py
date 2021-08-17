"""
Example d'un script d'analyse de roi1
"""

# import du module
import laue # c'est suffisant, il n'y a rien de plus a importer, tout est contenu dedans.

# Constantes a modifier.
rep = "/home/robin/images/zbb1_roi1" # le dossier qui contient 41**2 images.
calib = {
 'dd': 74.19554756050005,
 'pixelsize': 0.079856,
 'xbet': 0.2984958772257607,
 'xcen': 1148.1825715724603,
 'xgam': -0.1459525334751199,
 'ycen': 959.7753796921565}# Facultatif, permet de gagner 10 minutes.
pos = lambda i: divmod(i%(41*41), 41) # A chaque rang d'image, associe les positions x, y


# Creation de l'experience, l'objet de base qui permet de tout manipuler.
experiment = laue.OrderedExperiment(rep, position=pos, **calib)
experiment.

