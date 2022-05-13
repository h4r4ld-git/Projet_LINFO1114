import numpy as np

def pageRankLinear(A: np.matrix , alpha: float , v: np.array) -> np.array:
    """
    :param A: matrice d'adjacence
    :param alpha: facteur de téléportation
    :param v: vecteur de personnalisation
    :return: Un vecteur x contenant les scores d’importance des noeuds ordonnés dans le même ordre que la matrice d’adjacence.
    """
    P = A / np.sum(A, axis=1)[:, None] # normalise chaque ligne du la matrice
    longueur = P.shape[0]

    # équation page 144
    # comme ax = c := x = a'c
    c = np.multiply(1 - alpha, v)
    a_transpose = np.subtract(np.identity(longueur), np.multiply(alpha, P))
    x = np.multiply(a_transpose, c)
    x = np.divide(x, np.sum(x)) # normalise le vecteur
    return x