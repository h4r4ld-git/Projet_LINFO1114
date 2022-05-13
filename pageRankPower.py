import numpy as np

def pageRankPower(A: np.matrix , alpha: float , v: np.array) -> np.array:
    """
    :param A: matrice d'adjacence
    :param alpha: facteur de téléportation
    :param v: vecteur de personnalisation
    :return: Un vecteur x contenant les scores d’importance des noeuds ordonnés dans le même ordre que la matrice d’adjacence.
    """
    # normalise chaque ligne du la matrice
    P = A / np.sum(A, axis=1)[:, None]
    
    e = np.ones((v.size, 1))

    # definir la matrice G (Google)
    G = np.multiply(alpha, P) + np.multiply(1 - alpha, np.multiply(e, v.transpose()))

    # initialiser les scores
    x = np.copy(v)

    converged = False

    while not converged:
        new_x = np.multiply(x.transpose(), G).transpose()
        if new_x == x:
            converged = True
        else:
            x = new_x

    return x