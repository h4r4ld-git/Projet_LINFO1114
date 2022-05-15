import numpy as np

def pageRankPower(A: np.matrix , alpha: float , v: np.array) -> np.array:
    """
    :param A: matrice d'adjacence
    :param alpha: facteur de téléportation
    :param v: vecteur de personnalisation
    :return: Un vecteur x contenant les scores d’importance des noeuds ordonnés dans le même ordre que la matrice d’adjacence.
    """
    # normalise chaque ligne du la matrice
    P = A / np.sum(A, axis=1)[:, None] # normalise chaque ligne du la matrice

    e = np.ones((1, v.size))

    # definir la matrice G (Google)
    G = np.dot(alpha, P) + np.dot(1 - alpha, np.dot(e, v.transpose()))
    
    # initialiser les scores
    x = np.copy(v.transpose())
    converged = False
    i = 0
    print("\nMatrice d'adjacence A :")
    print("\n", A)
    print("\nMatrice de probabilités de transition P (Normalisée) :")
    print("\n", P)
    print("\nMatrice Google G :")
    print("\n", G)
    while not converged:
        new_x = np.dot(x.transpose(),G).transpose()
        new_x = new_x/np.sum(new_x)
        if (np.sum(np.absolute(new_x - x)) < 1e-15).all():
            converged = True
        x = new_x
        if i < 3:
            print(f'Iteration {str(i+1)}')
            print(x.transpose())
            i += 1

    return x.transpose()