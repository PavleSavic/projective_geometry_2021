import numpy as np
import math

def prebaciUHomogene(tacke):
    l = []
    for (x, y, z) in tacke:
        l.append((x, y, z, 1))
    return l

def prebaciUAfine(tacke):
    l = []
    for (x1, x2, x3, x4) in tacke:
        l.append((round(x1/x4, 5), round(x2/x4, 5), round(x3/x4, 5)))
    return l

def ParametriKamere(T):
    #odredjivanje centra kamere 
    T1 = T[:, 1:]
    T2 = np.delete(T, 1, axis=1)
    T3 = np.delete(T, 2, axis=1)
    T4 = T[:, :-1]

    det1 = np.linalg.det(T1)
    det2 = np.linalg.det(T2)
    det3 = np.linalg.det(T3)
    det4 = np.linalg.det(T4)

    C = (det1, -det2, det3, -det4)
    C = prebaciUAfine([C])[0]
 
    # odredjivanje matrica K i A
    T0 = T[:, :-1]

    if np.linalg.det(T0) < 0:
        T = -T
        T0 = T[:, :-1]

    T0_inverz = np.linalg.inv(T0)
    
    Q, R = np.linalg.qr(T0_inverz)

    if R[0][0] < 0:
        R = np.array([[-R[0][0], -R[0][1], -R[0][2]], [R[1][0], R[1][1], R[1][2]], [R[2][0], R[2][1], R[2][2]]])
        Q = np.array([[-Q[0][0], Q[0][1], Q[0][2]], [-Q[1][0], Q[1][1], Q[1][2]], [-Q[2][0], Q[2][1], Q[2][2]]])
    
    if R[1][1] < 0:
        R = np.array([[R[0][0], R[0][1], R[0][2]], [-R[1][0], -R[1][1], -R[1][2]], [R[2][0], R[2][1], R[2][2]]])
        Q = np.array([[Q[0][0], -Q[0][1], Q[0][2]], [Q[1][0], -Q[1][1], Q[1][2]], [Q[2][0], -Q[2][1], Q[2][2]]])

    if R[2][2] < 0:
        R = np.array([[R[0][0], R[0][1], R[0][2]], [R[1][0], R[1][1], R[1][2]], [-R[2][0], -R[2][1], -R[2][2]]])
        Q = np.array([[Q[0][0], Q[0][1], -Q[0][2]], [Q[1][0], Q[1][1], -Q[1][2]], [Q[2][0], Q[2][1], -Q[2][2]]])

    K = np.linalg.inv(R)
    # normiranje matrice kalibracije
    K = K / K[2][2]
    A = np.linalg.inv(Q) # posto je ortogonalna moze i transpose (Q^-1 == Q_T)

    return np.round(K, 5), np.round(A, 5), C

def MatricaKamereOdParametara(K, A, C):
    T_0 = np.dot(K, A)
    poslednja_kolona = -np.dot(T_0, np.array([[C[0]], [C[1]], [C[2]]]))
    T = np.hstack((T_0, poslednja_kolona))
    return T

def daLiSuProporcionalne(matrica1, matrica2):
    matrica2 = (matrica2 / matrica2[0][0]) * matrica1[0][0]
    print(matrica2)
    return np.allclose(matrica1,matrica2)

def CameraDLP(originali, projekcije):
    if len(originali) < 6 or len(originali) != len(projekcije):
        return None

    matrica_korespondencija = []
    
    for i in range(len(originali)):
        tacka = originali[i]
        slika_tacke = projekcije[i]

        prvi_red = np.array([0, 0, 0, 0, -slika_tacke[2] * tacka[0], -slika_tacke[2] * tacka[1], -slika_tacke[2] * tacka[2], -slika_tacke[2] * tacka[3],
         slika_tacke[1] * tacka[0], slika_tacke[1] * tacka[1], slika_tacke[1] * tacka[2], slika_tacke[1] * tacka[3]])

        drugi_red = np.array([slika_tacke[2] * tacka[0], slika_tacke[2] * tacka[1], slika_tacke[2] * tacka[2], slika_tacke[2] * tacka[3], 0, 0, 0, 0,
         -slika_tacke[0] * tacka[0], -slika_tacke[0] * tacka[1], -slika_tacke[0] * tacka[2], -slika_tacke[0] * tacka[3]])

        matrica_korespondencija.append(prvi_red)
        matrica_korespondencija.append(drugi_red)
    
    matrica_korespondencija = np.array(matrica_korespondencija)
    #print(matrica_korespondencija)

    u, d, vt = np.linalg.svd(matrica_korespondencija)
    v = vt[-1]
    T = np.array([ [v[0], v[1], v[2], v[3]] , [v[4], v[5], v[6], v[7]], [v[8], v[9], v[10], v[11]] ])
    return T

def primer1():
    n = 9
    T = np.array([[5, -1 - 2*n, 3, 18-3*n], [0, -1, 5, 21], [0, -1, 0, 1]])
    print("Matrica kamere:")
    print(f"T = {T}")

    K, A, C = ParametriKamere(T)
    print("Dobijeni parametri:")
    print(f"K = {K}")
    print(f"A = {A}")
    print(f"C = {C}")
    
    print("Provera (matrica kamere od dobijenih parametara):")
    T1 = MatricaKamereOdParametara(K, A, C)
    print(f"T = {T1}")
    print(f"Da li su proporcinalne? {daLiSuProporcionalne(T,T1)}")

def primer2():
    n = 9
    M1 = (460, 280, 250, 1)
    M2 = (50, 380, 350, 1)
    M3 = (470, 500, 100, 1)
    M4 = (380, 630, 50*n, 1)
    M5 = (30*n, 290, 0, 1)
    M6 = (580, 0, 130, 1)
    originali = [M1, M2, M3, M4, M5, M6]

    M1p = (288, 251, 1)
    M2p = (79, 510, 1)
    M3p = (470, 440, 1)
    M4p = (520, 590, 1)
    M5p = (365, 388, 1)
    M6p = (365, 20, 1)
    projekcije = [M1p, M2p, M3p, M4p, M5p, M6p]

    print("Matrica kamere dobijena za date tacke i njihove projekcije:")
    T = CameraDLP(originali, projekcije)
    T = T / T[0][0]
    T = np.round(T, 5)
    print(f"T = {T}")

# primer sa prilozene slike
def primer3():
    M1 = (210, 140, 20, 1)
    M2 = (205, 0, 75, 1)
    M3 = (150, 75, 110, 1)
    M4 = (200, 50, 250, 1)
    M5 = (400, 30, 20, 1)
    M6 = (420, 140, 20, 1)
    originali = [M1, M2, M3, M4, M5, M6]

    # aproksimacija prebacivanja piksel koordinata u milimetre (deljenje sa 110)
    M1p = (646/110, 739/110, 1)
    M2p = (443/100, 513/110, 1)
    M3p = (266/110, 709/110, 1)
    M4p = (52/110, 552/110, 1)
    M5p = (788/110, 221/100, 1)
    M6p = (412/110, 488/110, 1)
    projekcije = [M1p, M2p, M3p, M4p, M5p, M6p]

    print("Matrica kamere dobijena za date tacke i njihove projekcije:")
    T = CameraDLP(originali, projekcije)
    print(f"T = {T}")

    K, A, C = ParametriKamere(T)
    print("Dobijeni parametri od ove matrice:")
    print(f"K = {K}")
    print(f"A = {A}")
    print(f"C = {C}")

    print("Provera (matrica kamere od dobijenih parametara):")
    T1 = MatricaKamereOdParametara(K, A, C)
    print(f"T = {T1}")
    print(f"Da li su proporcinalne? {daLiSuProporcionalne(T,T1)}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True) # Lepsi ispis float-ova
    #primer1()
    #primer2()
    primer3()

    

