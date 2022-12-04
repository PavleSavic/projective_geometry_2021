import cv2
import numpy as np
import math

def prebaciUHomogene(tacke):
    l = []
    for (x,y) in tacke:
        l.append((x, y, 1))
    return l

def prebaciUAfine(tacke):
    l = []
    for (x1,x2,x3) in tacke:
        l.append((round(x1/x3, 5), round(x2/x3, 5)))
    return l


def preslikajBazne(dst):
    A = dst[0]
    B = dst[1]
    C = dst[2]
    D = dst[3]

    # Kramerovo pravilo

    Delta = np.array([[A[0],B[0],C[0]], [A[1],B[1],C[1]], [A[2],B[2],C[2]]])
    Delta1 = np.array([[D[0],B[0],C[0]], [D[1],B[1],C[1]], [D[2],B[2],C[2]]])
    Delta2 = np.array([[A[0],D[0],C[0]], [A[1],D[1],C[1]], [A[2],D[2],C[2]]])
    Delta3 = np.array([[A[0],B[0],D[0]], [A[1],B[1],D[1]], [A[2],B[2],D[2]]])

    det = np.linalg.det(Delta)
    det1 = np.linalg.det(Delta1)
    det2 = np.linalg.det(Delta2)
    det3 = np.linalg.det(Delta3)

    lambdaA = det1/det
    lambdaB = det2/det
    lambdaC = det3/det

    preslikavanje = np.array([ [lambdaA * A[0], lambdaB * B[0], lambdaC * C[0]], [lambdaA * A[1], lambdaB * B[1], lambdaC * C[1]], [lambdaA * A[2], lambdaB * B[2], lambdaC * C[2]] ])

    return preslikavanje

    
def naivniAlgoritam(src,dst):
    g = preslikajBazne(src)
    h = preslikajBazne(dst)

    inverz_g = np.linalg.inv(g)

    return np.dot(h,inverz_g)

def dlt(src,dst):
    A = []
    
    for i in range(len(src)):
        tacka = src[i]
        slika_tacke = dst[i]

        prvi_red = np.array([0, 0, 0, -slika_tacke[2] * tacka[0], -slika_tacke[2] * tacka[1], -slika_tacke[2] * tacka[2],
         slika_tacke[1] * tacka[0], slika_tacke[1] * tacka[1], slika_tacke[1] * tacka[2]])

        drugi_red = np.array([slika_tacke[2] * tacka[0], slika_tacke[2] * tacka[1], slika_tacke[2] * tacka[2], 0, 0, 0,
         -slika_tacke[0] * tacka[0], -slika_tacke[0] * tacka[1], -slika_tacke[0] * tacka[2]])

        A.append(prvi_red)
        A.append(drugi_red)
    
    A = np.array(A)
    #print(A)

    u, d, vt = np.linalg.svd(A)
    v = vt[-1]
    return np.array([ [v[0], v[1], v[2]] , [v[3], v[4], v[5]], [v[6], v[7], v[8]] ])

def preslikajTacku(preslikavanje, tacka):
    tacka = np.array([[tacka[0]], [tacka[1]], [tacka[2]]])

    slika_tacke = np.dot(preslikavanje, tacka)
    
    return (slika_tacke[0,0], slika_tacke[1,0], slika_tacke[2,0])

def daLiSuProporcionalne(matrica1, matrica2):
    matrica2 = (matrica2 / matrica2[0][0]) * matrica1[0][0]

    print(matrica2)
    return np.allclose(matrica1,matrica2)
    

def translacijaKoordinatnogPocetka(tacke):
    centroid = [0,0]
    brojTacaka = len(tacke)

    for (x,y) in tacke:
        centroid[0] += x
        centroid[1] += y
    
    centroid = round(centroid[0]/brojTacaka, 5), round(centroid[1]/brojTacaka, 5)

    transformisane_tacke = [(x-centroid[0], y-centroid[1]) for (x,y) in tacke]

    matrica_translacije = np.array([[1,0,-centroid[0]], [0,1, -centroid[1]], [0,0,1]])

    return matrica_translacije, transformisane_tacke

def euklidsko_rastojanje(tacka1, tacka2):
    return round(math.sqrt((tacka1[0] - tacka2[0]) ** 2 + (tacka1[1] - tacka2[1]) ** 2),5)

def skaliranje(tacke):
    rastojanja = [euklidsko_rastojanje(tacka, (0,0)) for tacka in tacke]
    d = np.sum(rastojanja) / len(rastojanja)
    
    parametar = round(math.sqrt(2) / d, 5)

    matrica_skaliranja = np.array([[parametar, 0, 0], [0, parametar, 0] , [0, 0, 1]])
    return matrica_skaliranja

def normalizacija(tacke):
    afineTacke = prebaciUAfine(tacke)

    matrica_translacije, transformisane_tacke = translacijaKoordinatnogPocetka(afineTacke)
    matrica_skaliranja = skaliranje(transformisane_tacke)

    matrica_normalizacije = np.dot(matrica_skaliranja, matrica_translacije)

    return [preslikajTacku(matrica_normalizacije, tacka) for tacka in tacke], matrica_normalizacije


def normalizovanidlt(src,dst):
    normalizovane_tacke, matrica_normalizacije_tacaka = normalizacija(src)
    normalizovane_slike_tacaka, matrica_normalizacije_slika_tacaka = normalizacija(dst)

    pndlt = dlt(normalizovane_tacke, normalizovane_slike_tacaka)
    pndltPolazne = np.dot(np.dot(np.linalg.inv(matrica_normalizacije_slika_tacaka), pndlt), matrica_normalizacije_tacaka)
    
    return pndltPolazne

def otklanjanje_distorzije(putanja_do_slike, koordinate):
    img = cv2.imread(putanja_do_slike)
    img_copy = np.copy(img)

    A = koordinate[0]
    B = koordinate[1]
    C = koordinate[2]
    D = koordinate[3]

    AB = np.sqrt(((A[0] - B[0]) ** 2) + ((A[1] - B[1]) ** 2))
    CD = np.sqrt(((C[0] - D[0]) ** 2) + ((C[1] - D[1]) ** 2))
    maxSirina = max(int(AB), int(CD))
 
    AD = np.sqrt(((A[0] - D[0]) ** 2) + ((A[1] - D[1]) ** 2))
    BC = np.sqrt(((B[0] - C[0]) ** 2) + ((B[1] - C[1]) ** 2))
    maxVisina = max(int(AD), int(BC))

    nove_koordinate = [(0,maxVisina-1),(maxSirina-1,maxVisina-1), (maxSirina-1,0), (0,0)]

    koordinate = prebaciUHomogene(koordinate)
    nove_koordinate = prebaciUHomogene(nove_koordinate)

    p = dlt(koordinate, nove_koordinate)

    img_transform = cv2.warpPerspective(img_copy, p, (maxSirina,maxVisina))

    print("Prikaz slike!")
    cv2.imshow('konacna slika', img_transform)
    cv2.waitKey(30000)
    cv2.destroyAllWindows()
    

def primer1():
    p = naivniAlgoritam([(-3,-1,1), (3,-1,1), (1,1,1), (-1,1,1)], [(-2,-1,1), (2,-1,1), (2,1,1), (-2,1,1)])
    print(f"Matrica preslikavanja dobijena naivnim algoritmom:\n {p}")

    pdlt = dlt([(-3,-1,1), (3,-1,1), (1,1,1), (-1,1,1), (1,2,3), (-8,-2,1)], [(-2,-1,1), (2,-1,1), (2,1,1), (-2,1,1), preslikajTacku(p, (1,2,3)), preslikajTacku(p, (-8,-2,1))])
    print(f"Matrica preslikavanja dobijena dlt algoritmom:\n {pdlt}")

    pdltRounded = np.matrix.round(pdlt, decimals = 5)
    print(f"Zaokruzena na 5 decimala:\n {pdltRounded}")

    print(f"Provera proporcionalnosti: {daLiSuProporcionalne(p,pdltRounded)}")

    normalizovane_tacke, matrica_normalizacije_tacaka = normalizacija([(-3,-1,1), (3,-1,1), (1,1,1), (-1,1,1), (1,2,3), (-8,-2,1)])
    print(f"Matrica koja normalizuje tacke: \n{matrica_normalizacije_tacaka}")
    print(f"Normalizovane tacke:\n {normalizovane_tacke}")

    normalizovane_slike_tacaka, matrica_normalizacije_slika_tacaka = normalizacija([(-2,-1,1), (2,-1,1), (2,1,1), (-2,1,1), preslikajTacku(p, (1,2,3)), preslikajTacku(p, (-8,-2,1))])
    print(f"Matrica koja normalizuje slike tacaka: \n{matrica_normalizacije_slika_tacaka}")
    print(f"Normalizovane slike tacaka:\n {normalizovane_slike_tacaka}")

    pndlt = dlt(normalizovane_tacke, normalizovane_slike_tacaka)
    pndltRounded = np.matrix.round(pndlt, decimals = 5)
    print(f"Matrica preslikavanja dobijena normalizovanim dlt algoritmom za normalizovane koordinate tacaka (zaokruzena na 5 decimala):\n {pndltRounded}")

    pndltPolazne = np.dot(np.dot(np.linalg.inv(matrica_normalizacije_slika_tacaka), pndlt), matrica_normalizacije_tacaka)
    pndltPolazneRounded = np.matrix.round(pndltPolazne, decimals = 4)
    print(f"Matrica preslikavanja u polaznim koordinatama:\n {pndltPolazneRounded}")

    print(f"Provera proporcionalnosti: {daLiSuProporcionalne(p,pndltPolazneRounded)}")

def primer2():
    a = (1, 1, 1)
    b = (5, 1, 1)
    c = (5, 3, 1)
    d = (1, 5, 1)
    tacke = [a,b,c,d]
    
    slika_a = (-1, -2, 1)
    slika_b = (3, -2, 1)
    slika_c = (3, 2, 1)
    slika_d = (1, 2, 1)
    slike_tacaka = [slika_a, slika_b, slika_c, slika_d]

    p = naivniAlgoritam(tacke, slike_tacaka)
    print(f"Matrica preslikavanja dobijena naivnim algoritmom:\n {p}")

    pdlt = dlt(tacke, slike_tacaka)
    print(f"Matrica preslikavanja dobijena dlt algoritmom za 4 korespodencije:\n {pdlt}")
    
    print("Proveri proporcionalnost:")
    print(daLiSuProporcionalne(p, pdlt))

    pndlt = normalizovanidlt(tacke, slike_tacaka)
    print(f"Matrica preslikavanja dobijena modifikovanim dlt algoritmom za 4 korespodencije:\n {pndlt}")

    print("Proveri proporcionalnost:")
    print(daLiSuProporcionalne(p,pndlt))
    print("-------------------------------------------------------------------------------------------------------")

    e = (2,2,1)
    f = (4,1,1)

    tacke.append(e)
    tacke.append(f)

    slika_e = (0.2, -0.1, 0.6)
    slika_f = (0.4, -0.6, 0.3)

    slike_tacaka.append(slika_e)
    slike_tacaka.append(slika_f)

    print(f"Dodate dve tacke i njihove slike: {tacke}\n{slike_tacaka}")

    pdlt = dlt(tacke, slike_tacaka)
    print(f"Matrica preslikavanja dobijena dlt algoritmom za 6 korespodencija:\n {pdlt}")
    
    print("Proveri proporcionalnost:")
    print(daLiSuProporcionalne(p,pdlt))

    pndlt = normalizovanidlt(tacke, slike_tacaka)
    print(f"Matrica preslikavanja dobijena modifikovanim dlt algoritmom za 6 korespodencija:\n {pndlt}")

    print("Proveri proporcionalnost:")
    print(daLiSuProporcionalne(p,pndlt))

    print("-------------------------------------------------------------------------------------------------------")

    print("Invarijantnost na promenu koordinata:")

    matrica1 = np.array([[0, 1, 2], [-1, 0, 3], [0, 0, 1]])
    print(f"Matrica transformacije tacaka:\n{matrica1}")
    nove_tacke = [preslikajTacku(matrica1, tacka) for tacka in tacke]
    print(f"Nove koordinate tacaka: {nove_tacke}")

    matrica2 = np.array([[1, -1, 5], [1, 1, -2], [0, 0, 1]])
    print(f"Matrica transformacije slika tacaka:\n{matrica2}")
    nove_slike_tacaka = [preslikajTacku(matrica2, tacka) for tacka in slike_tacaka]
    print(f"Nove koordinate slika tacaka: {nove_slike_tacaka}")

    npdlt = dlt(nove_tacke, nove_slike_tacaka)
    pdltStari = np.dot(np.dot(np.linalg.inv(matrica2), npdlt), matrica1)
    print(f"Rezultat dlt-a:\n{pdltStari}")

    print("Proveri invarijantnost na promenu koordinata obicnog dlt-a:")
    print(daLiSuProporcionalne(pdlt,pdltStari))
    print("Postoji razlika na 5. decimali!")

    npndlt = normalizovanidlt(nove_tacke, nove_slike_tacaka)
    pndltStari = np.dot(np.dot(np.linalg.inv(matrica2), npndlt), matrica1)
    print(f"Rezultat normalizovanog dlt-a:\n{pndltStari}")

    print("Proveri invarijantnost na promenu koordinata normalizovanog dlt-a:")
    print(daLiSuProporcionalne(pndlt, pndltStari))
    print("Identicne matrice!")


def primer3():

    print("Unesi adresu slike: ")
    adresa = input()

    tacke = []
    print("Unesi koordinate 4 tacke: ")
    for i in range(4):
        x = int(input())
        y = int(input())
        tacke.append((x,y))
        print('---------------')
    
    otklanjanje_distorzije(adresa, tacke)

def primer4():
    print("1)")

    y1 = (2,1,1)
    y2 = (1,2,1)
    y3 = (3,4,1)
    y4 = (-1,-3,1)
    tacke = [y1,y2,y3,y4]

    y1p = (0,1,1)
    y2p = (5,0,1)
    y3p = (2,-5,1)
    y4p = (-1,-1,1)
    slike_tacaka = [y1p,y2p,y3p,y4p]

    p = naivniAlgoritam(tacke,slike_tacaka)
    p = (p / p[0][0]) * 1.0
    print(f"Matrica preslikavanja dobijena naivnim algoritmom:\n {p}")

    pdlt = dlt(tacke, slike_tacaka)
    pdlt = (pdlt / pdlt[0][0]) * 1.0
    print(f"Matrica preslikavanja dobijena dlt algoritmom:\n {pdlt}")

    pndlt = normalizovanidlt(tacke, slike_tacaka)
    pndlt = (pndlt / pndlt[0][0]) * 1.0
    print(f"Matrica preslikavanja dobijena modifikovanim dlt algoritmom:\n {pndlt}")

    print("2)")

    y5 = (-2,5,1)
    tacke.append(y5)

    y5p = (4,1,2)
    slike_tacaka.append(y5p)

    pdlt = dlt(tacke, slike_tacaka)
    pdlt = (pdlt / pdlt[0][0]) * 1.0
    print(f"Matrica preslikavanja dobijena dlt algoritmom (5 tacaka):\n {pdlt}")

    pndlt = normalizovanidlt(tacke, slike_tacaka)
    pndlt = (pndlt / pndlt[0][0]) * 1.0
    print(f"Matrica preslikavanja dobijena modifikovanim dlt algoritmom (5 tacaka):\n {pndlt}")

    print("3)")

    yn1 = (0,-3,1)
    yn2 = (0,-1,1)
    yn3 = (4,-1,1)
    yn4 = (-7,-4,1)
    yn5 = (0,5,1)
    tacke = [yn1,yn2,yn3,yn4,yn5]

    yn1p = (3,-1,1)
    yn2p = (4,4,1)
    yn3p = (9,1,1)
    yn4p = (5,-2,1)
    yn5p = (7,2,2)
    slike_tacaka = [yn1p,yn2p,yn3p,yn4p,yn5p]

    pndlt = normalizovanidlt(tacke, slike_tacaka)
    pndlt = (pndlt / pndlt[0][0]) * 1.0
    print(f"Matrica preslikavanja dobijena modifikovanim dlt algoritmom:\n {pndlt}")

    













    
if __name__ == "__main__":

    #primer1()
    #primer2()
    #primer3()
    primer4()



