import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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


def VektorskiProizvod(t1,t2):
    return tuple(np.cross(list(t1), list(t2)))


def odrediKanonskeMatrice(xx, yy):
    if len(xx) < 8 or len(xx) != len(yy):
        return None

    matrica_korespondencija = []
    
    for i in range(len(xx)):
        a = xx[i]
        b = yy[i]

        # y^T * F * x = 0
        jednacina = np.array([a[0] * b[0], a[1] * b[0], a[2] * b[0], a[0] * b[1], a[1] * b[1], a[2] * b[1], a[0] * b[2], a[1] * b[2], a[2] * b[2] ])
        matrica_korespondencija.append(jednacina)

    matrica_korespondencija = np.array(matrica_korespondencija)
    #print(matrica_korespondencija)

    u, d, vt = np.linalg.svd(matrica_korespondencija)
    v = vt[-1]

    # fundamentalna matrica
    F = np.array([ [v[0], v[1], v[2]] , [v[3], v[4], v[5]], [v[6], v[7], v[8]] ])
    print(f"F = {F}")

    print("Provera da li vazi za sve parove y^T * F * x = 0 (priblizno):")
    vrednosti = []
    for i in range(len(xx)):
        x = xx[i]
        y = yy[i]

        #kolona
        x = np.array([[x[0]], [x[1]], [x[2]]])
        #vrsta
        y_t = np.array([y[0], y[1], y[2]])

        vrednost = np.dot(np.dot(y_t , F), x)
        vrednosti.append(vrednost[0])
    
    vrednosti = np.array(vrednosti)
    print(vrednosti)
    print(np.allclose(vrednosti, np.zeros(len(xx))))

    print("Provera da li je det(F) = 0 (priblizno):")
    detF = np.linalg.det(F)
    print(f"Determinanta matrice F: {detF}")
    print(math.isclose(detF, 0, abs_tol = 1.e-9))

    #trazimo epipolove e1 i e2 (F * e1 = 0, F_t * e2 = 0)
    u, d, vt = np.linalg.svd(F)
   
    e1 = vt[-1]
    # u, d, vt je svd dekompozicija F => v, d, ut je svd dekompozicija F_t
    ut = np.transpose(u)
    e2 = ut[-1] 

    #da dobijemo trecu homogenu koordinatu jednaku 1
    e1 = e1 / e1[2]
    e2 = e2 / e2[2]

    print(f"Epipolovi: e1 = {e1},\ne2 = {e2}")

    print("Zelimo da priblizimo determinantu matrice F nuli")
    dijag = np.diag(np.array([1,1,0]))
    # f-ja za svd vraca samo dijagonalu matrice d (sve van nje je 0) pa moramo da je pretvorimo u dijagonalnu matricu 
    d1 = np.dot(dijag, np.diag(d))
    F1 = np.dot(np.dot(u, d1), vt)
    print(f"Nakon transformacije: F1 = {F1}")

    print("Provera da li je det(F1) = 0 (priblizno):")
    detF1 = np.linalg.det(F1)
    print(f"Determinanta matrice F1: {detF1}")
    print(math.isclose(detF1, 0, abs_tol = 1.e-9))

    print("Uzimamo kanonske matrice kamera:")

    T1 = np.hstack((np.eye(3), np.zeros((3,1))))
    print(f"T1 = {T1}")

    E2 = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    T2 = np.transpose(np.vstack((np.transpose(np.dot(E2, F1)), np.array([e2[0], e2[1], e2[2]]))))
    print(f"T2 = {T2}")

    return T1, T2


def rekonstruisiTackeProstora(xx, yy, T1, T2):

    if len(xx) != len(yy):
        return None

    tackeProstora = []
    for i in range(len(xx)):
        x = xx[i]
        y = yy[i]

        jednacina = np.array([x[1] * T1[2] - x[2] * T1[1], -x[0] * T1[2] + x[2] * T1[0], y[1] * T2[2] - y[2] * T2[1], -y[0] * T2[2] + y[2] * T2[0]])
        
        u, d, vt = np.linalg.svd(jednacina)
        tacka = tuple(vt[-1])

        tackeProstora.append(tacka)

    tackeProstora = prebaciUAfine(tackeProstora)
    
    #pomnoziti z koordinatu sa nekoliko stotina jer nismo radili normalizaciju
    for i in range(len(tackeProstora)):
        tackeProstora[i] = (tackeProstora[i][0], tackeProstora[i][1], tackeProstora[i][2] * 400)
        
    return tackeProstora


def primer1():
    x1 = (958, 38, 1)
    y1 = (933, 33, 1)
    x2 = (1117, 111, 1)
    y2 = (1027, 132, 1)
    x3 = (874, 285, 1)
    y3 = (692, 223, 1)
    x4 = (707, 218, 1)
    y4 = (595, 123, 1)
    x9 = (292, 569, 1)
    y9 = (272, 360, 1)
    x10 = (770, 969, 1)
    y10 = (432, 814, 1)
    x11 = (770, 1465, 1)
    y11 = (414, 1284, 1)
    x12 = (317, 1057, 1)
    y12 = (258, 818, 1)

    xx = [x1, x2, x3, x4, x9, x10, x11, x12]
    yy = [y1, y2, y3, y4, y9, y10, y11, y12]

    print("Korisceno 8 tacaka za odredjivanje fundamentalne matrice, nevidljive tacke procenjene")
    T1, T2 = odrediKanonskeMatrice(xx, yy)

    x6 = (1094, 536, 1)
    y6 = (980, 535, 1)
    x7 = (862, 729, 1)
    y7 = (652, 638, 1)
    x8 = (710, 648, 1)
    y8 = (567, 532, 1)
    x14 = (1487, 598, 1)
    y14 = (1303, 700, 1)
    x15 = (1462, 1079, 1)
    y15 = (1257, 1165, 1)
    y13 = (1077, 269, 1)

    # procena za nevidljive tacke
    x5 = (938, 481, 1)
    x13 = (985, 251, 1)
    x16 = (1057, 618, 1)

    y5 = (889, 448, 1)
    y16 = (1058, 711, 1)

    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]
    yy = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16]

    tackeProstora = rekonstruisiTackeProstora(xx, yy, T1, T2)
    print(f"Dobijene su tacke prostora:\n{tackeProstora}")


    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x_values = [tackeProstora[0][0], tackeProstora[1][0]]
    y_values = [tackeProstora[0][1], tackeProstora[1][1]]
    z_values = [tackeProstora[0][2], tackeProstora[1][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[1][0], tackeProstora[2][0]]
    y_values = [tackeProstora[1][1], tackeProstora[2][1]]
    z_values = [tackeProstora[1][2], tackeProstora[2][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[2][0], tackeProstora[3][0]]
    y_values = [tackeProstora[2][1], tackeProstora[3][1]]
    z_values = [tackeProstora[2][2], tackeProstora[3][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[3][0], tackeProstora[0][0]]
    y_values = [tackeProstora[3][1], tackeProstora[0][1]]
    z_values = [tackeProstora[3][2], tackeProstora[0][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[4][0], tackeProstora[5][0]]
    y_values = [tackeProstora[4][1], tackeProstora[5][1]]
    z_values = [tackeProstora[4][2], tackeProstora[5][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[5][0], tackeProstora[6][0]]
    y_values = [tackeProstora[5][1], tackeProstora[6][1]]
    z_values = [tackeProstora[5][2], tackeProstora[6][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[6][0], tackeProstora[7][0]]
    y_values = [tackeProstora[6][1], tackeProstora[7][1]]
    z_values = [tackeProstora[6][2], tackeProstora[7][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[7][0], tackeProstora[4][0]]
    y_values = [tackeProstora[7][1], tackeProstora[4][1]]
    z_values = [tackeProstora[7][2], tackeProstora[4][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[0][0], tackeProstora[4][0]]
    y_values = [tackeProstora[0][1], tackeProstora[4][1]]
    z_values = [tackeProstora[0][2], tackeProstora[4][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[1][0], tackeProstora[5][0]]
    y_values = [tackeProstora[1][1], tackeProstora[5][1]]
    z_values = [tackeProstora[1][2], tackeProstora[5][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[2][0], tackeProstora[6][0]]
    y_values = [tackeProstora[2][1], tackeProstora[6][1]]
    z_values = [tackeProstora[2][2], tackeProstora[6][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[3][0], tackeProstora[7][0]]
    y_values = [tackeProstora[3][1], tackeProstora[7][1]]
    z_values = [tackeProstora[3][2], tackeProstora[7][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[8][0], tackeProstora[9][0]]
    y_values = [tackeProstora[8][1], tackeProstora[9][1]]
    z_values = [tackeProstora[8][2], tackeProstora[9][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[9][0], tackeProstora[10][0]]
    y_values = [tackeProstora[9][1], tackeProstora[10][1]]
    z_values = [tackeProstora[9][2], tackeProstora[10][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[10][0], tackeProstora[11][0]]
    y_values = [tackeProstora[10][1], tackeProstora[11][1]]
    z_values = [tackeProstora[10][2], tackeProstora[11][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[11][0], tackeProstora[8][0]]
    y_values = [tackeProstora[11][1], tackeProstora[8][1]]
    z_values = [tackeProstora[11][2], tackeProstora[8][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[12][0], tackeProstora[13][0]]
    y_values = [tackeProstora[12][1], tackeProstora[13][1]]
    z_values = [tackeProstora[12][2], tackeProstora[13][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[13][0], tackeProstora[14][0]]
    y_values = [tackeProstora[13][1], tackeProstora[14][1]]
    z_values = [tackeProstora[13][2], tackeProstora[14][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[14][0], tackeProstora[15][0]]
    y_values = [tackeProstora[14][1], tackeProstora[15][1]]
    z_values = [tackeProstora[14][2], tackeProstora[15][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[15][0], tackeProstora[12][0]]
    y_values = [tackeProstora[15][1], tackeProstora[12][1]]
    z_values = [tackeProstora[15][2], tackeProstora[12][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[8][0], tackeProstora[12][0]]
    y_values = [tackeProstora[8][1], tackeProstora[12][1]]
    z_values = [tackeProstora[8][2], tackeProstora[12][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[9][0], tackeProstora[13][0]]
    y_values = [tackeProstora[9][1], tackeProstora[13][1]]
    z_values = [tackeProstora[9][2], tackeProstora[13][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[10][0], tackeProstora[14][0]]
    y_values = [tackeProstora[10][1], tackeProstora[14][1]]
    z_values = [tackeProstora[10][2], tackeProstora[14][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[11][0], tackeProstora[15][0]]
    y_values = [tackeProstora[11][1], tackeProstora[15][1]]
    z_values = [tackeProstora[11][2], tackeProstora[15][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    plt.title('3D rekonstrukcija')
    plt.show()


def primer2():
    # leva slika: nevidljive tacke x8, x16, x24
    x1 = (815, 110, 1)
    x2 = (951, 158, 1)
    x3 = (991, 122, 1)
    x4 = (855, 78, 1)
    x5 = (790, 303, 1)
    x6 = (914, 359, 1)
    x7 = (952, 319, 1)
    x9 = (321, 344, 1)
    x10 = (452, 368, 1)
    x11 = (512, 270, 1)
    x12 = (387, 247, 1)
    x13 = (362, 558, 1)
    x14 = (479, 584, 1)
    x15 = (528, 486, 1)
    x17 = (135, 549, 1)
    x18 = (433, 760, 1)
    x19 = (816, 382, 1)
    x20 = (546, 252, 1)
    x21 = (175, 654, 1)
    x22 = (450, 861, 1)
    x23 = (806, 491, 1)
    # desna slika: nevidljive tacke y5, y13, y17, y21
    y1 = (913, 444, 1)
    y2 = (810, 561, 1)
    y3 = (919, 613, 1)
    y4 = (1015, 489, 1)
    y6 = (772, 769, 1)
    y7 = (865, 825, 1)
    y8 = (959, 700, 1)
    y9 = (298, 75, 1)
    y10 = (251, 121, 1)
    y11 = (371, 137, 1)
    y12 = (415, 89, 1)
    y14 = (288, 325, 1)
    y15 = (399, 343, 1)
    y16 = (433, 289, 1)
    y18 = (135, 320, 1)
    y19 = (526, 529, 1)
    y20 = (744, 346, 1)
    y22 = (163, 427, 1)
    y23 = (537, 642, 1)
    y24 = (734, 456, 1)

    # odredjivanje nevidljivih temena

    # x8
    a = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(x2, x6), VektorskiProizvod(x3, x7)), x4)
    b = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(x6, x5), VektorskiProizvod(x2, x1)), x7)    
    x8 = np.array(VektorskiProizvod(a,b))
    x8 = tuple(x8 / x8[2])
    #print(x8)

    # x16
    a = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(x9, x13), VektorskiProizvod(x10, x14)), x12)
    b = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(x14, x13), VektorskiProizvod(x10, x9)), x15)    
    x16 = np.array(VektorskiProizvod(a,b))
    x16 = tuple(x16 / x16[2])
    #print(x16)

    # x24
    a = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(x18, x22), VektorskiProizvod(x19, x23)), x20)
    b = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(x22, x21), VektorskiProizvod(x18, x17)), x23)    
    x24 = np.array(VektorskiProizvod(a,b))
    x24 = tuple(x24 / x24[2])
    #print(x24)

    # y5
    a = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(y2, y6), VektorskiProizvod(y3, y7)), y1)
    b = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(y7, y6), VektorskiProizvod(y3, y2)), y8)    
    y5 = np.array(VektorskiProizvod(a,b))
    y5 = tuple(y5 / y5[2])
    #print(y5)

    # y13
    a = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(y10, y14), VektorskiProizvod(y11, y15)), y9)
    b = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(y15, y14), VektorskiProizvod(y11, y10)), y16)    
    y13 = np.array(VektorskiProizvod(a,b))
    y13 = tuple(y13 / y13[2])
    #print(y13)

    # y17
    a = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(y19, y20), VektorskiProizvod(y23, y24)), y18)
    b = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(y23, y22), VektorskiProizvod(y19, y18)), y20)    
    y17 = np.array(VektorskiProizvod(a,b))
    y17 = tuple(y17 / y17[2])
    #print(y17)

    # y21
    a = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(y19, y23), VektorskiProizvod(y18, y22)), y17)
    b = VektorskiProizvod(VektorskiProizvod(VektorskiProizvod(y23, y22), VektorskiProizvod(y19, y18)), y24)    
    y21 = np.array(VektorskiProizvod(a,b))
    y21 = tuple(y21 / y21[2])
    #print(y21)

    xx = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24]
    yy = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24]

    print("Koriscene sve tacke za odredjivanje fundamentalne matrice, nevidljive tacke izracunate")
    T1, T2 = odrediKanonskeMatrice(xx, yy)

    tackeProstora = rekonstruisiTackeProstora(xx, yy, T1, T2)
    print(f"Dobijene su tacke prostora:\n{tackeProstora}")

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x_values = [tackeProstora[0][0], tackeProstora[1][0]]
    y_values = [tackeProstora[0][1], tackeProstora[1][1]]
    z_values = [tackeProstora[0][2], tackeProstora[1][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[1][0], tackeProstora[2][0]]
    y_values = [tackeProstora[1][1], tackeProstora[2][1]]
    z_values = [tackeProstora[1][2], tackeProstora[2][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[2][0], tackeProstora[3][0]]
    y_values = [tackeProstora[2][1], tackeProstora[3][1]]
    z_values = [tackeProstora[2][2], tackeProstora[3][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[3][0], tackeProstora[0][0]]
    y_values = [tackeProstora[3][1], tackeProstora[0][1]]
    z_values = [tackeProstora[3][2], tackeProstora[0][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[4][0], tackeProstora[5][0]]
    y_values = [tackeProstora[4][1], tackeProstora[5][1]]
    z_values = [tackeProstora[4][2], tackeProstora[5][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[5][0], tackeProstora[6][0]]
    y_values = [tackeProstora[5][1], tackeProstora[6][1]]
    z_values = [tackeProstora[5][2], tackeProstora[6][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[6][0], tackeProstora[7][0]]
    y_values = [tackeProstora[6][1], tackeProstora[7][1]]
    z_values = [tackeProstora[6][2], tackeProstora[7][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[7][0], tackeProstora[4][0]]
    y_values = [tackeProstora[7][1], tackeProstora[4][1]]
    z_values = [tackeProstora[7][2], tackeProstora[4][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[0][0], tackeProstora[4][0]]
    y_values = [tackeProstora[0][1], tackeProstora[4][1]]
    z_values = [tackeProstora[0][2], tackeProstora[4][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[1][0], tackeProstora[5][0]]
    y_values = [tackeProstora[1][1], tackeProstora[5][1]]
    z_values = [tackeProstora[1][2], tackeProstora[5][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[2][0], tackeProstora[6][0]]
    y_values = [tackeProstora[2][1], tackeProstora[6][1]]
    z_values = [tackeProstora[2][2], tackeProstora[6][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[3][0], tackeProstora[7][0]]
    y_values = [tackeProstora[3][1], tackeProstora[7][1]]
    z_values = [tackeProstora[3][2], tackeProstora[7][2]]

    ax.plot3D(x_values, y_values, z_values, 'blue')

    x_values = [tackeProstora[8][0], tackeProstora[9][0]]
    y_values = [tackeProstora[8][1], tackeProstora[9][1]]
    z_values = [tackeProstora[8][2], tackeProstora[9][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[9][0], tackeProstora[10][0]]
    y_values = [tackeProstora[9][1], tackeProstora[10][1]]
    z_values = [tackeProstora[9][2], tackeProstora[10][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[10][0], tackeProstora[11][0]]
    y_values = [tackeProstora[10][1], tackeProstora[11][1]]
    z_values = [tackeProstora[10][2], tackeProstora[11][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[11][0], tackeProstora[8][0]]
    y_values = [tackeProstora[11][1], tackeProstora[8][1]]
    z_values = [tackeProstora[11][2], tackeProstora[8][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[12][0], tackeProstora[13][0]]
    y_values = [tackeProstora[12][1], tackeProstora[13][1]]
    z_values = [tackeProstora[12][2], tackeProstora[13][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[13][0], tackeProstora[14][0]]
    y_values = [tackeProstora[13][1], tackeProstora[14][1]]
    z_values = [tackeProstora[13][2], tackeProstora[14][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[14][0], tackeProstora[15][0]]
    y_values = [tackeProstora[14][1], tackeProstora[15][1]]
    z_values = [tackeProstora[14][2], tackeProstora[15][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[15][0], tackeProstora[12][0]]
    y_values = [tackeProstora[15][1], tackeProstora[12][1]]
    z_values = [tackeProstora[15][2], tackeProstora[12][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[8][0], tackeProstora[12][0]]
    y_values = [tackeProstora[8][1], tackeProstora[12][1]]
    z_values = [tackeProstora[8][2], tackeProstora[12][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[9][0], tackeProstora[13][0]]
    y_values = [tackeProstora[9][1], tackeProstora[13][1]]
    z_values = [tackeProstora[9][2], tackeProstora[13][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[10][0], tackeProstora[14][0]]
    y_values = [tackeProstora[10][1], tackeProstora[14][1]]
    z_values = [tackeProstora[10][2], tackeProstora[14][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[11][0], tackeProstora[15][0]]
    y_values = [tackeProstora[11][1], tackeProstora[15][1]]
    z_values = [tackeProstora[11][2], tackeProstora[15][2]]

    ax.plot3D(x_values, y_values, z_values, 'red')

    x_values = [tackeProstora[16][0], tackeProstora[17][0]]
    y_values = [tackeProstora[16][1], tackeProstora[17][1]]
    z_values = [tackeProstora[16][2], tackeProstora[17][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[17][0], tackeProstora[18][0]]
    y_values = [tackeProstora[17][1], tackeProstora[18][1]]
    z_values = [tackeProstora[17][2], tackeProstora[18][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[18][0], tackeProstora[19][0]]
    y_values = [tackeProstora[18][1], tackeProstora[19][1]]
    z_values = [tackeProstora[18][2], tackeProstora[19][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[19][0], tackeProstora[16][0]]
    y_values = [tackeProstora[19][1], tackeProstora[16][1]]
    z_values = [tackeProstora[19][2], tackeProstora[16][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[20][0], tackeProstora[21][0]]
    y_values = [tackeProstora[20][1], tackeProstora[21][1]]
    z_values = [tackeProstora[20][2], tackeProstora[21][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[21][0], tackeProstora[22][0]]
    y_values = [tackeProstora[21][1], tackeProstora[22][1]]
    z_values = [tackeProstora[21][2], tackeProstora[22][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[22][0], tackeProstora[23][0]]
    y_values = [tackeProstora[22][1], tackeProstora[23][1]]
    z_values = [tackeProstora[22][2], tackeProstora[23][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[23][0], tackeProstora[20][0]]
    y_values = [tackeProstora[23][1], tackeProstora[20][1]]
    z_values = [tackeProstora[23][2], tackeProstora[20][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[16][0], tackeProstora[20][0]]
    y_values = [tackeProstora[16][1], tackeProstora[20][1]]
    z_values = [tackeProstora[16][2], tackeProstora[20][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[17][0], tackeProstora[21][0]]
    y_values = [tackeProstora[17][1], tackeProstora[21][1]]
    z_values = [tackeProstora[17][2], tackeProstora[21][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[18][0], tackeProstora[22][0]]
    y_values = [tackeProstora[18][1], tackeProstora[22][1]]
    z_values = [tackeProstora[18][2], tackeProstora[22][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    x_values = [tackeProstora[19][0], tackeProstora[23][0]]
    y_values = [tackeProstora[19][1], tackeProstora[23][1]]
    z_values = [tackeProstora[19][2], tackeProstora[23][2]]

    ax.plot3D(x_values, y_values, z_values, 'green')

    plt.title('3D rekonstrukcija')
    plt.show()

if __name__ == "__main__":
    #np.set_printoptions(suppress=True)
    #primer1()
    primer2()