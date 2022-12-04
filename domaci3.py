import numpy as np
import math
import scipy.sparse.linalg
import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# normiranje vektora na jedinicni
def normalize(v):
    norm = np.linalg.norm(v)
    if math.isclose(norm, 0.0): 
       return v
    return v / norm


# uglovi u radijanima
def euler2a(phi,theta,psi):

    Rz = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0] , [-math.sin(theta), 0, math.cos(theta)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])

    A = np.round(np.dot(np.dot(Rz, Ry), Rx), 10)
    return A


def axisAngle(A):

    if np.allclose(A, np.eye(3, dtype=float)):
        print("Matrica je identitet (rotacija za 0 radijana oko proizvoljne prave)!")
        return
    
    #provera ortogonalnosti
    A_t = np.transpose(A)
    if not np.allclose(np.dot(A, A_t), np.eye(3, dtype=float)):
        print("Matrica nije ortogonalna!")
        return
    
    #provera vrednosti determinante
    if not math.isclose(np.linalg.det(A), 1.0):
        print("Vrednost determinante date matrice nije 1!")
        return

    #sopstveni jedinicni vektor za sopstvenu vrednost 1
    values, vectors = scipy.sparse.linalg.eigs(A, k=1, sigma=1)
    vector = [a.real for a in vectors[:, 0]]
    vector = normalize(vector)

    #proizvoljan jedinicni vektor normalan na sopstveni
    u = np.random.randn(3)
    u -= u.dot(vector) * vector
    u = normalize(u)

    #slika tog vektora (vec jedinicni)
    u_prim = np.dot(A, np.array([[u[0]], [u[1]], [u[2]]]))
    u_prim = [u_prim[0,0], u_prim[1,0], u_prim[2,0]]

    #trazeni ugao
    angle = math.acos(np.dot(u,u_prim))

    #ispitivanje orijentacije baze
    matrica = np.array([u, u_prim, vector])
    if np.linalg.det(matrica) < 0:
        vector = -vector
    
    return (vector, angle)

def rodrigez(p, angle):

    if np.allclose(p,[0,0,0]):
        return None
    
    # normiranje i prebacivanje vektora p da bude kolona zbog kasnijeg mnozenja
    p = normalize(p)    
    p = np.array([[p[0]], [p[1]], [p[2]]])

    px = np.array([[0, -p[2][0], p[1][0]], [p[2][0], 0, -p[0][0]], [-p[1][0], p[0][0], 0]])

    # transponat p je vrsta
    p_t = np.transpose(p)
    
    pp_t = np.dot(p, p_t)
    
    A = pp_t + math.cos(angle) * (np.eye(3,dtype=float) - pp_t) + math.sin(angle) * px
    return np.round(A,10)


def a2euler(A):
    #provera ortogonalnosti
    A_t = np.transpose(A)
    if not np.allclose(np.dot(A, A_t), np.eye(3, dtype=float)):
        print("Matrica nije ortogonalna!")
        return
    
    #provera vrednosti determinante
    if not math.isclose(np.linalg.det(A), 1.0):
        print("Vrednost determinante date matrice nije 1!")
        return

    if A[2][0] < 1:
        if A[2][0] > -1: #jedinstveno resenje
            psi = math.atan2(A[1][0], A[0][0])
            theta = math.asin(-A[2][0])
            phi = math.atan2(A[2][1], A[2][2])
        else: # nije jedinstveno, slucaj Ox3 = -Oz
            psi = math.atan2(-A[0][1], A[1][1])
            theta = math.pi / 2
            phi = 0
    else: # nije jedinstveno, slucaj Ox3 = Oz
        psi = math.atan2(-A[0][1], A[1][1])
        theta = -math.pi / 2
        phi = 0
    
    return phi, theta, psi

def axisAngle2Q(p, angle):
    if np.allclose(p,[0,0,0]):
        return None
    
    w = math.cos(angle/2)
    p = normalize(p)

    q = math.sin(angle/2) * p
    q = np.append(q,w)

    return q

def Q2axisAngle(q):
    q = normalize(q)

    #if q[3] < 0:  #<--- ako zelimo angle iz [0,pi]
    #  q = -q

    angle = 2 * math.acos(q[3])

    if math.isclose(abs(q[3]), 1.0) :
        p = np.array([1,0,0]) # identitet - p bilo koji jedinicni

    else:
        p = normalize(np.array([q[0], q[1], q[2]]))
    
    return p, angle

def slerp(q1,q2,tm,t):
    q1 = normalize(q1)
    q2 = normalize(q2)

    cos0 = np.dot(q1,q2)

    if cos0 < 0: # idi po kracem luku sfere
        q1 = -q1
        cos0 = -cos0
    
    if cos0 > 0.95: # kvaterioni q1 i q2 previse blizu
        return q1

    angle = math.acos(cos0)

    qtleft = math.sin(angle * (1-t/tm)) / math.sin(angle) * q1
    qtright = math.sin(angle * t/tm) / math.sin(angle) * q2
    
    return qtleft + qtright


def primer1():
    phi = -math.atan(1/4)
    theta = -math.asin(8/9)
    psi = math.atan(4)
    print(f"phi = {phi}")
    print(f"theta = {theta}")
    print(f"psi = {psi}")
    A = euler2a(phi, theta, psi)
    print(f"A = {A}")

    try:
        vector, angle = axisAngle(A)
        print(f"axis = {vector}, angle = {angle}")
    except TypeError:
        exit()
    
    A = rodrigez(vector, angle)
    if A is None:
        print("Nula vektor prosledjen kao argument!")
        exit()
    print(f"A = {A}")

    try:
        phi, theta, psi = a2euler(A)
        print(f"phi = {phi}")
        print(f"theta = {theta}")
        print(f"psi = {psi}")
    except TypeError:
        exit()

    q = axisAngle2Q(vector, angle)
    if q is None:
        print("Nula vektor prosledjen kao argument!")
        exit()
    print(f"q = {q}")

    vector, angle = Q2axisAngle(q)
    print(f"axis = {vector}, angle = {angle}")

def primer2():
    phi = math.pi / 4
    theta = math.pi / 3
    psi = math.pi / 6
    print(f"phi = {phi}")
    print(f"theta = {theta}")
    print(f"psi = {psi}")
    A = euler2a(phi, theta, psi)
    print(f"A = {A}")

    try:
        vector, angle = axisAngle(A)
        print(f"axis = {vector}, angle = {angle}")
    except TypeError:
        exit()
    
    A = rodrigez(vector, angle)
    if A is None:
        print("Nula vektor prosledjen kao argument!")
        exit()
    print(f"A = {A}")

    try:
        phi, theta, psi = a2euler(A)
        print(f"phi = {phi}")
        print(f"theta = {theta}")
        print(f"psi = {psi}")
    except TypeError:
        exit()

    q = axisAngle2Q(vector, angle)
    if q is None:
        print("Nula vektor prosledjen kao argument!")
        exit()
    print(f"q = {q}")

    vector, angle = Q2axisAngle(q)
    print(f"axis = {vector}, angle = {angle}")


def slerpPrimer():
    q1 = np.array([0,0,0,1])
    q2 = np.array([0,1/math.sqrt(2), 0, 1/math.sqrt(2)])

    qt = slerp(q1,q2,120,80)
    print(qt)

#def animation_frame(i):



def animacija():
   
    poz1 = (1,2,3)
    phi1 = math.pi/9
    theta1 = math.pi/3
    psi1 = math.pi/2

    poz2 = (5,4,3)
    phi2 = 2 * math.pi/3
    theta2 = 0
    psi2 = 3 * math.pi/4

    A = euler2a(phi1, theta1, psi1)
    vector, angle = axisAngle(A)
    q1 = axisAngle2Q(vector, angle)
    print(q1)

    A = euler2a(phi2, theta2, psi2)
    vector, angle = axisAngle(A)
    q2 = axisAngle2Q(vector, angle)
    print(q2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_zlim([0,10])
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.scatter(poz1[0], poz1[1], poz1[2])
    ax.scatter(poz2[0], poz2[1], poz2[2])

    plt.show()
    




if __name__ == '__main__': 
    #primer1()
    primer2()
    #slerpPrimer()
    #animacija()
