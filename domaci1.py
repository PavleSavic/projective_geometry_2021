import numpy as np

def prebaciUHomogene(tacke):
    l = []
    for (x,y) in tacke:
        l.append((x, y, 1))
    return l

def prebaciUAfine(tacke):
    l = []
    for (x1,x2,x3) in tacke:
        l.append((int(round(x1/x3)), int(round(x2/x3))))
    return l

def VektorskiProizvod(t1,t2):
    return tuple(np.cross(list(t1), list(t2)))

def uprosecavanje(l):
    avg = [0,0]
    duzina = len(l)

    for (x,y) in l:
        avg[0] += x
        avg[1] += y
    
    return (int(round(avg[0]/ duzina)), int(round(avg[1]/ duzina)))


def Nevidljivo(p1,p2,p3,p5,p6,p7,p8):
    tacke = [p1,p2,p3,p5,p6,p7,p8]

    tacke = prebaciUHomogene(tacke)
    #                                            p2      p6                            p1      p5
    Xb1 = VektorskiProizvod(VektorskiProizvod(tacke[1],tacke[4]), VektorskiProizvod(tacke[0],tacke[3]))
    #                                            p2      p6                            p3      p7
    Xb2 = VektorskiProizvod(VektorskiProizvod(tacke[1],tacke[4]), VektorskiProizvod(tacke[2],tacke[5]))
    #                                            p1      p5                            p3      p7
    Xb3 = VektorskiProizvod(VektorskiProizvod(tacke[0],tacke[3]), VektorskiProizvod(tacke[2],tacke[5]))

    afine = prebaciUAfine([Xb1,Xb2,Xb3])

    Xb = uprosecavanje(afine)

    #                                            p5      p6                           p7       p8
    Yb1 = VektorskiProizvod(VektorskiProizvod(tacke[3],tacke[4]), VektorskiProizvod(tacke[5],tacke[6]))
    #                                            p1      p2                           p5       p6
    Yb2 = VektorskiProizvod(VektorskiProizvod(tacke[0],tacke[1]), VektorskiProizvod(tacke[3],tacke[4]))
    #                                            p1      p2                           p7       p8
    Yb3 = VektorskiProizvod(VektorskiProizvod(tacke[0],tacke[1]), VektorskiProizvod(tacke[5],tacke[6]))

    afine = prebaciUAfine([Yb1,Yb2, Yb3])

    Yb = uprosecavanje(afine)

    (Xb,Yb) = prebaciUHomogene([Xb,Yb]) 

    #                                          p8                               p3     
    p4 = VektorskiProizvod(VektorskiProizvod(tacke[6], Xb), VektorskiProizvod(tacke[2], Yb))

    return prebaciUAfine([p4])[0]

    


if __name__ == "__main__":
    #                           p1         p2         p3         p5         p6         p7         p8
    print(f"p4 = {Nevidljivo((595,301), (292,517), (157,379), (665,116), (304,295), (135,163), (509,43))}") # test primer iz uputstva

    print(f"p4 = {Nevidljivo((849,437), (419,756), (212,590), (978,140), (447,422), (155,242), (753,37))}") # primer sa slike



     

    
  


