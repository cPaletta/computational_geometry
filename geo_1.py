import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math


class Punkt:
    def __init__(self, x, y):
        self.x=x
        self.y=y

class Linia:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        if (self.p2.x - self.p1.x) != 0:
            self.a = (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)
            self.b = self.p1.y - self.a * self.p1.x
        else:
            self.a = float('inf')  # pionowa linia
            self.b = self.p1.x  
        
class Trojkat:
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3


def RownanieProstej(linia):
    px = [linia.p1.x, linia.p2.x]
    py = [linia.p1.y, linia.p2.y]

#rysowanie 
    plt.plot(px, py)
    plt.show()
    return 0

def przynaleznoscPunktuDoProstej(punkt, linia):

    x = np.linspace(-100, 100)

    y = linia.a * x + linia.b
    plt.plot(x, y)

    plt.scatter(punkt.x, punkt.y, c="g")
    plt.show()

    if punkt.y == linia.a*punkt.x + linia.b:
        return "Punkt nalezy do prostej"
    else:
        return "Punkt nie nalezy do prostej"

def przynaleznoscPunktuDoOdcinka(punkt,linia):

    px = [linia.p1.x, linia.p2.x]
    py = [linia.p1.y, linia.p2.y]

    plt.plot(px, py)
    plt.scatter(punkt.x, punkt.y, c="g")
    plt.show()

    if punkt.y == linia.a*punkt.x + linia.b and punkt.x < max(p1) and punkt.x > min(px):
        return "Punkt nalezy do odcinka"
    else:
        return "Punkt nie nalezy do odcinka"

def polozeniePunktuWzgledemProstej(punkt, linia):

    # x = np.linspace(-100, 100)

    # y = linia.a * x + linia.b
    # plt.plot(x, y)

    # plt.scatter(punkt.x, punkt.y, c="g")
    # plt.show()

    det = (linia.p2.x - linia.p1.x) * (punkt.y - linia.p1.y) - (linia.p2.y - linia.p1.y) * (punkt.x - linia.p1.x)
    if det > 0:
        # print("L")
        return 1  # Punkt po lewej stronie prostej
    elif det < 0:
        # print("P")
        return 2  # Punkt po prawej stronie prostej
    else:
        return 0  # Punkt na prostej

def translacjaOdcinkaOPodanyWektor(linia,wektor):

    px = [linia.p1.x, linia.p2.x]
    py = [linia.p1.y, linia.p2.y]

    plt.plot(px, py)

    px = [p + wektor[0] for p in px]
    py = [p + wektor[1] for p in py]

    plt.plot(px, py, c="g")

    plt.show()

def odbiciePunktuWzgledemProstej(punkt, linia):

    # Współczynnik kierunkowy prostej prostopadłej do danej prostej
    ap = -1 / linia.a
    c = punkt.y - ap * punkt.x

    x_przeciecia = (c - linia.b) / (linia.a - ap)
    y_przeciecia = linia.a * x_przeciecia + linia.b

    punktxp = 2 * x_przeciecia - punkt.x
    punktyp = 2 * y_przeciecia - punkt.y

    px = [linia.p1.x, linia.p2.x]
    py = [linia.p1.y, linia.p2.y]
    plt.plot(px, py)

    plt.scatter(punkt.x, punkt.y, color='r')
    plt.scatter(punktxp, punktyp, color='g')

    plt.show()

def punktPrzecieciaDwochProstych(linia, linia2):
    punktp = Punkt(0,0)
    punktp.x = (linia2.b-linia.b)/(linia.a-linia2.a)
    punktp.y = linia.a*punktp.x + linia.b
    print(punktp.x)
    print(punktp.y)

    x = np.linspace(punktp.x -1, punktp.x+1, 400)
    y = linia.a * x + linia.b
    y2 = linia2.a * x + linia2.b
    plt.plot(x, y, label='Linia 1')
    plt.plot(x, y2, label='Linia 2')

    plt.scatter(punktp.x, punktp.y, c="g")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Punkt przeciecia dwoch prostych ')
    plt.grid(True)
    plt.show()



def punktPrzecieciaDwochProstychWspolczynniki(A1, B1, C1, A2, B2, C2):
    x_min = -100
    x_max = 100

    x = np.linspace(x_min, x_max, 100)

    y1 = - (A1 * x + C1) / B1
    y2 = - (A2 * x + C2) / B2

    plt.plot(x, y1, label='Linia 1')
    plt.plot(x, y2, label='Linia 2')


    W = A1*B2 - A2*B1
    if W == 0:
        print("Proste sie nie przecinaja")
    else:
        Wx = (-C1*B2) - (-C2*B1)
        Wy = (A1*-C2) - (A2*-C1)
        x = Wx/W
        y = Wy/W
        print(x)
        print(y)
        plt.scatter(x,y, c = "r")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Przeciecie dwoch prostych z wspolczynnika')
    plt.grid(True)
    plt.show()

def mierzenieOdleglosci(linia,punkt):

    x = np.linspace(punkt.x - 20, punkt.x + 20)
    y = linia.a * x + linia.b

    odl=abs(linia.a*punkt.x - punkt.y + linia.b)/math.sqrt((linia.a*linia.a)+1)

    x_values = np.linspace(punkt.x - 10, punkt.x + 10, 400)
    y_values = linia.a * x_values + linia.b
    plt.plot(x_values, y_values, label='Linia 1')
    plt.scatter(punkt.x, punkt.y, c='r')
    print(odl)
    plt.xlim(punkt.x - 5, punkt.x + 5)
    plt.ylim(punkt.y - 5, punkt.y + 5)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Odleglosc')
    plt.grid(True)
    plt.show()
    return odl

def punktPrzeciecia(linia1, linia2):
    if linia1.a == linia2.a:
        return None  # równoległe lub pokrywają się

    if linia1.a == float('inf'):
        x = linia1.b
        y = linia2.a * x + linia2.b
        return Punkt(x, y)
    elif linia2.a == float('inf'):
        x = linia2.b
        y = linia1.a * x + linia1.b
        return Punkt(x, y)
    else:
        x = (linia2.b - linia1.b) / (linia1.a - linia2.a)
        y = linia1.a * x + linia1.b
        return Punkt(x, y)

def dlugoscOdcinka(p1, p2):
    return (math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2))






def poleTrojkata(trojkat):
    return 0.5 * abs(trojkat.p1.x * (trojkat.p2.y - trojkat.p3.y) + trojkat.p2.x * (trojkat.p3.y - trojkat.p1.y) + trojkat.p3.x * (trojkat.p1.y - trojkat.p2.y))

def poleTrojkataPunkty(p1, p2, p3):
    return 0.5 * abs(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

def czyPunktNalezyDoTrojkata(punkt, trojkat):
    # Obliczenie pola całego trójkąta
    pole_calego_trojkata = poleTrojkata(trojkat)
    
    # Obliczenie pól trzech trójkątów utworzonych z danego punktu
    pole_t1 = poleTrojkataPunkty(punkt, trojkat.p1, trojkat.p2)
    pole_t2 = poleTrojkataPunkty(punkt, trojkat.p2, trojkat.p3)
    pole_t3 = poleTrojkataPunkty(punkt, trojkat.p3, trojkat.p1)
    
    # Sprawdzenie, czy suma pól trzech mniejszych trójkątów jest równa polu całego trójkąta
    if (pole_t1 + pole_t2 + pole_t3)  == pole_calego_trojkata: 
        return True
    else:
        return False

def stworzTrojkat(linia1, linia2, linia3, punkt):
    p1 = punktPrzeciecia(linia1, linia2)
    p2 = punktPrzeciecia(linia1, linia3)
    p3 = punktPrzeciecia(linia2, linia3)

    if not p1 or not p2 or not p3 or p1 == p2 or p1 == p3 or p2 == p3:
        print("Nie można utworzyć trójkąta.")
        return


    trojkat = Trojkat(p1, p2, p3)


    a = dlugoscOdcinka(p1, p2)
    b = dlugoscOdcinka(p2, p3)
    c = dlugoscOdcinka(p1, p3)
    s = (a + b + c) / 2
    print(math.sqrt((s*(s-a)*(s-b)*(s-c))))

    # Wizualizacja
    x_values = np.linspace(min(p1.x, p2.x, p3.x) - 1, max(p1.x, p2.x, p3.x) + 1, 400)
    plt.plot(x_values, linia1.a * x_values + linia1.b, label='Linia 1')
    plt.plot(x_values, linia2.a * x_values + linia2.b, label='Linia 2')
    plt.plot(x_values, linia3.a * x_values + linia3.b, label='Linia 3')

    plt.scatter([p1.x, p2.x, p3.x], [p1.y, p2.y, p3.y], c='red')
    # Sprawdzanie przez Pola:
    nalezy = czyPunktNalezyDoTrojkata(punkt, trojkat)
    plt.scatter([punkt.x],[punkt.y], c='g')
    plt.text(punkt.x, punkt.y, ' Punkt\nwewnątrz' if nalezy else ' Punkt\nna zewnątrz', ha='left', va='bottom')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Wizualizacja trójkąta')
    plt.grid(True)
    plt.show()
    return poleTrojkata(trojkat)

def stworzTrojkatSprawdzLewoPrawo(linia1, linia2, linia3, punkt):
    p1 = punktPrzeciecia(linia1, linia2)
    p2 = punktPrzeciecia(linia1, linia3)
    p3 = punktPrzeciecia(linia2, linia3)

    if not p1 or not p2 or not p3 or p1 == p2 or p1 == p3 or p2 == p3:
        print("Nie można utworzyć trójkąta.")
        return


    trojkat = Trojkat(p1, p2, p3)


    a = dlugoscOdcinka(p1, p2)
    b = dlugoscOdcinka(p2, p3)
    c = dlugoscOdcinka(p1, p3)
    s = (a + b + c) / 2
    print(math.sqrt((s*(s-a)*(s-b)*(s-c))))

    # Wizualizacja
    x_values = np.linspace(min(p1.x, p2.x, p3.x) - 1, max(p1.x, p2.x, p3.x) + 1, 400)
    plt.plot(x_values, linia1.a * x_values + linia1.b, label='Linia 1')
    plt.plot(x_values, linia2.a * x_values + linia2.b, label='Linia 2')
    plt.plot(x_values, linia3.a * x_values + linia3.b, label='Linia 3')

    plt.scatter([p1.x, p2.x, p3.x], [p1.y, p2.y, p3.y], c='red')

    if polozeniePunktuWzgledemProstej(punkt, linia1) == 1 and polozeniePunktuWzgledemProstej(punkt, linia2) == 1 and polozeniePunktuWzgledemProstej(punkt, linia3) == 1:
        nalezy = True
    elif polozeniePunktuWzgledemProstej(punkt, linia1) == 2 and polozeniePunktuWzgledemProstej(punkt, linia2) == 2 and polozeniePunktuWzgledemProstej(punkt, linia3) == 2:
        nalezy = True
    else:
        nalezy =False
    plt.scatter([punkt.x],[punkt.y], c='g')
    plt.text(punkt.x, punkt.y, ' Punkt\nwewnątrz' if nalezy else ' Punkt\nna zewnątrz', ha='left', va='bottom')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Wizualizacja trójkąta')
    plt.grid(True)
    plt.show()
    return poleTrojkata(trojkat)

def katPomiedzyWektorami(vec1, vec2):
    # Iloczyn skalarny wektorów
    dot_product = np.dot(vec1, vec2)
    
    # Normy wektorów
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Obliczenie cosinusa kąta
    cos_theta = dot_product / (norm_vec1 * norm_vec2)
    
    # Obliczenie kąta w radianach i konwersja na stopnie
    theta = math.acos(cos_theta)
    theta_degrees = math.degrees(theta)
    
    return theta_degrees

def wizualizujKatPomiedzyLiniami(linia1, linia2):
    # Tworzenie wektorów z punktów linii
    vec1 = [linia1.p2.x - linia1.p1.x, linia1.p2.y - linia1.p1.y]
    vec2 = [linia2.p2.x - linia2.p1.x, linia2.p2.y - linia2.p1.y]
    
    # Obliczanie kąta
    kat = katPomiedzyWektorami(vec1, vec2)
    x = np.linspace(-100, 100)

    y = linia1.a * x + linia1.b
    plt.plot(x, y)

    y2 = linia2.a * x + linia2.b
    plt.plot(x, y2)
    plt.legend()
    plt.title(f'Kąt pomiędzy liniami: {kat:.2f} stopni')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

class Wielokat:
    def __init__(self):
        self.wierzcholki = []
    
    def dodajPunkt(self, punkt):
        self.wierzcholki.append(punkt)

    def czyPunktNalezy(self, punkt):
        licznik_przeciec = 0
        n = len(self.wierzcholki)
    
        for i in range(n):
            p1 = self.wierzcholki[i]
            p2 = self.wierzcholki[(i + 1) % n]  
            
            if ((punkt.y > min(p1.y, p2.y) and punkt.y < max(p1.y, p2.y)) or  # Przecięcie między wierzchołkami (bez końców)
                (punkt.y == p1.y and p1.y == p2.y and punkt.x < max(p1.x, p2.x))):  # Specjalny przypadek dla krawędzi poziomej
                    if p1.y != p2.y:  # Uniknięcie dzielenia przez 0
                        xinters = (punkt.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                    else:  # Dla krawędzi poziomej
                        xinters = punkt.x + 1  
                    
                    if punkt.x <= xinters:
                        licznik_przeciec += 1
        
        return licznik_przeciec % 2 != 0


def wizualizujWielokatIWeryfikujPunkt(wielokat, punkt):
    fig, ax = plt.subplots()
    
    wierzcholki = [[p.x, p.y] for p in wielokat.wierzcholki]
    polygon = patches.Polygon(wierzcholki, closed=True, edgecolor='r', facecolor='none')
    ax.add_patch(polygon)
    
    ax.plot(punkt.x, punkt.y, 'o', color='blue')
    nalezy = wielokat.czyPunktNalezy(punkt)
    plt.text(punkt.x, punkt.y, ' Punkt\nwewnątrz' if nalezy else ' Punkt\nna zewnątrz', ha='left', va='bottom')
    
    plt.grid(True)
    plt.show()


p1 = Punkt(0,0)
p2 = Punkt(1,1)
p3 = Punkt(0,1)
p4 = Punkt(1,0)
p5 = Punkt(-1,-1)
p6 = Punkt(0.75,2)
p7 = Punkt(2, 0.5)
p8 = Punkt(-0.25, 0)
p9 = Punkt(1, 0)

linia = Linia(p1,p2)
linia2 = Linia(p4,p3)
linia3 = Linia(p6, p5)
vec1 = [2,1]
vec2 = [2, -1]
wektor = [2, 1]

wielokat = Wielokat()
wielokat.dodajPunkt(p5)
wielokat.dodajPunkt(p1)
wielokat.dodajPunkt(p4)
wielokat.dodajPunkt(p2)
wielokat.dodajPunkt(p6)

print(RownanieProstej(linia))
print(przynaleznoscPunktuDoProstej(p3,linia))
print(przynaleznoscPunktuDoOdcinka(p3,linia))
print(polozeniePunktuWzgledemProstej(p3,linia))
translacjaOdcinkaOPodanyWektor(linia,wektor)
odbiciePunktuWzgledemProstej(p3,linia)
punktPrzecieciaDwochProstych(linia3, linia)
punktPrzecieciaDwochProstychWspolczynniki(-1, 2, 3, 2, 3, 4)
mierzenieOdleglosci(linia,p3)

stworzTrojkat(linia, linia2, linia3, p7)
stworzTrojkatSprawdzLewoPrawo(linia, linia2, linia3,p8)
wizualizujKatPomiedzyLiniami(linia3, linia)
wizualizujWielokatIWeryfikujPunkt(wielokat, p7) 
wizualizujWielokatIWeryfikujPunkt(wielokat, p8) 
