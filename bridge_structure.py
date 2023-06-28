# Interface de l'implémentation du mobile
class MobileImplementation:
    def move_mobile(self, start, end):
        pass

    def get_position(self):
        pass

# Implémentation 1 : Utilise les coordonnées cartésiennes
class CartesianMobileImplementation(MobileImplementation):
    def __init__(self):
        self.position = (0, 0)

    def move_mobile(self, start, end):
        print("Déplacement cartésien :")
        print(f"Départ : ({start[0]}, {start[1]})")
        print(f"Arrivée : ({end[0]}, {end[1]})")
        self.position = end

    def get_position(self):
        return self.position

# Implémentation 2 : Utilise les coordonnées polaires
class PolarMobileImplementation(MobileImplementation):
    def __init__(self):
        self.position = (0, 0)

    def move_mobile(self, start, end):
        print("Déplacement polaire :")
        print(f"Départ : angle {start[0]}, rayon {start[1]}")
        print(f"Arrivée : angle {end[0]}, rayon {end[1]}")
        self.position = end

    def get_position(self):
        return self.position

# Classe abstraite du mobile
class Mobile:
    def __init__(self, implementation):
        self.implementation = implementation

    def move(self, start, end):
        self.implementation.move_mobile(start, end)

    def get_position(self):
        return self.implementation.get_position()



# Utilisation
start = (0, 0)
end = (5, 5)

cartesian_mobile = Mobile(CartesianMobileImplementation())
cartesian_mobile.move(start, end)
print("Position d'arrivée :", cartesian_mobile.get_position())

polar_mobile = Mobile(PolarMobileImplementation())
polar_mobile.move(start, end)
print("Position d'arrivée :", polar_mobile.get_position())