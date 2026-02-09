from typing import Tuple

class Node:
    def __init__(self, node_id, x, z, bc=(False, False), force=(0.0, 0.0)):
        self.id = node_id
        self.x = float(x)
        self.z = float(z)

        # Randbedingungen (Boundary Conditions) - ob die Freiheitsgrade x und z für den Punkt existieren
        self.bc = tuple(bc)

        # Kraft - ob die äußere Kräfte (Fx, Fz) faufür den Punkt existieren
        self.force = tuple(force)

    def __repr__(self):
        return (f"Node(id={self.id}, x={self.x}, z={self.z}, "
                f"bc={self.bc}, force={self.force})")