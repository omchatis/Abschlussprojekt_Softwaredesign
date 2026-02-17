import math
import numpy as np

import structure

class Spring:
    def __init__(self, spring_id, i, j, EA, L0):
        self.id = spring_id
        self.i = i
        self.j = j
        self.EA = float(EA)
        self.L0 = float(L0)

        if self.L0 <= 0:
            raise ValueError("Länge der Feder ist Null oder negativ")

        self.k = self.EA / self.L0

    def other(self, node_id):
        if node_id == self.i:
            return self.j
        if node_id == self.j:
            return self.i
        raise ValueError(f"Node {node_id} gehört nicht zu Spring {self.id}")

    def __repr__(self):
        return (f"Spring(id={self.id}, i={self.i}, j={self.j}, "
                f"L0={self.L0}, k={self.k})")

    def local_stiffness(self, structure):
        ni = structure.nodes[self.i]
        nj = structure.nodes[self.j]

        dx = nj.x - ni.x
        dz = nj.z - ni.z
        L = math.sqrt(dx**2 + dz**2)

        c = dx / L
        s = dz / L

        k = self.EA / L

        return k * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]
        ])
    
    def strain_energy(self, structure, u):
        ni = structure.nodes[self.i]
        nj = structure.nodes[self.j]

        dx = nj.x - ni.x
        dz = nj.z - ni.z
        L = (dx**2 + dz**2)**0.5

        c = dx / L if L > 0 else 0
        s = dz / L

        ui = np.array([u[2*self.i], u[2*self.i+1]])
        uj = np.array([u[2*self.j], u[2*self.j+1]])

        delta_L = c*(uj[0] - ui[0]) + s*(uj[1] - ui[1])

        return 0.5 * self.k * delta_L**2
