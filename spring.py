import math

class Spring:
    def __init__(self, spring_id, i, j, EA, L0, tol=1e-8):
        self.id = spring_id
        self.i = i
        self.j = j
        self.EA = float(EA)
        self.L0 = float(L0)

        if self.L0 <= 0:
            raise ValueError("Länge der Feder ist Null oder Negativ")
        if not (
            abs(self.L0 - 1.0) < tol or
            abs(self.L0 - math.sqrt(2)) < tol
        ):
            raise ValueError(
                f"Die Länge der Feder beträgt={self.L0:.6f}. "
                "Sie muss zwischen 1 und Wurzel2 liegen"
            )
        
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
