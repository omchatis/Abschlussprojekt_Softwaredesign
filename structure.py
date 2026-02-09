import math
from node import Node
from spring import Spring


class Structure:
    def __init__(self):
        
        self.nodes = {}
        self.springs = {}

        #Dictionary für Knoten und alle mit diesem verbundenen Federn
        self.node_spring_group = {}

        # ID des ersten Knoten/Feder ist 0
        self._next_node_id = 0
        self._next_spring_id = 0


    def add_node(self, x, z, bc=(False, False), force=(0.0, 0.0), node_id=None):
        
        # Fügt Knoten hinzu mit Standard: bc=(False,False) => frei, force=(0,0) => keine äußere Kraft.
    
        if node_id is None:
            node_id = self._next_node_id
            self._next_node_id += 1
        else:
            if node_id in self.nodes:
                raise ValueError(f"Node id {node_id} existiert schon.")
            self._next_node_id = max(self._next_node_id, node_id + 1)

        self.nodes[node_id] = Node(node_id, x, z, bc=bc, force=force)
        self.node_spring_group.setdefault(node_id, set())
        return node_id

    def remove_node(self, node_id):
        
        #Löscht einen Knoten UND alle Federn, die an ihm hängen.
        
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} existiert nicht.")

        #alle incident springs löschen
        incident = list(self.node_spring_group.get(node_id, set()))
        for sid in incident:
            self.remove_spring(sid)

        #node entfernen
        self.node_spring_group.pop(node_id, None)
        self.nodes.pop(node_id)

  
    def add_spring(self, i, j, EA=1.0, spring_id=None, tol=1e-8):
        
        #Fügt eine Feder zwischen den Nodes i und j hinzu.

        ni = self.nodes[i]
        nj = self.nodes[j]

        if i == j:
            raise ValueError("Spring endpoints müssen unterscheidlich sein")
        if i not in self.nodes or j not in self.nodes:
            raise KeyError("Beide Konten müssen existieren")

        if spring_id is None:
            spring_id = self._next_spring_id
            self._next_spring_id += 1
        else:
            if spring_id in self.springs:
                raise ValueError(f"Spring id {spring_id} existstiert schon")
            self._next_spring_id = max(self._next_spring_id, spring_id + 1)

        dx = nj.x - ni.x
        dz = nj.z - ni.z
        L0 = math.sqrt(dx*dx + dz*dz)

        # Feder erzeugen
        s = Spring(spring_id, i, j, EA, L0, tol=tol)

        self.springs[spring_id] = s
        self.node_spring_group.setdefault(i, set()).add(spring_id)
        self.node_spring_group.setdefault(j, set()).add(spring_id)
        return spring_id

    def remove_spring(self, spring_id):
        
        #Löscht genau diese Feder
        
        if spring_id not in self.springs:
            raise KeyError(f"Spring {spring_id} existiert nicht")

        s = self.springs.pop(spring_id)

        
        if s.i in self.node_spring_group:
            self.node_spring_group[s.i].discard(spring_id)
        if s.j in self.node_spring_group:
            self.node_spring_group[s.j].discard(spring_id)


    def incident_springs(self, node_id):
        #Gibt die Spring-IDs zurück, die an node_id hängen
        
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} existiert nicht")
        return set(self.node_spring_group.get(node_id, set()))

    def neighbors(self, node_id):
        
        #Gibt die Nachbar-Node-IDs zurück
        
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id}existiert nicht")
        nbrs = set()
        for sid in self.node_spring_group.get(node_id, set()):
            s = self.springs[sid]
            nbrs.add(s.other(node_id))
        return nbrs

    def __repr__(self):
        return f"Structure(nodes={len(self.nodes)}, springs={len(self.springs)})"
