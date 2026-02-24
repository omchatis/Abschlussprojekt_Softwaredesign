import math
from node import Node
from spring import Spring
import numpy as np
from solver import solve as solve_with_solver_py


class Structure:
    def __init__(self):
        self.nodes = {}
        self.springs = {}
        self.node_spring_group = {}

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
        s = Spring(spring_id, i, j, EA, L0)

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

    def _build_node_index(self):
        node_ids = sorted(self.nodes.keys())
        node_index = {nid: idx for idx, nid in enumerate(node_ids)}
        return node_ids, node_index
    
    def assemble_global_stiffness(self):
        node_ids, node_index = self._build_node_index()
        size = 2 * len(node_ids)
        K = np.zeros((size, size))

        for s in self.springs.values():
            k_local = s.local_stiffness(self)
            i, j = s.i, s.j

            ii = node_index[i]
            jj = node_index[j]

            dofs = [2*ii, 2*ii+1, 2*jj, 2*jj+1]

        # Einfügen der lokalen Steifigkeitsmatrix in die globale Matrix
            for a in range(4):
                for b in range(4):
                    K[dofs[a], dofs[b]] += k_local[a, b]

        return K
    
    def assemble_force_vector(self):
        node_ids, node_index = self._build_node_index()
        F = np.zeros(2 * len(node_ids))

        for nid, node in self.nodes.items():
            idx = node_index[nid]
            F[2*idx]     = node.force[0]
            F[2*idx + 1] = node.force[1]

        return F
    
    def fixed_dof_indices(self):
        node_ids, node_index = self._build_node_index()
        fixed_dofs = []
        for nid, n in self.nodes.items():
            idx = node_index[nid]
            if n.bc[0]:
                fixed_dofs.append(2*idx)
            if n.bc[1]:
                fixed_dofs.append(2*idx + 1)
        return fixed_dofs

    def solve(self):
        K = self.assemble_global_stiffness()
        F = self.assemble_force_vector()
        fixed_dofs = self.fixed_dof_indices()

        fixed = set(fixed_dofs)
        free_dofs = [i for i in range(len(F)) if i not in fixed]

        K_reduced = K[np.ix_(free_dofs, free_dofs)]
        F_reduced = F[free_dofs]

        u = np.zeros(len(F))
        u_free = np.linalg.solve(K_reduced, F_reduced)

        for idx, dof in enumerate(free_dofs):
            u[dof] = u_free[idx]

        return u
    
    def solve_with_solver_py(self):
        K = self.assemble_global_stiffness()
        F = self.assemble_force_vector()
        fixed_dof = self.fixed_dof_indices()

        u_solver_py = solve_with_solver_py(K.copy(), F.copy(), fixed_dof)

        return u_solver_py

    def compute_strain_energies(self, u):
        energies = {}
        node_ids, node_index = self._build_node_index()

        for sid, spring in self.springs.items():
          energies[sid] = spring.strain_energy(self, u, node_index)

        return energies
    
    def prioritise_energies(self):

        energies =self.compute_strain_energies()

        sorted_energies = sorted(energies.items(), key=lambda x: x[1])
