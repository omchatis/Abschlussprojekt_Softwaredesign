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
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} existiert nicht.")
        incident = list(self.node_spring_group.get(node_id, set()))
        for sid in incident:
            self.remove_spring(sid)
        self.node_spring_group.pop(node_id, None)
        self.nodes.pop(node_id)

    def add_spring(self, i, j, EA=1.0, spring_id=None, tol=1e-8):
        ni = self.nodes[i]
        nj = self.nodes[j]
        if i == j:
            raise ValueError("Spring endpoints müssen unterschiedlich sein")
        if i not in self.nodes or j not in self.nodes:
            raise KeyError("Beide Knoten müssen existieren")

        if spring_id is None:
            spring_id = self._next_spring_id
            self._next_spring_id += 1
        else:
            if spring_id in self.springs:
                raise ValueError(f"Spring id {spring_id} existiert schon")
            self._next_spring_id = max(self._next_spring_id, spring_id + 1)

        dx = nj.x - ni.x
        dz = nj.z - ni.z
        L0 = math.sqrt(dx*dx + dz*dz)

        s = Spring(spring_id, i, j, EA, L0)
        self.springs[spring_id] = s
        self.node_spring_group.setdefault(i, set()).add(spring_id)
        self.node_spring_group.setdefault(j, set()).add(spring_id)
        return spring_id

    def remove_spring(self, spring_id):
        if spring_id not in self.springs:
            raise KeyError(f"Spring {spring_id} existiert nicht")
        s = self.springs.pop(spring_id)
        if s.i in self.node_spring_group:
            self.node_spring_group[s.i].discard(spring_id)
        if s.j in self.node_spring_group:
            self.node_spring_group[s.j].discard(spring_id)

    def incident_springs(self, node_id):
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} existiert nicht")
        return set(self.node_spring_group.get(node_id, set()))

    def number_of_incident_springs(self, node_id):
        return len(self.node_spring_group.get(node_id, set()))

    def neighbors(self, node_id):
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} existiert nicht")
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

    def assemble_global_stiffness(self, node_ids, node_index):
        size = 2 * len(node_ids)
        K = np.zeros((size, size))
        for s in self.springs.values():
            k_local = s.local_stiffness(self)
            ii = node_index[s.i]
            jj = node_index[s.j]
            dofs = [2*ii, 2*ii+1, 2*jj, 2*jj+1]
            for a in range(4):
                for b in range(4):
                    K[dofs[a], dofs[b]] += k_local[a, b]
        return K

    def assemble_force_vector(self, node_ids, node_index):
        F = np.zeros(2 * len(node_ids))
        for nid, node in self.nodes.items():
            idx = node_index[nid]
            F[2*idx]     = node.force[0]
            F[2*idx + 1] = node.force[1]
        return F

    def fixed_dof_indices(self, node_index):
        fixed_dofs = []
        for nid, n in self.nodes.items():
            idx = node_index[nid]
            if n.bc[0]: fixed_dofs.append(2*idx)
            if n.bc[1]: fixed_dofs.append(2*idx + 1)
        return fixed_dofs

    def solve_with_solver_py(self):
        node_ids, node_index = self._build_node_index()
        K = self.assemble_global_stiffness(node_ids, node_index)
        F = self.assemble_force_vector(node_ids, node_index)
        fixed_dof = self.fixed_dof_indices(node_index)
        u = solve_with_solver_py(K.copy(), F.copy(), fixed_dof)
        return u, node_ids, node_index

    def compute_strain_energies(self, u_full):
        """
        u_full ist von run_solver: u_full[2*nid] = ux, u_full[2*nid+1] = uz
        node_index ist Identity (nid -> nid) damit spring.strain_energy
        u[2*node_index[nid]] = u[2*nid] korrekt liest.
        """
        node_index_id = {nid: nid for nid in self.nodes}
        energies = {}
        for sid, spring in self.springs.items():
            energies[sid] = spring.strain_energy(self, u_full, node_index_id)
        return energies

    def current_locations_nodes(self, u, node_index, pos0, all_ids):
        frame = {}
        for nid in all_ids:
            if nid not in self.nodes:
                frame[nid] = (np.nan, np.nan)
            else:
                idx = node_index[nid]
                x0, z0 = pos0[nid]
                frame[nid] = (x0 + u[2*idx], z0 + u[2*idx+1])
        return frame