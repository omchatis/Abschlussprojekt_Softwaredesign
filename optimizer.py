import math
from collections import deque

def num_to_remove(st, mass_reduction_factor, start_mass):
    current_mass = len(st.nodes)
    target_mass = math.ceil(mass_reduction_factor * start_mass)
    return current_mass - target_mass
    
def is_protected_node(st, node_id: int) -> bool:
    
    #Knoten darf entfernt werden, wenn:
    #- Lagerknoten: mindestens ein bc=True
    #- Lastknoten: Kraft != (0,0)
    
    node = st.nodes[node_id]
    has_bc = (node.bc[0] == True) or (node.bc[1] == True)
    has_force = (node.force[0] != 0.0) or (node.force[1] != 0.0)
    return has_bc or has_force


def protected_nodes(st) -> set[int]:
    #Menge aller protected node_ids.
    return {nid for nid in st.nodes.keys() if is_protected_node(st, nid)}

def avg_nodal_energies(st, spring_energies):
    E_node = {}
    for nid in st.nodes.keys():
        number_springs = st.number_of_incident_springs(nid)
        E_avg = 0.0
        for sid in st.incident_springs(nid):
            E_avg += 0.5 * spring_energies[sid]/number_springs  # halb/halb auf Endknoten
        E_node[nid] = E_avg
    return E_node


def rank_nodes_by_energy(st, spring_energies: dict[int, float]):
    E_node = avg_nodal_energies(st, spring_energies)
    protected = protected_nodes(st)

    # Protected Nodes herausfiltern
    items = [(nid, E) for nid, E in E_node.items() if nid not in protected]

    # Sortieren: erst Energie (aufsteigend)
    items.sort(key=lambda x: x[1])

    return items  # [(node_id, energy), ...]

def is_connected_one_piece(st) -> bool:
    # 0 oder 1 Knoten => trivial zusammenhängend
    if len(st.nodes) <= 1:
        return True

    # Startknoten
    start = next(iter(st.nodes.keys()))

    visited = set([start])
    q = deque([start])

    while q:
        cur = q.popleft()
        for nb in st.neighbors(cur):
            if nb not in visited:
                visited.add(nb)
                q.append(nb)

    return len(visited) == len(st.nodes)

def has_support(st) -> bool:
    return any(any(n.bc) for n in st.nodes.values())
    
def solve_works(st) -> bool:
    try:
        _ = st.solve()
        return True
    except Exception:
        return False
    
def snapshot_node_removal(st, node_id: int) -> dict:
    node_obj = st.nodes[node_id]
    incident = list(st.incident_springs(node_id))
    spring_objs = {}
    for sid in incident:
        spring_objs[sid] = st.springs[sid]

    affected_nodes = {node_id}
    for sp in spring_objs.values():
        affected_nodes.add(sp.i)
        affected_nodes.add(sp.j)

    saved_groups = {nid: set(st.node_spring_group.get(nid, set())) for nid in affected_nodes}

    return {
        "node_id": node_id,
        "node_obj": node_obj,
        "spring_objs": spring_objs,
        "saved_groups": saved_groups,
    }


def undo_node_removal(st, snap: dict) -> None:
    node_id = snap["node_id"]
    st.nodes[node_id] = snap["node_obj"]

    for sid, sp in snap["spring_objs"].items():
        st.springs[sid] = sp

    for nid, group in snap["saved_groups"].items():
        st.node_spring_group[nid] = set(group)

def remove_one_node(st, spring_energies: dict[int, float]):
    ranking = rank_nodes_by_energy(st, spring_energies)

    for nid, E in ranking:
        snap = snapshot_node_removal(st, nid)

        st.remove_node(nid)

        ok = is_connected_one_piece(st) and has_support(st) and solve_works(st)

        if ok:
            return nid  # <-- WICHTIG

        undo_node_removal(st, snap)

    return None
"""
def can_remove_node(st, node_id: int) -> bool:
    snap = snapshot_node_removal(st, node_id)

    st.remove_node(node_id)

    ok = is_connected_one_piece(st) and has_support(st) and solve_works(st)

    undo_node_removal(st, snap)
    return ok

def remove_one_node(st, spring_energies: dict[int, float]) -> bool:
    protected = protected_nodes(st)
    ranking = rank_nodes_by_energy(st, spring_energies, protected)

    rejected = set()

    for nid, E in ranking:
        # Probe-Check (entfernen -> check -> undo)
        if can_remove_node(st, nid):
            # Jetzt endgültig entfernen
            st.remove_node(nid)
            return True
        else:
            rejected.add(nid)

    return False
"""