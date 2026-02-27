import math
from collections import deque

# Gewichte – werden von app.py überschrieben
_W1 = 2.0
_W2 = 3.0
_W3 = 4.0


def num_to_remove(st, mass_reduction_factor, start_mass):
    current_mass = len(st.nodes)
    target_mass = math.ceil(mass_reduction_factor * start_mass)
    return current_mass - target_mass


def is_protected_node(st, node_id: int) -> bool:
    node = st.nodes[node_id]
    has_bc    = (node.bc[0] == True) or (node.bc[1] == True)
    has_force = (node.force[0] != 0.0) or (node.force[1] != 0.0)
    return has_bc or has_force


def protected_nodes(st) -> set:
    return {nid for nid in st.nodes.keys() if is_protected_node(st, nid)}


def avg_nodal_energies(st, spring_energies):
    E_node = {}
    for nid in st.nodes.keys():
        incident = list(st.incident_springs(nid))
        number_springs = len(incident) if len(incident) > 0 else 1
        E_avg = 0.0
        for sid in incident:
            E_avg += 0.5 * spring_energies[sid] / number_springs
        E_node[nid] = E_avg
    return E_node


def bfs_distances(st, start_nodes: set) -> dict:
    """BFS von mehreren Startknoten – gibt minimale Graph-Distanz zu start_nodes zurück."""
    dist = {nid: float("inf") for nid in st.nodes}
    q = deque()
    for s in start_nodes:
        if s in dist:
            dist[s] = 0
            q.append(s)
    while q:
        cur = q.popleft()
        for nb in st.neighbors(cur):
            if dist[nb] == float("inf"):
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def weighted_nodal_energies(st, spring_energies):
    """
    Erweiterte Knotenenergie mit zwei Faktoren:
    1. Direktheit: Je kürzer der Graph-Pfad zwischen Kraft- und Lagerknoten,
       desto höher der Faktor (Knoten auf dem Lastpfad werden geschützt).
    2. Hochenergiefeder: Wenn eine anliegende Feder > 75% des Maximums hat,
       wird ein zusätzlicher Schutzfaktor addiert.
    """
    E_avg = avg_nodal_energies(st, spring_energies)

    # Kraft- und Lagerknoten identifizieren
    force_nodes   = {nid for nid, n in st.nodes.items()
                     if abs(n.force[0]) > 0 or abs(n.force[1]) > 0}
    support_nodes = {nid for nid, n in st.nodes.items()
                     if n.bc[0] or n.bc[1]}

    # BFS-Distanzen von Kraft- und Lagerknoten
    dist_force   = bfs_distances(st, force_nodes)
    dist_support = bfs_distances(st, support_nodes)

    E_weighted = {}
    for nid in st.nodes:
        d_f = dist_force.get(nid, float("inf"))
        d_s = dist_support.get(nid, float("inf"))

        # Faktor 1: Direktheit (0..1, höher = direkter auf Lastpfad)
        combined_dist = d_f + d_s
        direktheit = 1.0 / (combined_dist + 1)

        # Nur Direktheitsfaktor – w2 entfernt da kontraproduktiv
        w1 = min(_W1, 1.0)
        E_weighted[nid] = E_avg[nid] * (1 + w1 * direktheit)

    return E_weighted


def rank_nodes_by_energy(st, spring_energies: dict, solver_fn=None):
    E_weighted = weighted_nodal_energies(st, spring_energies)
    protected  = protected_nodes(st)
    items      = [(nid, E) for nid, E in E_weighted.items() if nid not in protected]
    items.sort(key=lambda x: x[1])
    return items


def bulk_remove_initial(st, spring_energies: dict, total_to_remove: int, bulk_fraction: float = 1/3) -> list:
    """
    Phase 1: Entfernt direkt 1/3 der Zielanzahl auf einmal.
    Nur Konnektivitaets- und Lagerpruefung - kein Solver-Aufruf.
    Gibt Liste der entfernten Knoten-IDs zurueck.
    """
    n_bulk  = max(1, int(total_to_remove * bulk_fraction))
    ranking = rank_nodes_by_energy(st, spring_energies)
    removed = []

    for nid, E in ranking:
        if len(removed) >= n_bulk:
            break
        snap = snapshot_node_removal(st, nid)
        st.remove_node(nid)

        if is_connected_one_piece(st) and has_support(st):
            removed.append(nid)
        else:
            undo_node_removal(st, snap)

    return removed


def is_connected_one_piece(st) -> bool:
    if len(st.nodes) <= 1:
        return True
    start   = next(iter(st.nodes.keys()))
    visited = {start}
    q       = deque([start])
    while q:
        cur = q.popleft()
        for nb in st.neighbors(cur):
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return len(visited) == len(st.nodes)


def has_support(st) -> bool:
    return any(any(n.bc) for n in st.nodes.values())


def solve_works(st, solver_fn=None) -> bool:
    """FIX: solver_fn wird von app.py übergeben damit der Remap-Solver genutzt wird."""
    try:
        if solver_fn is not None:
            u = solver_fn(st)
            return u is not None
        else:
            _ = st.solve()
            return True
    except Exception:
        return False


def snapshot_node_removal(st, node_id: int) -> dict:
    node_obj  = st.nodes[node_id]
    incident  = list(st.incident_springs(node_id))
    spring_objs = {sid: st.springs[sid] for sid in incident}

    affected_nodes = {node_id}
    for sp in spring_objs.values():
        affected_nodes.add(sp.i)
        affected_nodes.add(sp.j)

    saved_groups = {nid: set(st.node_spring_group.get(nid, set())) for nid in affected_nodes}

    return {
        "node_id":     node_id,
        "node_obj":    node_obj,
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


def remove_one_node(st, spring_energies: dict, solver_fn=None):
    ranking = rank_nodes_by_energy(st, spring_energies)

    for nid, E in ranking:
        snap = snapshot_node_removal(st, nid)
        st.remove_node(nid)

        ok = (
            is_connected_one_piece(st)
            and has_support(st)
            and solve_works(st, solver_fn)
        )

        if ok:
            return nid

        undo_node_removal(st, snap)

    return None