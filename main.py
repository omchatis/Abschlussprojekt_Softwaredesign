import numpy as np
from structure import Structure
import optimizer
import math

if __name__ == "__main__":
    st = Structure()

    n0 = st.add_node(0, 0, bc=(True, True))     # Festlager
    n1 = st.add_node(1, 0)
    n2 = st.add_node(0, 1)
    n3 = st.add_node(1, 1, force=(0, -1))
    n4 = st.add_node(2, 0, bc=(False, True), force=(1, -1))    # Loslager (nur z fixiert)
    n5 = st.add_node(2, 1)



    # horizontale
    st.add_spring(n0, n1)
    st.add_spring(n2, n3)
    st.add_spring(n1, n4)
    st.add_spring(n3, n5)

    # vertikale
    st.add_spring(n0, n2)
    st.add_spring(n1, n3)
    st.add_spring(n4, n5)

    # diagonale
    st.add_spring(n0, n3)
    st.add_spring(n1, n2)
    st.add_spring(n1, n5)
    st.add_spring(n4, n3)

    all_ids = sorted(st.nodes.keys())
    pos0 = {}
    for nid, node in st.nodes.items():
        pos0[nid] = (node.x, node.z) 

    mass_reduction_factor = 0.75
    start_mass = len(st.nodes)
    target_mass = math.ceil(mass_reduction_factor * start_mass)

    while len(st.nodes) > target_mass:
        #u = st.solve()
        u, node_ids, node_index = st.solve_with_solver_py()
        '''
        print("Verschiebungen:")
        node_ids, node_index = st._build_node_index()
        for nid in node_ids:
            idx = node_index[nid]
            print(f"Node {nid}: ux = {u[2*idx]:.6f}, uz = {u[2*idx+1]:.6f}")
        '''
        K = st.assemble_global_stiffness(node_ids, node_index)
        print("Symmetrisch:", np.allclose(K, K.T))

        spring_energies = st.compute_strain_energies(u, node_index)
        print("Dehnungsenergien:")
        for sid, energy in spring_energies.items():
            print(f"Spring {sid}: {energy:.6f}")


        print("Verschiebungen mit Solver.py:")
        for nid in node_ids:
            idx = node_index[nid]
            print(f"Node {nid}: ux = {u[2*idx]:.6f}, uz = {u[2*idx+1]:.6f}")
        
        
        ranked_nodal_energies = optimizer.rank_nodes_by_energy(st, spring_energies)

        removed_node = optimizer.remove_one_node(st, spring_energies)
        
        u, node_ids, node_index = st.solve_with_solver_py()
        if removed_node is not None:
            print(f"Knoten {removed_node} wurde entfernt.")
            print("Aktuelle Knoten:", list(st.nodes.keys()))
            Node_positions = st.current_locations_nodes(u, node_index, pos0, all_ids)
            print(f"{Node_positions}")

        else:
            print("Es konnte kein Knoten entfernt werden.")
            Node_positions = st.current_locations_nodes(u, node_index, pos0, all_ids)
            print(f"{Node_positions}")

            break
