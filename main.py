import numpy as np
from structure import Structure
import optimizer
import math

if __name__ == "__main__":
    st = Structure()

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

        spring_energies = st.compute_strain_energies(u)
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
