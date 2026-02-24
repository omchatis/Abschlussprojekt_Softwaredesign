import numpy as np
from structure import Structure
import optimizer
import math

if __name__ == "__main__":
    st = Structure()

    n0 = st.add_node(0, 0, bc=(True, True))     # Festlager
    n1 = st.add_node(1, 0, bc=(False, True))    # Loslager (nur z fixiert)
    n2 = st.add_node(0, 1)
    n3 = st.add_node(1, 1, force=(0, -1))


    # horizontale
    st.add_spring(n0, n1)
    st.add_spring(n2, n3)

    # vertikale
    st.add_spring(n0, n2)
    st.add_spring(n1, n3)

    # diagonale
    st.add_spring(n0, n3)
    st.add_spring(n1, n2)

    mass_reduction_factor = 0.75
    start_mass = len(st.nodes)
    target_mass = math.ceil(mass_reduction_factor * start_mass)

    while len(st.nodes) > target_mass:
        u = st.solve()

        print("Verschiebungen:")
        for i in range(len(u)//2):
            print(f"Node {i}: ux = {u[2*i]:.6f}, uz = {u[2*i+1]:.6f}")

        K = st.assemble_global_stiffness()
        print("Symmetrisch:", np.allclose(K, K.T))

        spring_energies = st.compute_strain_energies(u)
        print("Dehnungsenergien:")
        for sid, energy in spring_energies.items():
            print(f"Spring {sid}: {energy:.6f}")



        u_from_solver_py = st.solve_with_solver_py()

        print("Verschiebungen mit Solver.py:")
        for i in range(len(u_from_solver_py)//2):
            print(f"Node {i}: ux = {u_from_solver_py[2*i]:.6f}, uz = {u_from_solver_py[2*i+1]:.6f}")
        
        
        ranked_nodal_energies = optimizer.rank_nodes_by_energy(st, spring_energies)

        removed_node = optimizer.remove_one_node(st, spring_energies)
        
        if removed_node is not None:
            print(f"Knoten {removed_node} wurde entfernt.")
            print("Aktuelle Knoten:", list(st.nodes.keys()))
        else:
            print("Es konnte kein Knoten entfernt werden.")
            break