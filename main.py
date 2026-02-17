import numpy as np
from structure import Structure

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


    u = st.solve()
    print("Verschiebungen:")
    for i in range(len(u)//2):
        print(f"Node {i}: ux = {u[2*i]:.6f}, uz = {u[2*i+1]:.6f}")

    K = st.assemble_global_stiffness()
    print("Symmetrisch:", np.allclose(K, K.T))

    energies = st.compute_strain_energies(u)
    print("Dehnungsenergien:")
    for sid, energy in energies.items():
        print(f"Spring {sid}: {energy:.6f}")