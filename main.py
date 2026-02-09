
from structure import Structure

if __name__ == "__main__":
    st = Structure()

    n0 = st.add_node(0, 0, bc=(True, True))   # Festlager
    n1 = st.add_node(1, 0)                   # freier Knoten

    s0 = st.add_spring(n0, n1, EA=1.0)

    print(st)
    print("Nodes:", st.nodes)
    print("Springs:", st.springs)
