from structure import Structure

def create_grid(width: int, height: int):
    st = Structure()

    # 1️⃣ Knoten erzeugen
    for i in range(height):
        for j in range(width):
            st.add_node(j, i)

    # 2️⃣ Federn erzeugen
    for i in range(height):
        for j in range(width):

            node_id = i * width + j

            # horizontal
            if j < width - 1:
                right_id = i * width + (j + 1)
                st.add_spring(node_id, right_id)

            # vertikal
            if i < height - 1:
                bottom_id = (i + 1) * width + j
                st.add_spring(node_id, bottom_id)

            # diagonale rechts unten
            if i < height - 1 and j < width - 1:
                diag_id = (i + 1) * width + (j + 1)
                st.add_spring(node_id, diag_id)

            # diagonale links unten
            if i < height - 1 and j > 0:
                diag_id = (i + 1) * width + (j - 1)
                st.add_spring(node_id, diag_id)

    return st