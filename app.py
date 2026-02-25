import streamlit as st
from grid import create_grid
from plotter import plot_structure_interactive

st.set_page_config(page_title="Topologieoptimierung 2D")

st.title("Topologieoptimierung")

if "structure" not in st.session_state:
    st.session_state.structure = None

if "selected_node" not in st.session_state:
    st.session_state.selected_node = None

st.sidebar.header("Modellparameter")

width = st.sidebar.number_input("Breite", min_value=2, value=4)
height = st.sidebar.number_input("Höhe", min_value=2, value=4)

if st.sidebar.button("Grid erzeugen"):
    st.session_state.structure = create_grid(width, height)

if "selected_node" not in st.session_state:
    st.session_state.selected_node = None


if st.session_state.structure is not None:

    fig = plot_structure_interactive(
        st.session_state.structure,
        selected_node=st.session_state.selected_node
    )

    selected = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun"
    )

    if (
        selected
        and isinstance(selected, dict)
        and "selection" in selected
        and selected["selection"]
        and "points" in selected["selection"]
        and selected["selection"]["points"]
    ):
        point = selected["selection"]["points"][0]
        selected_node_id = int(point["text"])
        st.session_state.selected_node = selected_node_id

    if st.session_state.selected_node is not None:

        st.sidebar.subheader("Knoten bearbeiten")

        node = st.session_state.structure.nodes[
            st.session_state.selected_node
        ]

        fix_x = st.sidebar.checkbox("Fixiere X", value=node.bc[0])
        fix_z = st.sidebar.checkbox("Fixiere Z", value=node.bc[1])

        Fx = st.sidebar.number_input("Fx", value=float(node.force[0]))
        Fz = st.sidebar.number_input("Fz", value=float(node.force[1]))

        if st.sidebar.button("Änderungen übernehmen"):
            node.bc = (fix_x, fix_z)
            node.force = (Fx, Fz)
            st.sidebar.success("Knoten aktualisiert")

if (
    "selected_node" in st.session_state
    and st.session_state.selected_node is not None
):

    st.sidebar.subheader("Knoten bearbeiten")

    node_id = st.session_state.selected_node
    node = st.session_state.structure.nodes[node_id]

    st.sidebar.write(f"Knoten ID: {node_id}")

    # Randbedingungen
    fix_x = st.sidebar.checkbox("Fixiere X", value=node.bc[0])
    fix_z = st.sidebar.checkbox("Fixiere Z", value=node.bc[1])

    # Kräfte
    Fx = st.sidebar.number_input("Fx", value=float(node.force[0]))
    Fz = st.sidebar.number_input("Fz", value=float(node.force[1]))

    if st.sidebar.button("Änderungen übernehmen"):
        node.bc = (fix_x, fix_z)
        node.force = (Fx, Fz)
        st.sidebar.success("Knoten aktualisiert")