import streamlit as st
from grid import create_grid
from plotter import plot_structure_interactive
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Topologieoptimierung 2D", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0A0A14;
        color: #E2E8F0;
    }
    h1 { font-family: 'JetBrains Mono', monospace; letter-spacing: -1px; }
    .stButton > button {
        background: #1E293B;
        color: #94A3B8;
        border: 1px solid #334155;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #3B82F6;
        color: white;
        border-color: #3B82F6;
    }
    .stSidebar { background: #0F0F1E; border-right: 1px solid #1E293B; }
    .node-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    .legend-dot {
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("structure", None),
    ("selected_node", None),
    ("displacements", None),
    ("strain_energies", None),
    ("manual_id", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Modellparameter")
    width  = st.number_input("Breite",  min_value=2, max_value=20, value=4)
    height = st.number_input("Höhe",    min_value=2, max_value=20, value=4)

    if st.button("🔲 Grid erzeugen", use_container_width=True):
        st.session_state.structure       = create_grid(width, height)
        st.session_state.selected_node   = None
        st.session_state.displacements   = None
        st.session_state.strain_energies = None

    st.divider()

    if st.session_state.displacements is not None:
        scale = st.slider("Verformung skalieren", 0.0, 50.0, 10.0)
    else:
        scale = 10.0

    st.divider()
    st.markdown("**Legende**")
    st.markdown("""
    <div class='node-card'>
        <span class='legend-dot' style='background:#FF3B3B'></span> Ausgewählt<br>
        <span class='legend-dot' style='background:#F59E0B'></span> Gelagert<br>
        <span class='legend-dot' style='background:#10B981'></span> Belastet<br>
        <span class='legend-dot' style='background:#3B82F6'></span> Frei
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.structure is not None:
        st.divider()
        st.markdown("## 🔍 Knoten auswählen")
        st.session_state.manual_id = st.number_input(
            "Knoten ID eingeben",
            min_value=0,
            max_value=len(st.session_state.structure.nodes) - 1,
            value=st.session_state.manual_id,
            step=1
        )
        if st.button("Knoten auswählen", use_container_width=True):
            if st.session_state.manual_id in st.session_state.structure.nodes:
                st.session_state.selected_node = st.session_state.manual_id
                st.rerun()
            else:
                st.warning(f"Knoten {st.session_state.manual_id} existiert nicht.")

    if st.session_state.structure is not None and st.session_state.selected_node is not None:
        st.divider()
        st.markdown("## ✏️ Knoten bearbeiten")
        node_id = st.session_state.selected_node
        node    = st.session_state.structure.nodes[node_id]

        st.markdown(
            f"<div class='node-card'>Knoten ID: <b>{node_id}</b><br>x={node.x}, z={node.z}</div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        fix_x = col1.checkbox("Fixiere X", value=bool(node.bc[0]))
        fix_z = col2.checkbox("Fixiere Z", value=bool(node.bc[1]))
        Fx = st.number_input("Fx (horizontale Kraft)", value=float(node.force[0]))
        Fz = st.number_input("Fz (vertikale Kraft)",   value=float(node.force[1]))

        if st.button("✅ Änderungen übernehmen", use_container_width=True):
            node.bc    = (fix_x, fix_z)
            node.force = (Fx, Fz)
            st.success(f"Knoten {node_id} aktualisiert")

# ── Hauptbereich ──────────────────────────────────────────────────────────────
st.markdown("# Topologieoptimierung")

if st.session_state.structure is None:
    st.info("👈 Grid erzeugen um zu starten.")
else:
    fig = plot_structure_interactive(
        st.session_state.structure,
        selected_node   = st.session_state.selected_node,
        displacements   = st.session_state.displacements,
        scale           = scale,
        strain_energies = st.session_state.strain_energies,
    )

    selected_points = plotly_events(
        fig,
        click_event  = True,
        hover_event  = False,
        select_event = False,
        key          = "plot"
    )

    if selected_points:
        point   = selected_points[0]
        click_x = point.get("x")
        click_y = point.get("y")

        if click_x is not None and click_y is not None:
            best_id, best_dist = None, float("inf")
            for nid, node in st.session_state.structure.nodes.items():
                dist = (node.x - click_x) ** 2 + (node.z - click_y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_id   = nid

            if best_dist < 0.5 and best_id != st.session_state.selected_node:
                st.session_state.selected_node = best_id
                st.rerun()

    if st.session_state.displacements is not None and st.session_state.selected_node is not None:
        node = st.session_state.structure.nodes[st.session_state.selected_node]
        u    = st.session_state.displacements
        ux   = u[2 * node.id]
        uz   = u[2 * node.id + 1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Knoten", f"#{node.id}")
        col2.metric("Verschiebung ux", f"{ux:.4f}")
        col3.metric("Verschiebung uz", f"{uz:.4f}")
