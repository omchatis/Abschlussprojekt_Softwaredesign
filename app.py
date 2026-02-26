import streamlit as st
import numpy as np
import copy
from grid import create_grid
from plotter import plot_structure_interactive
from streamlit_plotly_events import plotly_events
import solver as slv
import optimizer as opt

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
        background: #1E293B; color: #94A3B8;
        border: 1px solid #334155; border-radius: 6px;
        font-family: 'JetBrains Mono', monospace; font-size: 12px;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: #3B82F6; color: white; border-color: #3B82F6; }
    .stSidebar { background: #0F0F1E; border-right: 1px solid #1E293B; }
    .node-card {
        background: #1E293B; border: 1px solid #334155; border-radius: 8px;
        padding: 12px 16px; margin: 8px 0;
        font-family: 'JetBrains Mono', monospace; font-size: 12px;
    }
    .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("structure", None),
    ("selected_node", None),
    ("displacements", None),
    ("strain_energies", None),
    ("manual_id", 0),
    ("opt_running", False),
    ("opt_paused", False),
    ("opt_iteration", 0),
    ("opt_start_mass", None),
    ("opt_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Hilfsfunktion: Solver aufrufen über structure ─────────────────────────────
def run_solver(structure):
    """Baut K und F aus der Structure und ruft solver.solve() auf.
    Nach dem Entfernen von Knoten sind IDs nicht mehr lueckenlos,
    daher remappen wir auf kompakte Indizes 0..n-1.
    """
    id_to_idx = {nid: i for i, nid in enumerate(structure.nodes.keys())}
    n = len(structure.nodes)
    K = np.zeros((2*n, 2*n))
    F = np.zeros(2*n)

    for s in structure.springs.values():
        k_local = s.local_stiffness(structure)
        ii = id_to_idx[s.i]
        jj = id_to_idx[s.j]
        dofs = [2*ii, 2*ii+1, 2*jj, 2*jj+1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += k_local[a, b]

    for nid, node in structure.nodes.items():
        idx = id_to_idx[nid]
        F[2*idx]   = node.force[0]
        F[2*idx+1] = node.force[1]

    fixed = []
    for nid, node in structure.nodes.items():
        idx = id_to_idx[nid]
        if node.bc[0]: fixed.append(2*idx)
        if node.bc[1]: fixed.append(2*idx+1)

    u_compact = slv.solve(K, F, fixed)
    if u_compact is None:
        return None

    # Ergebnis zurueck auf node.id-Raum mappen
    max_id = max(structure.nodes.keys())
    u_full = np.zeros(2 * (max_id + 1))
    for nid, idx in id_to_idx.items():
        u_full[2*nid]   = u_compact[2*idx]
        u_full[2*nid+1] = u_compact[2*idx+1]

    return u_full

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
        st.session_state.opt_running     = False
        st.session_state.opt_paused      = False
        st.session_state.opt_iteration   = 0
        st.session_state.opt_start_mass  = None
        st.session_state.opt_log         = []

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────────
    st.markdown("## ▶️ Simulation")
    if st.button("Simulation starten", use_container_width=True):
        if st.session_state.structure is not None:
            u = run_solver(st.session_state.structure)
            if u is not None:
                st.session_state.displacements   = u
                st.session_state.strain_energies = st.session_state.structure.compute_strain_energies(u)
                st.success("Simulation erfolgreich!")
            else:
                st.error("Singular – Randbedingungen prüfen.")
        else:
            st.warning("Zuerst Grid erzeugen.")

    if st.session_state.displacements is not None:
        scale = st.slider("Verformung skalieren", 0.0, 50.0, 10.0)
    else:
        scale = 10.0

    st.divider()

    # ── Optimierung ───────────────────────────────────────────────────────────
    st.markdown("## 🔧 Optimierung")

    mass_reduction = st.slider(
        "Massenreduktion", 0.1, 0.95, 0.5,
        help="Ziel: wie viel % der ursprünglichen Knoten sollen übrig bleiben"
    )

    col1, col2 = st.columns(2)

    # Start / Fortsetzen
    if col1.button("▶ Start", use_container_width=True):
        if st.session_state.structure is None:
            st.warning("Zuerst Grid erzeugen.")
        elif st.session_state.displacements is None:
            st.warning("Zuerst Simulation starten.")
        else:
            if st.session_state.opt_start_mass is None:
                st.session_state.opt_start_mass = len(st.session_state.structure.nodes)
            st.session_state.opt_running = True
            st.session_state.opt_paused  = False

    # Pause
    if col2.button("⏸ Pause", use_container_width=True):
        st.session_state.opt_paused  = True
        st.session_state.opt_running = False

    # Stop
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.opt_running    = False
        st.session_state.opt_paused     = False
        st.session_state.opt_start_mass = None
        st.session_state.opt_iteration  = 0
        st.session_state.opt_log        = []

    # Status anzeigen
    if st.session_state.opt_running:
        st.info(f"🔄 Läuft… Iteration {st.session_state.opt_iteration}")
    elif st.session_state.opt_paused:
        st.warning(f"⏸ Pausiert bei Iteration {st.session_state.opt_iteration}")

    if st.session_state.opt_log:
        with st.expander("Optimierungslog", expanded=False):
            for line in st.session_state.opt_log[-20:]:
                st.text(line)



    # ── Knoten auswählen ──────────────────────────────────────────────────────
    if st.session_state.structure is not None:
        st.divider()
        st.markdown("## 🔍 Knoten auswählen")
        max_id = max(st.session_state.structure.nodes.keys())
        # Sicherstellen dass manual_id nicht größer als max_id ist
        st.session_state.manual_id = min(st.session_state.manual_id, max_id)
        st.session_state.manual_id = st.number_input(
            "Knoten ID eingeben",
            min_value=0,
            max_value=max_id,
            value=st.session_state.manual_id,
            step=1
        )
        if st.button("Knoten auswählen", use_container_width=True):
            if st.session_state.manual_id in st.session_state.structure.nodes:
                st.session_state.selected_node = st.session_state.manual_id
                st.rerun()
            else:
                st.warning(f"Knoten {st.session_state.manual_id} existiert nicht.")

    # ── Knoten bearbeiten ─────────────────────────────────────────────────────
    if st.session_state.structure is not None and st.session_state.selected_node is not None:
        # Sicherheitscheck: node könnte durch Optimierung entfernt worden sein
        if st.session_state.selected_node not in st.session_state.structure.nodes:
            st.session_state.selected_node = None
        else:
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
                st.session_state.displacements   = None
                st.session_state.strain_energies = None
                st.success(f"Knoten {node_id} aktualisiert")

# ── Hauptbereich ──────────────────────────────────────────────────────────────
st.markdown("# Topologieoptimierung")

if st.session_state.structure is None:
    st.info("👈 Grid erzeugen um zu starten.")
else:
    # ── Optimierungsschritt ausführen ─────────────────────────────────────────
    if st.session_state.opt_running and st.session_state.strain_energies is not None:
        ss = st.session_state
        structure  = ss.structure
        start_mass = ss.opt_start_mass

        to_remove = opt.num_to_remove(structure, mass_reduction, start_mass)

        if to_remove <= 0:
            ss.opt_running = False
            ss.opt_log.append(f"✅ Ziel erreicht nach {ss.opt_iteration} Iterationen.")
        else:
            removed = opt.remove_one_node(structure, ss.strain_energies)
            if removed is None:
                ss.opt_running = False
                ss.opt_log.append(f"⛔ Kein weiterer Knoten entfernbar (Iteration {ss.opt_iteration}).")
            else:
                ss.opt_iteration += 1
                ss.opt_log.append(f"Iteration {ss.opt_iteration}: Knoten {removed} entfernt.")

                u = run_solver(structure)
                if u is not None:
                    ss.displacements   = u
                    ss.strain_energies = structure.compute_strain_energies(u)
                else:
                    ss.opt_running = False
                    ss.opt_log.append("⛔ Simulation nach Entfernung fehlgeschlagen.")

        st.rerun()

    else:
        # normales Rendering ohne Optimierungsschritt
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

        # Legende rechts neben Plot
        st.markdown("""
        <div class='node-card' style='display:inline-block; margin-top:8px'>
            <b>Legende</b><br>
            <span class='legend-dot' style='background:#FF3B3B'></span> Ausgewählt&nbsp;&nbsp;
            <span class='legend-dot' style='background:#F59E0B'></span> Gelagert&nbsp;&nbsp;
            <span class='legend-dot' style='background:#10B981'></span> Belastet&nbsp;&nbsp;
            <span class='legend-dot' style='background:#3B82F6'></span> Frei
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.displacements is not None and st.session_state.selected_node is not None:
            if st.session_state.selected_node in st.session_state.structure.nodes:
                node = st.session_state.structure.nodes[st.session_state.selected_node]
                u    = st.session_state.displacements
                ux   = u[2 * node.id]
                uz   = u[2 * node.id + 1]
                col1, col2, col3 = st.columns(3)
                col1.metric("Knoten", f"#{node.id}")
                col2.metric("Verschiebung ux", f"{ux:.4f}")
                col3.metric("Verschiebung uz", f"{uz:.4f}")
