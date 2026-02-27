import streamlit as st
import numpy as np
import copy
from grid import create_grid
from plotter import plot_structure_interactive
from streamlit_plotly_events import plotly_events
import solver as slv
import optimizer as opt

st.set_page_config(page_title="Topologieoptimierung 2D", layout="wide")

# CSS
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

# Session State
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
    ("w1", 0.5),
    ("opt_bulk_done", False),
    ("bulk_fraction", 0.15),
    ("param_results", []),
    ("opt_steps_todo", 0),
    ("opt_bulk_queue", []),
    ("opt_until_unstable", False),
    ("opt_frames", []),  # Liste von PNG-Bytes pro Schritt
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Hilfsfunktion: Solver
def run_solver(structure):
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

    max_id = max(structure.nodes.keys())
    u_full = np.zeros(2 * (max_id + 1))
    for nid, idx in id_to_idx.items():
        u_full[2*nid]   = u_compact[2*idx]
        u_full[2*nid+1] = u_compact[2*idx+1]

    return u_full

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Modellparameter")
    width  = st.number_input("Breite",  min_value=2, max_value=200, value=4)
    height = st.number_input("Höhe",    min_value=2, max_value=200, value=4)
    EA     = st.number_input("Federsteifigkeit EA", min_value=0.1, max_value=10000.0, value=100.0, step=10.0)

    if st.button("🔲 Grid erzeugen", use_container_width=True):
        st.session_state.structure       = create_grid(width, height, EA=EA)
        st.session_state.selected_node   = None
        st.session_state.displacements   = None
        st.session_state.strain_energies = None
        st.session_state.opt_running     = False
        st.session_state.opt_paused      = False
        st.session_state.opt_iteration   = 0
        st.session_state.opt_start_mass  = None
        st.session_state.opt_log         = []
        st.session_state.opt_bulk_done   = False
        st.session_state.opt_bulk_queue  = []
        st.session_state.opt_frames      = []

    st.divider()

    # Knoten auswählen
    if st.session_state.structure is not None:
        st.markdown("## 🔍 Knoten auswählen")
        max_node_id = max(st.session_state.structure.nodes.keys())
        st.session_state.manual_id = min(st.session_state.manual_id, max_node_id)
        st.session_state.manual_id = st.number_input(
            "Knoten ID eingeben",
            min_value=0, max_value=max_node_id,
            value=st.session_state.manual_id, step=1
        )
        if st.button("Knoten auswählen", use_container_width=True):
            if st.session_state.manual_id in st.session_state.structure.nodes:
                st.session_state.selected_node = st.session_state.manual_id
                st.rerun()
            else:
                st.warning(f"Knoten {st.session_state.manual_id} existiert nicht.")

    # Knoten bearbeiten
    if st.session_state.structure is not None and st.session_state.selected_node is not None:
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

            ecol1, ecol2 = st.columns(2)
            fix_x = ecol1.checkbox("Fixiere X", value=bool(node.bc[0]), key=f"fix_x_{node_id}")
            fix_z = ecol2.checkbox("Fixiere Z", value=bool(node.bc[1]), key=f"fix_z_{node_id}")
            Fx = st.number_input("Fx", value=float(node.force[0]), key=f"Fx_{node_id}")
            Fz = st.number_input("Fz", value=float(node.force[1]), key=f"Fz_{node_id}")

            if st.button("✅ Änderungen übernehmen", use_container_width=True, key=f"apply_{node_id}"):
                node.bc    = (fix_x, fix_z)
                node.force = (Fx, Fz)
                st.session_state.displacements   = None
                st.session_state.strain_energies = None
                st.success(f"Knoten {node_id} aktualisiert")

    st.divider()

    # Simulation
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

    scale = 1.0
    if st.session_state.displacements is not None:
        u = st.session_state.displacements
        key_nids = [nid for nid, n in st.session_state.structure.nodes.items()
                    if abs(n.force[0]) > 0 or abs(n.force[1]) > 0 or n.bc[0] or n.bc[1]]
        if key_nids:
            ku = max(max(abs(u[2*nid]), abs(u[2*nid+1]))
                     for nid in key_nids if 2*nid+1 < len(u))
            st.caption(f"Max. Verschiebung (Kraft/Lager): {ku:.5f}")
        else:
            st.caption(f"Max. Verschiebung: {float(np.max(np.abs(u))):.5f}")
        # Konditionszahl anzeigen
        try:
            id_to_idx = {nid: i for i, nid in enumerate(st.session_state.structure.nodes.keys())}
            n = len(st.session_state.structure.nodes)
            K = np.zeros((2*n, 2*n))
            for s in st.session_state.structure.springs.values():
                k_local = s.local_stiffness(st.session_state.structure)
                ii, jj = id_to_idx[s.i], id_to_idx[s.j]
                dofs = [2*ii, 2*ii+1, 2*jj, 2*jj+1]
                for a in range(4):
                    for b in range(4):
                        K[dofs[a], dofs[b]] += k_local[a, b]
            fixed_set = set()
            for nid, node in st.session_state.structure.nodes.items():
                idx = id_to_idx[nid]
                if node.bc[0]: fixed_set.add(2*idx)
                if node.bc[1]: fixed_set.add(2*idx+1)
            free = [i for i in range(2*n) if i not in fixed_set]
            cond = np.linalg.cond(K[np.ix_(free, free)])
            if cond < 1e6:
                st.caption(f"Konditionszahl: {cond:.1e} ✅")
            elif cond < 1e10:
                st.caption(f"Konditionszahl: {cond:.1e} ⚠️ schlecht konditioniert")
            else:
                st.caption(f"Konditionszahl: {cond:.1e} ❌ numerisch instabil")
        except Exception:
            pass

    st.divider()

    # Optimierung
    st.markdown("## 🔧 Optimierung")

    tab_opt, tab_params = st.tabs(["Optimierung", "Parameter"])

    with tab_opt:
        # Modus: Ziel oder bis zur Instabilität
        st.session_state.opt_until_unstable = st.toggle(
            "Bis zur Instabilität",
            value=st.session_state.opt_until_unstable,
            help="Entfernt Knoten solange bis keiner mehr entfernt werden kann"
        )

        if st.session_state.opt_until_unstable:
            mass_reduction = 0.0
            st.caption("→ Läuft bis zur Stabilitätsgrenze")
        else:
            mass_reduction = st.slider(
                "Verbleibende Knoten (%)", 0.1, 0.95, 0.8,
                help="0.8 = 80% der Knoten bleiben übrig"
            )
            n_total = len(st.session_state.structure.nodes) if st.session_state.structure else 0
            if n_total > 0:
                n_keep   = int(mass_reduction * n_total)
                n_remove = n_total - n_keep
                st.caption(f"→ {n_keep} behalten, {n_remove} entfernen")

        bcol1, bcol2 = st.columns(2)

        if bcol1.button("▶ Start", use_container_width=True):
            if st.session_state.structure is None:
                st.warning("Zuerst Grid erzeugen.")
            elif st.session_state.displacements is None:
                st.warning("Zuerst Simulation starten.")
            else:
                if st.session_state.opt_start_mass is None:
                    st.session_state.opt_start_mass = len(st.session_state.structure.nodes)
                st.session_state.opt_bulk_done  = False
                st.session_state.opt_bulk_queue = []
                st.session_state.opt_steps_todo = 0
                st.session_state.opt_running    = True
                st.session_state.opt_paused     = False

        if bcol2.button("⏸ Pause", use_container_width=True):
            st.session_state.opt_paused  = True
            st.session_state.opt_running = False

        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.opt_running    = False
            st.session_state.opt_paused     = False
            st.session_state.opt_start_mass = None
            st.session_state.opt_iteration  = 0
            st.session_state.opt_log        = []
            st.session_state.opt_bulk_done  = False
            st.session_state.opt_bulk_queue = []
            st.session_state.opt_frames     = []

        # Schritt-Buttons
        n_nodes = len(st.session_state.structure.nodes) if st.session_state.structure else 0
        step_sizes = [1]
        for s in [2, 5, 10, 20]:
            if n_nodes >= s * 3:
                step_sizes.append(s)
        step_cols = st.columns(len(step_sizes))
        for i, s in enumerate(step_sizes):
            if step_cols[i].button(f"+{s}", use_container_width=True, key=f"step_{s}"):
                if st.session_state.structure is not None and st.session_state.displacements is not None:
                    if st.session_state.opt_start_mass is None:
                        st.session_state.opt_start_mass = len(st.session_state.structure.nodes)
                        st.session_state.opt_bulk_done  = False
                        st.session_state.opt_bulk_queue = []
                    st.session_state.opt_steps_todo = s
                    st.session_state.opt_running    = True
                    st.session_state.opt_paused     = False

        if st.session_state.opt_running:
            st.info(f"🔄 Läuft… Iteration {st.session_state.opt_iteration}")
        elif st.session_state.opt_paused:
            st.warning(f"⏸ Pausiert bei Iteration {st.session_state.opt_iteration}")

        if st.session_state.opt_log:
            with st.expander("Optimierungslog", expanded=False):
                for line in st.session_state.opt_log[-20:]:
                    st.text(line)

        # Download
        if st.session_state.opt_frames:
            n_frames = len(st.session_state.opt_frames)
            st.caption(f"{n_frames} Frames gespeichert")
            dcol1, dcol2 = st.columns(2)

            # ZIP Download
            import io, zipfile
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, frame_bytes in enumerate(st.session_state.opt_frames):
                    zf.writestr(f"frame_{idx:04d}.png", frame_bytes)
            zip_buf.seek(0)
            dcol1.download_button(
                "📦 ZIP", data=zip_buf,
                file_name="optimierung.zip", mime="application/zip",
                use_container_width=True
            )

            # GIF Download
            try:
                from PIL import Image
                import io as _io
                frames_pil = []
                for fb in st.session_state.opt_frames:
                    img = Image.open(_io.BytesIO(fb)).convert("RGB")
                    # auf max 800px skalieren für kleinere GIFs
                    img.thumbnail((800, 600))
                    frames_pil.append(img)
                gif_buf = _io.BytesIO()
                frames_pil[0].save(
                    gif_buf, format="GIF",
                    save_all=True,
                    append_images=frames_pil[1:],
                    duration=200,   # ms pro Frame
                    loop=0
                )
                gif_buf.seek(0)
                dcol2.download_button(
                    "🎞️ GIF", data=gif_buf,
                    file_name="optimierung.gif", mime="image/gif",
                    use_container_width=True
                )
            except ImportError:
                dcol2.caption("PIL fehlt für GIF")

    with tab_params:
        st.markdown("**Gewichtungen**")
        st.session_state.w1 = st.slider("w1 – Direktheit (0=aus, 1=max)", 0.0, 1.0,
                                         st.session_state.w1, 0.05)
        st.session_state.bulk_fraction = st.slider("Bulk-Anteil (Phase 1)", 0.0, 0.8,
                                                    st.session_state.bulk_fraction, 0.05)
        st.divider()
        st.markdown("**Parametersuche** (w1 × 11 Kombinationen)")
        st.caption("Testet w1 von 0.0 bis 1.0 und zeigt welcher Wert am besten abschneidet.")

        if st.button("🔍 Suche starten", use_container_width=True):
            if st.session_state.structure is None or st.session_state.displacements is None:
                st.warning("Erst Grid erzeugen und Simulation starten.")
            else:
                import optimizer as _opt

                combos_w1 = [round(x * 0.1, 1) for x in range(11)]
                results   = []
                base_mass   = len(st.session_state.structure.nodes)
                target_mass = max(3, int(0.7 * base_mass))
                bf          = st.session_state.bulk_fraction

                progress    = st.progress(0)
                status_text = st.empty()

                def key_disp(u, structure):
                    key_nids = [nid for nid, n in structure.nodes.items()
                                if abs(n.force[0]) > 0 or abs(n.force[1]) > 0
                                or n.bc[0] or n.bc[1]]
                    if not key_nids or u is None:
                        return 9999.0
                    vals = [max(abs(u[2*nid]), abs(u[2*nid+1]))
                            for nid in key_nids if 2*nid+1 < len(u)]
                    return max(vals) if vals else 9999.0

                u_ref  = st.session_state.displacements
                max_u0 = key_disp(u_ref, st.session_state.structure)

                for i, w1 in enumerate(combos_w1):
                    status_text.text(f"Teste w1={w1}…")
                    st_copy = copy.deepcopy(st.session_state.structure)
                    u_copy  = u_ref.copy()
                    se_copy = dict(st.session_state.strain_energies)

                    _opt._W1 = w1

                    # Phase 1: Bulk
                    to_remove = base_mass - target_mass
                    _opt.bulk_remove_initial(st_copy, se_copy, to_remove, bulk_fraction=bf)
                    u_new = run_solver(st_copy)
                    if u_new is not None and float(np.max(np.abs(u_new))) < 1e6:
                        se_copy = st_copy.compute_strain_energies(u_new)
                        u_copy  = u_new

                    # Phase 2: Iterativ
                    iters = 0
                    while len(st_copy.nodes) > target_mass and iters < (base_mass - target_mass):
                        removed = _opt.remove_one_node(st_copy, se_copy, solver_fn=run_solver)
                        if removed is None:
                            break
                        u_new = run_solver(st_copy)
                        if u_new is None or float(np.max(np.abs(u_new))) > 1e6:
                            break
                        se_copy = st_copy.compute_strain_energies(u_new)
                        u_copy  = u_new
                        iters  += 1

                    max_u = key_disp(u_copy, st_copy)
                    score = max_u0 / (max_u + 1e-9)
                    results.append({
                        "w1": w1,
                        "u_key": round(max_u, 6),
                        "Δu/u₀": round(max_u / max(max_u0, 1e-9), 3),
                        "score": round(score, 4)
                    })
                    progress.progress((i + 1) / len(combos_w1))

                _opt._W1 = st.session_state.w1  # restoren
                results.sort(key=lambda x: -x["score"])
                st.session_state.param_results = results
                best = results[0]
                status_text.text(f"✅ Fertig! Beste w1={best['w1']} (Score={best['score']})")

        if st.session_state.param_results:
            st.markdown("**Ergebnisse** (nach Score sortiert)")
            if st.button("✅ Beste Parameter übernehmen", use_container_width=True):
                st.session_state.w1 = st.session_state.param_results[0]["w1"]
                st.rerun()
            import pandas as pd
            df = pd.DataFrame(st.session_state.param_results)
            st.dataframe(df, use_container_width=True, hide_index=True)

# Hauptbereich
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
    if not st.session_state.opt_running:
        selected_points = plotly_events(
            fig, click_event=True, hover_event=False,
            select_event=False, key="plot"
        )
        if selected_points:
            point   = selected_points[0]
            click_x = point.get("x")
            click_y = point.get("y")
            if click_x is not None and click_y is not None:
                u = st.session_state.displacements
                best_id, best_dist = None, float("inf")
                for nid, node in st.session_state.structure.nodes.items():
                    px = node.x + u[2*nid]   if u is not None else node.x
                    pz = node.z + u[2*nid+1] if u is not None else node.z
                    dist = (px - click_x)**2 + (pz - click_y)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_id   = nid
                if best_id is not None and best_id != st.session_state.selected_node:
                    st.session_state.selected_node = best_id
                    st.rerun()
    else:
        st.plotly_chart(fig, use_container_width=True)

    def _save_frame(ss, structure):
        """Rendert die aktuelle Struktur und speichert als PNG-Bytes."""
        try:
            import io
            _fig = plot_structure_interactive(
                structure,
                selected_node   = ss.selected_node,
                displacements   = ss.displacements,
                scale           = 1.0,
                strain_energies = ss.strain_energies,
            )
            png_bytes = _fig.to_image(format="png", width=900, height=600)
            ss.opt_frames.append(png_bytes)
        except Exception:
            pass  # kaleido nicht installiert – frames überspringen

    # Optimierungsschritt
    if st.session_state.opt_running and st.session_state.strain_energies is not None:
        ss         = st.session_state
        structure  = ss.structure
        start_mass = ss.opt_start_mass
        until_unstable = ss.opt_until_unstable

        opt._W1   = ss.w1
        to_remove = opt.num_to_remove(structure, mass_reduction, start_mass) if not until_unstable else 9999

        # Bulk-Queue beim ersten Start befüllen
        # Bei until_unstable: kein Bulk – von Anfang an sorgfältig iterieren
        if not ss.opt_bulk_done:
            if not until_unstable and to_remove > 3:
                n_bulk  = max(1, int(to_remove * ss.bulk_fraction))
                ranking = opt.rank_nodes_by_energy(structure, ss.strain_energies)
                queue   = [nid for nid, _ in ranking[:n_bulk]]
                ss.opt_bulk_queue = queue
            ss.opt_bulk_done = True

        def do_one_step():
            """Entfernt genau einen Knoten. Gibt True bei Erfolg zurück."""

            # Bulk-Queue abarbeiten
            while ss.opt_bulk_queue:
                nid = ss.opt_bulk_queue.pop(0)
                if nid not in structure.nodes:
                    continue
                snap = opt.snapshot_node_removal(structure, nid)
                structure.remove_node(nid)
                if opt.is_connected_one_piece(structure) and opt.has_support(structure):
                    ss.opt_iteration += 1
                    ss.opt_log.append(f"Iter. {ss.opt_iteration}: Knoten {nid} entfernt")
                    u = run_solver(structure)
                    if u is not None and float(np.max(np.abs(u))) < 1e6:
                        ss.displacements   = u
                        ss.strain_energies = structure.compute_strain_energies(u)
                        _save_frame(ss, structure)
                    else:
                        # Solver fehlgeschlagen: alte Energien bereinigen
                        ss.strain_energies = {sid: e for sid, e in ss.strain_energies.items()
                                              if sid in structure.springs}
                    return True
                else:
                    opt.undo_node_removal(structure, snap)
                    continue  # nächsten Kandidaten versuchen

            # Bulk fertig: Energien bereinigen (entfernte Federn raus)
            ss.strain_energies = {sid: e for sid, e in ss.strain_energies.items()
                                  if sid in structure.springs}

            # Normale Iteration
            to_rem = opt.num_to_remove(structure, mass_reduction, start_mass) if not until_unstable else 9999
            if to_rem <= 0:
                ss.opt_running    = False
                ss.opt_steps_todo = 0
                ss.opt_log.append(f"✅ Ziel erreicht nach {ss.opt_iteration} Iterationen.")
                return False

            removed = opt.remove_one_node(structure, ss.strain_energies, solver_fn=run_solver)
            if removed is None:
                ss.opt_running    = False
                ss.opt_steps_todo = 0
                msg = "🏁 Stabilitätsgrenze erreicht" if until_unstable else f"⛔ Kein Knoten entfernbar (Iter. {ss.opt_iteration})"
                ss.opt_log.append(msg)
                return False

            ss.opt_iteration += 1
            ss.opt_log.append(f"Iter. {ss.opt_iteration}: Knoten {removed} entfernt")

            u = run_solver(structure)
            if u is None:
                ss.opt_running = False; ss.opt_steps_todo = 0
                ss.opt_log.append("⛔ Solver fehlgeschlagen."); return False
            if float(np.max(np.abs(u))) > 1e6:
                ss.opt_running = False; ss.opt_steps_todo = 0
                ss.opt_log.append(f"⚠️ Instabil bei Iter. {ss.opt_iteration}."); return False
            ss.displacements   = u
            ss.strain_energies = structure.compute_strain_energies(u)
            _save_frame(ss, structure)
            return True

        # Schleife läuft bis N *erfolgreiche* Entfernungen (nicht N Versuche)
        steps_target = ss.opt_steps_todo if ss.opt_steps_todo > 0 else 1
        did = 0
        while did < steps_target:
            if not do_one_step():
                break
            did += 1

        if ss.opt_steps_todo > 0:
            ss.opt_steps_todo = max(0, ss.opt_steps_todo - did)
            if ss.opt_steps_todo == 0:
                ss.opt_running = False

        st.rerun()

    else:
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