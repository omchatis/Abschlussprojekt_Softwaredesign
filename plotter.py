import plotly.graph_objects as go
import numpy as np


def plot_structure_interactive(
    structure,
    selected_node=None,
    displacements=None,
    scale=1.0,
    show_labels=True,
    strain_energies=None
):
    fig = go.Figure()

    # ── 1+2. Federn ───────────────────────────────────────────────────────────
    # Wenn Verschiebungen vorhanden: nur verformte Struktur zeigen
    # Wenn Energien vorhanden: jede Feder einzeln mit Farbe
    # Sonst: alle in einem Trace

    if displacements is not None:
        # Verformte Federn mit Energiefärbung
        e_vals = list(strain_energies.values()) if strain_energies else []
        e_max  = max(e_vals) if e_vals else 1
        e_max  = e_max if e_max > 0 else 1

        for spring in structure.springs.values():
            ni = structure.nodes[spring.i]
            nj = structure.nodes[spring.j]
            xi = ni.x + scale * displacements[2 * spring.i]
            zi = ni.z + scale * displacements[2 * spring.i + 1]
            xj = nj.x + scale * displacements[2 * spring.j]
            zj = nj.z + scale * displacements[2 * spring.j + 1]

            if strain_energies is not None:
                t = strain_energies.get(spring.id, 0) / e_max
                r = int(50  + 200 * t)
                g = int(120 - 100 * t)
                b = int(220 - 200 * t)
                w = 1.0 + 3.5 * t
                color = f"rgba({r},{g},{b},0.85)"
            else:
                color = "rgba(100,140,255,0.6)"
                w = 1.0

            fig.add_trace(go.Scatter(
                x=[xi, xj], y=[zi, zj],
                mode="lines",
                line=dict(color=color, width=w),
                hoverinfo="skip",
                showlegend=False
            ))
    else:
        # Keine Verschiebungen: unverformte Struktur hellblau
        xs, ys = [], []
        for spring in structure.springs.values():
            ni = structure.nodes[spring.i]
            nj = structure.nodes[spring.j]
            xs += [ni.x, nj.x, None]
            ys += [ni.z, nj.z, None]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(color="rgba(100,140,255,0.4)", width=1.0),
                hoverinfo="skip",
                showlegend=False
            ))

    # ── 3. Knotentrace ────────────────────────────────────────────────────────
    x_nodes, y_nodes, labels, colors, sizes = [], [], [], [], []

    for node in structure.nodes.values():
        if displacements is not None:
            px = node.x + scale * displacements[2 * node.id]
            pz = node.z + scale * displacements[2 * node.id + 1]
        else:
            px = node.x
            pz = node.z

        x_nodes.append(px)
        y_nodes.append(pz)
        labels.append(
            f"Node {node.id}<br>"
            f"x₀={node.x}, z₀={node.z}<br>"
            f"bc={node.bc}<br>"
            f"F={node.force}"
        )

        if node.id == selected_node:
            colors.append("#FF3B3B"); sizes.append(16)
        elif node.bc[0] or node.bc[1]:
            colors.append("#F59E0B"); sizes.append(13)
        elif abs(node.force[0]) > 0 or abs(node.force[1]) > 0:
            colors.append("#10B981"); sizes.append(13)
        else:
            colors.append("#3B82F6"); sizes.append(11)

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(color="white", width=1.5)),
        text=labels,
        hovertemplate="%{text}<extra></extra>",
        name="nodes",
        showlegend=False
    ))

    # ── 4. Colorbar für Dehnungsenergie ───────────────────────────────────────
    if strain_energies is not None:
        e_max = max(strain_energies.values()) if strain_energies else 1
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                colorscale=[[0, "rgba(50,120,220,0.85)"], [1, "rgba(250,20,20,0.85)"]],
                cmin=0, cmax=round(e_max, 4),
                color=[0],
                colorbar=dict(
                    title=dict(text="Energie", font=dict(color="white")),
                    thickness=12,
                    len=0.6,
                    x=1.02,
                    tickfont=dict(color="white"),
                ),
                showscale=True
            ),
            hoverinfo="skip",
            showlegend=False
        ))

    # ── 5. Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        yaxis=dict(autorange="reversed", dtick=1, showgrid=True,
                   gridcolor="rgba(200,200,200,0.2)"),
        xaxis=dict(dtick=1, showgrid=True,
                   gridcolor="rgba(200,200,200,0.2)"),
        dragmode="pan",
        showlegend=False,
        clickmode="event",
        hovermode="closest",
        plot_bgcolor="rgba(15,15,25,1)",
        paper_bgcolor="rgba(15,15,25,1)",
        font=dict(color="white"),
        margin=dict(l=20, r=60, t=20, b=20),
        height=520
    )

    return fig