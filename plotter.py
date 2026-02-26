import plotly.graph_objects as go


def plot_structure_interactive(
    structure,
    selected_node=None,
    displacements=None,
    scale=1.0,
    show_labels=True,
    strain_energies=None
):
    fig = go.Figure()

    # ── 1. Alle unverformten Federn in EINEM Trace ────────────────────────────
    xs, ys = [], []
    for spring in structure.springs.values():
        ni = structure.nodes[spring.i]
        nj = structure.nodes[spring.j]

        color = "rgba(100,140,255,0.4)"
        width = 1.0
        if strain_energies is not None:
            e     = strain_energies.get(spring.id, 0)
            e_max = max(strain_energies.values()) if strain_energies else 1
            if e_max > 0:
                t     = e / e_max
                r     = int(50  + 180 * t)
                g     = int(100 -  80 * t)
                b     = int(200 - 180 * t)
                color = f"rgba({r},{g},{b},0.8)"
                width = 1.5 + 3 * t

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

    # ── 2. Verformte Federn in EINEM Trace ───────────────────────────────────
    if displacements is not None:
        xd, yd = [], []
        for spring in structure.springs.values():
            ni = structure.nodes[spring.i]
            nj = structure.nodes[spring.j]
            xi = ni.x + scale * displacements[2 * spring.i]
            zi = ni.z + scale * displacements[2 * spring.i + 1]
            xj = nj.x + scale * displacements[2 * spring.j]
            zj = nj.z + scale * displacements[2 * spring.j + 1]
            xd += [xi, xj, None]
            yd += [zi, zj, None]

        if xd:
            fig.add_trace(go.Scatter(
                x=xd, y=yd,
                mode="lines",
                line=dict(color="rgba(220,50,50,0.7)", dash="dot", width=1.5),
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
        x=x_nodes,
        y=y_nodes,
        mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(color="white", width=1.5)),
        text=labels,
        hovertemplate="%{text}<extra></extra>",
        name="nodes",
        showlegend=False
    ))

    # ── 4. Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        yaxis=dict(autorange="reversed", dtick=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
        xaxis=dict(dtick=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
        dragmode="pan",
        showlegend=False,
        clickmode="event",
        hovermode="closest",        # ← Euklidischer Abstand statt nur x
        plot_bgcolor="rgba(15,15,25,1)",
        paper_bgcolor="rgba(15,15,25,1)",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=520
    )

    return fig