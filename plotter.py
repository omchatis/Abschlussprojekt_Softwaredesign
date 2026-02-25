import plotly.graph_objects as go

def plot_structure_interactive(structure, selected_node=None, show_labels=True):

    fig = go.Figure()

    # ---------------------
    # Federn
    # ---------------------
    for spring in structure.springs.values():
        ni = structure.nodes[spring.i]
        nj = structure.nodes[spring.j]

        fig.add_trace(
            go.Scatter(
                x=[ni.x, nj.x],
                y=[ni.z, nj.z],
                mode="lines",
                line=dict(color="black"),
                hoverinfo="none",
                showlegend=False
            )
        )

    # ---------------------
    # Knoten
    # ---------------------
    x_nodes = []
    y_nodes = []
    labels = []
    colors = []

    for node in structure.nodes.values():
        x_nodes.append(node.x)
        y_nodes.append(node.z)
        labels.append(str(node.id))

        if node.id == selected_node:
            colors.append("red")   # ausgewählter Knoten
        else:
            colors.append("blue")

    fig.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text" if show_labels else "markers",
            text=labels if show_labels else None,
            textposition="top center",
            marker=dict(size=10, color=colors),
            hovertemplate="Node %{text}<extra></extra>",
            showlegend=False
        )
    )

    # ---------------------
    # Layout
    # ---------------------
    fig.update_layout(
        yaxis=dict(
            autorange="reversed",
            dtick=1
        ),
        xaxis=dict(
            dtick=1
        ),
        dragmode="pan",
        clickmode="event+select",
        showlegend=False
    )

    return fig