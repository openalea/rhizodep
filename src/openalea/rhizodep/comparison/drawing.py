"""
Root systems drawn from a record (a Snapshot of legacy_run, or the port's in the same form), to compare them visually.

    xyz = positions(snapshot)                {vid: (start, end)}, in m
    draw_root_system(ax, snapshot, color_by="type")
    totals(snapshot)                         {"length": ..., "struct_mass": ..., "apices": ..., "lateral_apices": ...}

The geometry is rhizodep's (tools.plot_mtg and get_root_visitor), drawn by PlantGL's turtle: from the plant's
position, pointing down, each element turns down by angle_down then rolls by angle_roll (a root leaving the base first
rolls by its azimuth), bends towards the vertical by its gravitropism (gravitropism · its original radius over the
base's, Wheat-BRIDGES' 0.06 by default), and goes `length` forward. With gravitropism 0 (or without PlantGL), the same
turtle without tropism. The same function draws the legacy's records and the port's.
"""
import numpy as np

LABEL_APEX = 2
TYPE_NAMES = {1: "base", 2: "seminal support", 3: "seminal before emergence", 4: "adventitious support",
              5: "adventitious before emergence", 6: "before emergence", 7: "emerged", 8: "stopped",
              9: "just stopped", 10: "dead", 11: "just dead", 12: "nodule"}


def _rotation(axis, angle_degrees):
    a = np.radians(angle_degrees)
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(a), np.sin(a)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def positions(snapshot, origin=(0., 0., 0.), gravitropism=0.06):
    """{vid: (start, end)}: the elements' end points, by the turtle above, from *origin* (the plant's position)."""
    try:
        from openalea.plantgl.all import PglTurtle, Vector3
    except ImportError:
        return _positions_without_tropism(snapshot, origin)
    if not gravitropism:
        return _positions_without_tropism(snapshot, origin)
    props = snapshot.props
    children = {vid: [] for vid in snapshot.vertices}
    roots = []
    for vid in snapshot.vertices:
        parent = snapshot.parent[vid]
        (children[parent].append(vid) if parent in children else roots.append(vid))
    turtle = PglTurtle()
    turtle.down(180)                                   # rhizodep's plot_mtg: the graph upside down
    turtle.tropism = (0, 0, -1)
    start_state = (Vector3(*origin), turtle.getHeading(), turtle.getUp())
    radius = props.get("original_radius", props["radius"])
    out = {}
    for root in roots:
        base_radius = radius.get(root) or 1.
        stack = [(root, start_state)]
        while stack:
            vid, (position, heading, up) = stack.pop()
            turtle.move(position)
            turtle.setHead(heading, up)
            azimuth = props.get("azimuth", {}).get(vid, 0.) or 0.
            if azimuth:
                turtle.rollL(azimuth)
            turtle.down(props["angle_down"].get(vid, 0.) or 0.)
            turtle.rollL(props["angle_roll"].get(vid, 0.) or 0.)
            turtle.elasticity = gravitropism * ((radius.get(vid) or 0.) / base_radius)
            turtle.F(props["length"].get(vid, 0.) or 0.)
            end = turtle.getPosition()
            out[vid] = (np.array([position.x, position.y, position.z]), np.array([end.x, end.y, end.z]))
            state = (Vector3(end), turtle.getHeading(), turtle.getUp())
            for child in children[vid]:
                stack.append((child, state))
    return out


def _positions_without_tropism(snapshot, origin=(0., 0., 0.)):
    """positions without gravitropism (numpy, without PlantGL)."""
    props = snapshot.props
    children = {vid: [] for vid in snapshot.vertices}
    roots = []
    for vid in snapshot.vertices:
        parent = snapshot.parent[vid]
        (children[parent].append(vid) if parent in children else roots.append(vid))
    out = {}
    # the frame: heading, left
    stack = [(vid, np.array(origin, dtype=float), np.array([0., 0., -1.]), np.array([0., 1., 0.])) for vid in roots]
    while stack:
        vid, start, heading, left = stack.pop()
        azimuth = props.get("azimuth", {}).get(vid, 0.) or 0.
        if azimuth:
            left = _rotation(heading, azimuth) @ left
        pitch = _rotation(left, props["angle_down"].get(vid, 0.) or 0.)
        heading = pitch @ heading
        roll = _rotation(heading, props["angle_roll"].get(vid, 0.) or 0.)
        left = roll @ left
        end = start + heading * (props["length"].get(vid, 0.) or 0.)
        out[vid] = (start, end)
        for child in children[vid]:
            stack.append((child, end, heading, left))
    return out


def draw_root_system(ax, snapshot, color_by="type", cmap="viridis", plane=(0, 2), linewidth_by_radius=True,
                     xyz=None, vmin=None, vmax=None, origin=(0., 0., 0.)):
    """The elements as lines projected on *plane* (axes indices, x-z by default), coloured by *color_by* (a
    per-vertex property); returns the LineCollection (for a colorbar)."""
    from matplotlib.collections import LineCollection

    xyz = positions(snapshot, origin) if xyz is None else xyz
    vids = [vid for vid in snapshot.vertices if (snapshot.props["length"].get(vid, 0.) or 0.) > 0.]
    i, j = plane
    lines = [((xyz[v][0][i], xyz[v][0][j]), (xyz[v][1][i], xyz[v][1][j])) for v in vids]
    values = np.array([float(snapshot.props[color_by].get(v, np.nan)) for v in vids])
    radii = np.array([snapshot.props["radius"].get(v, 0.) for v in vids])
    widths = 0.4 + 3. * radii / radii.max() if linewidth_by_radius and len(radii) and radii.max() > 0 else 0.8
    collection = LineCollection(lines, array=values, cmap=cmap, linewidths=widths)
    if vmin is not None or vmax is not None:
        collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    ax.set_aspect("equal")
    ax.autoscale_view()
    return collection


def totals(snapshot):
    p = snapshot.props
    apices = [v for v in snapshot.vertices if p["label"].get(v) == LABEL_APEX]
    return {
        "length": sum(p["length"].get(v, 0.) for v in snapshot.vertices),
        "struct_mass": sum(p["struct_mass"].get(v, 0.) for v in snapshot.vertices),
        "living_struct_mass": snapshot.plant.get("total_living_struct_mass"),
        "vertices": len(snapshot.vertices),
        "apices": len(apices),
        "lateral_apices": sum(1 for v in apices if p["root_order"].get(v, 1) > 1),
    }
