""" 
author: Tirtsa den Haan 
06-07-2026

HTML functions for particle clicker
"""

import nd2
from pprint import pprint
import pandas as pd
from pathlib import Path
from matplotlib.path import Path as MplPath
import pandas as pd
import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image


def to_uint8(img, mode="minmax", p_low=1, p_high=99):

    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img.copy()

    img = img.astype(np.float32)

    if mode == "divide256":
        img8 = np.clip(img / 256.0, 0, 255).astype(np.uint8)

    elif mode == "minmax":
        mn, mx = img.min(), img.max()
        if mx <= mn:
            img8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img8 = ((img - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

    elif mode == "percentile":
        lo, hi = np.percentile(img, [p_low, p_high])
        if hi <= lo:
            img8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img8 = ((img - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)

    else:
        raise ValueError("mode must be 'minmax', 'percentile', or 'divide256'")

    return img8

def reading_nd2(file_path, frame_index=-1):
    
    FRAME_INDEX = frame_index   # -1 = last frame, 0 = first frame, 10 = frame 10, etc.

    # Optional: inspect ND2 metadata
    with nd2.ND2File(file_path) as f:
        print("shape:", f.shape)
        print("sizes:", f.sizes)
        print("voxel size:", f.voxel_size())

        print("\n=== text_info ===")
        pprint(f.text_info)

        print("\n=== experiment ===")
        pprint(f.experiment)

        print("\n=== first frame metadata ===")
        pprint(f.frame_metadata(0))


    # Read ND2 as lazy dask array
    framesarr = nd2.imread(file_path, dask=True)

    print("framesarr shape:", framesarr.shape)
    print("framesarr ndim :", framesarr.ndim)

    # Select frame safely
    if framesarr.ndim == 2:
        # ND2 contains only one 2D image
        raw_frame = framesarr

    elif framesarr.ndim == 3:
        # ND2 contains multiple frames: shape probably (T, Y, X)
        raw_frame = framesarr[FRAME_INDEX]

    elif framesarr.ndim == 4:
        # ND2 may have time + channel or z
        # common shape: (T, C, Y, X) or (T, Z, Y, X)
        raw_frame = framesarr[FRAME_INDEX, 0]

    else:
        raise ValueError(f"Unexpected ND2 shape: {framesarr.shape}")

    # Convert from dask array to numpy array
    try:
        img16 = raw_frame.compute()
    except AttributeError:
        img16 = np.asarray(raw_frame)

    img16 = np.squeeze(img16)

    print("selected img16 shape:", img16.shape)

    if img16.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img16.shape}")

    arr8 = to_uint8(img16, mode="percentile", p_low=1, p_high=99)
    frame = np.stack([arr8, arr8, arr8], axis=-1)

    return frame, arr8

def _image_to_base64_png(image):
    """
    Convert grayscale or RGB image array to base64 PNG for embedding in HTML.
    """
    arr = np.asarray(image)

    if arr.dtype != np.uint8:
        arr = arr.astype(float)
        arr = arr - np.nanmin(arr)
        arr = arr / max(np.nanmax(arr), 1e-12)
        arr = (255 * arr).astype(np.uint8)

    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3:
        im = Image.fromarray(arr)
    else:
        raise ValueError("image must be 2D grayscale or 3D RGB/RGBA")

    buffer = BytesIO()
    im.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _parse_polygon_value(value):
    """
    Safely parse polygon columns that may be JSON strings, lists, or NaN.
    """
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    if isinstance(value, str):
        try:
            poly = json.loads(value)
        except Exception:
            return None
    else:
        poly = value

    try:
        poly = np.asarray(poly, dtype=float)
    except Exception:
        return None

    if poly.ndim != 2 or poly.shape[1] != 2:
        return None

    return poly.tolist()


def make_particle_clicker_html(
    image,
    particles,
    output_html,
    output_csv_name="particle_manual_edits.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    polygon_col=None,
    point_radius=4,
):
    """
    Make one HTML clicker for particle correction.

    Modes
    -----
    T or toolbar button : pan mode
    A or toolbar button : add missed particles
    R or toolbar button : remove particles
    M or toolbar button : move particle centre

    Controls
    --------
    Mouse wheel : zoom
    U           : undo
    Reset view  : reset zoom and pan

    Output CSV
    ----------
    One combined CSV with columns:

        action, particle_id, x, y, new_x, new_y, is_added

    Rows:
        remove : remove an existing particle
        add    : add a missed particle
        move   : move the centre of an existing particle

    Notes
    -----
    - Added particles have empty particle_id in the CSV.
    - Moved particles keep their original particle_id.
    - If a moved particle is also removed, only the remove action is exported.
    """

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(particles, (str, Path)):
        df = pd.read_csv(particles)
    else:
        df = particles.copy()

    df = df.reset_index(drop=True)

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))

    required = [id_col, x_col, y_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"particles must contain column '{col}'")

    records = []

    for _, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        rec = {
            "particle_id": str(row[id_col]),
            "x": float(x),
            "y": float(y),
            "polygon": None,
        }

        if polygon_col is not None and polygon_col in df.columns:
            rec["polygon"] = _parse_polygon_value(row[polygon_col])

        records.append(rec)

    image_b64 = _image_to_base64_png(image)
    image_src = f"data:image/png;base64,{image_b64}"

    html_template = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Particle clicker</title>

<style>
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
}

#toolbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: #222;
    padding: 8px;
    z-index: 10;
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}

button {
    padding: 6px 10px;
    font-size: 14px;
}

button.active {
    background: #ffd84d;
    color: black;
    font-weight: bold;
}

#status {
    margin-left: 10px;
    font-size: 14px;
}

#help {
    width: 100%;
    font-size: 13px;
    color: #ddd;
    margin-top: 4px;
}

canvas {
    position: absolute;
    top: 92px;
    left: 0;
    background: black;
}
</style>
</head>

<body>

<div id="toolbar">
    <button id="panBtn" onclick="setMode('pan')">Pan mode [T]</button>
    <button id="addBtn" onclick="setMode('add')">Add missed [A]</button>
    <button id="removeBtn" onclick="setMode('remove')">Remove [R]</button>
    <button id="moveBtn" onclick="setMode('move')">Move centre [M]</button>

    <button onclick="undoLast()">Undo [U]</button>
    <button onclick="resetView()">Reset view</button>
    <button onclick="downloadCombinedCSV()">Download CSV</button>

    <span id="status">Loading...</span>

    <div id="help">
        T = pan, A = add missed particle, R = remove particle, M = move centre, U = undo, mouse wheel = zoom.
    </div>
</div>

<canvas id="canvas"></canvas>

<script>
const originalParticles = __PARTICLE_DATA__;
const imageSrc = "__IMAGE_SRC__";
const pointRadius = __POINT_RADIUS__;
const outputCsvName = "__OUTPUT_CSV_NAME__";

let mode = "pan";

let removed = new Set();
let addedParticles = [];
let moved = {};
let history = [];

let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

let img = new Image();

let scale = 1.0;
let offsetX = 0;
let offsetY = 0;

let isDragging = false;
let movedDuringDrag = false;

let dragStartClientX = 0;
let dragStartClientY = 0;
let dragOffsetX = 0;
let dragOffsetY = 0;

let movingParticle = null;
let movingStart = null;

img.onload = function() {
    resizeCanvas();
    resetView();
    setMode("pan");
    draw();
};

img.src = imageSrc;

window.addEventListener("resize", function() {
    resizeCanvas();
    draw();
});

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 92;
}

function setMode(newMode) {
    mode = newMode;

    document.getElementById("panBtn").classList.toggle("active", mode === "pan");
    document.getElementById("addBtn").classList.toggle("active", mode === "add");
    document.getElementById("removeBtn").classList.toggle("active", mode === "remove");
    document.getElementById("moveBtn").classList.toggle("active", mode === "move");

    if (mode === "pan") {
        canvas.style.cursor = "grab";
    } else if (mode === "move") {
        canvas.style.cursor = "move";
    } else {
        canvas.style.cursor = "crosshair";
    }

    draw();
}

document.addEventListener("keydown", function(e) {
    const key = e.key.toLowerCase();

    if (key === "t") setMode("pan");
    if (key === "a") setMode("add");
    if (key === "r") setMode("remove");
    if (key === "m") setMode("move");
    if (key === "u") undoLast();
});

function imageToScreen(x, y) {
    return {
        x: x * scale + offsetX,
        y: y * scale + offsetY
    };
}

function screenToImage(x, y) {
    return {
        x: (x - offsetX) / scale,
        y: (y - offsetY) / scale
    };
}

function getOriginalParticleById(id) {
    return originalParticles.find(p => String(p.particle_id) === String(id));
}

function getParticlePosition(p) {
    const id = String(p.particle_id);

    if (moved[id] !== undefined) {
        return {
            x: moved[id].new_x,
            y: moved[id].new_y
        };
    }

    return {
        x: p.x,
        y: p.y
    };
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(
        img,
        offsetX,
        offsetY,
        img.width * scale,
        img.height * scale
    );

    for (const p of originalParticles) {
        const id = String(p.particle_id);
        const isRemoved = removed.has(id);
        const isMoved = moved[id] !== undefined;
        const pos = getParticlePosition(p);

        if (p.polygon !== null) {
            ctx.beginPath();

            for (let k = 0; k < p.polygon.length; k++) {
                const q = imageToScreen(p.polygon[k][0], p.polygon[k][1]);

                if (k === 0) {
                    ctx.moveTo(q.x, q.y);
                } else {
                    ctx.lineTo(q.x, q.y);
                }
            }

            ctx.closePath();

            ctx.lineWidth = isRemoved ? 2.5 : 1.0;
            ctx.strokeStyle = isRemoved ? "red" : "cyan";
            ctx.globalAlpha = isRemoved ? 0.95 : 0.45;
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }

        const s = imageToScreen(pos.x, pos.y);

        ctx.beginPath();
        ctx.arc(s.x, s.y, pointRadius, 0, 2 * Math.PI);

        if (isRemoved) {
            ctx.fillStyle = "red";
            ctx.strokeStyle = "white";
            ctx.lineWidth = 1.5;
        } else if (isMoved) {
            ctx.fillStyle = "orange";
            ctx.strokeStyle = "black";
            ctx.lineWidth = 1.2;
        } else {
            ctx.fillStyle = "lime";
            ctx.strokeStyle = "black";
            ctx.lineWidth = 1.0;
        }

        ctx.fill();
        ctx.stroke();
    }

    for (let i = 0; i < addedParticles.length; i++) {
        const p = addedParticles[i];
        const s = imageToScreen(p.x, p.y);

        ctx.beginPath();
        ctx.arc(s.x, s.y, pointRadius + 2, 0, 2 * Math.PI);
        ctx.fillStyle = "magenta";
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
    }

    document.getElementById("status").innerText =
        "Mode: " + mode +
        " | original: " + originalParticles.length +
        " | removed: " + removed.size +
        " | added: " + addedParticles.length +
        " | moved: " + Object.keys(moved).length;
}

function findNearestParticle(screenX, screenY, includeAdded=true) {
    let best = null;
    let bestDist = Infinity;

    for (const p of originalParticles) {
        const id = String(p.particle_id);

        if (removed.has(id) && mode !== "remove") {
            continue;
        }

        const pos = getParticlePosition(p);
        const s = imageToScreen(pos.x, pos.y);

        const dx = s.x - screenX;
        const dy = s.y - screenY;
        const d = Math.sqrt(dx * dx + dy * dy);

        if (d < bestDist) {
            bestDist = d;
            best = {
                type: "original",
                particle: p,
                index: null
            };
        }
    }

    if (includeAdded) {
        for (let i = 0; i < addedParticles.length; i++) {
            const p = addedParticles[i];
            const s = imageToScreen(p.x, p.y);

            const dx = s.x - screenX;
            const dy = s.y - screenY;
            const d = Math.sqrt(dx * dx + dy * dy);

            if (d < bestDist) {
                bestDist = d;
                best = {
                    type: "added",
                    particle: p,
                    index: i
                };
            }
        }
    }

    const maxClickDist = Math.max(10, pointRadius * 3);

    if (bestDist <= maxClickDist) {
        return best;
    }

    return null;
}

canvas.addEventListener("mousedown", function(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    isDragging = true;
    movedDuringDrag = false;

    dragStartClientX = e.clientX;
    dragStartClientY = e.clientY;

    dragOffsetX = offsetX;
    dragOffsetY = offsetY;

    movingParticle = null;
    movingStart = null;

    if (mode === "move") {
        const nearest = findNearestParticle(sx, sy, true);

        if (nearest !== null) {
            movingParticle = nearest;

            if (nearest.type === "original") {
                const id = String(nearest.particle.particle_id);
                const pos = getParticlePosition(nearest.particle);

                movingStart = {
                    type: "original",
                    particle_id: id,
                    current_x: pos.x,
                    current_y: pos.y,
                    old_x: nearest.particle.x,
                    old_y: nearest.particle.y
                };
            } else {
                movingStart = {
                    type: "added",
                    index: nearest.index,
                    x: nearest.particle.x,
                    y: nearest.particle.y
                };
            }
        }
    }
});

canvas.addEventListener("mousemove", function(e) {
    if (!isDragging) return;

    const dx = e.clientX - dragStartClientX;
    const dy = e.clientY - dragStartClientY;

    if (Math.abs(dx) + Math.abs(dy) > 3) {
        movedDuringDrag = true;
    }

    if (mode === "move" && movingParticle !== null) {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const p = screenToImage(sx, sy);

        if (movingParticle.type === "original") {
            const id = String(movingParticle.particle.particle_id);

            moved[id] = {
                particle_id: id,
                old_x: movingParticle.particle.x,
                old_y: movingParticle.particle.y,
                new_x: p.x,
                new_y: p.y
            };
        } else {
            addedParticles[movingParticle.index].x = p.x;
            addedParticles[movingParticle.index].y = p.y;
        }

        draw();
        return;
    }

    if (mode === "pan") {
        offsetX = dragOffsetX + dx;
        offsetY = dragOffsetY + dy;
        draw();
        return;
    }
});

canvas.addEventListener("mouseup", function(e) {
    if (!isDragging) return;

    isDragging = false;

    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    if (mode === "move" && movingParticle !== null) {
        if (movingParticle.type === "original") {
            const id = movingStart.particle_id;

            history.push({
                action: "move_original",
                particle_id: id,
                before_x: movingStart.current_x,
                before_y: movingStart.current_y,
                after_x: moved[id].new_x,
                after_y: moved[id].new_y
            });
        } else {
            history.push({
                action: "move_added",
                index: movingStart.index,
                before_x: movingStart.x,
                before_y: movingStart.y,
                after_x: addedParticles[movingStart.index].x,
                after_y: addedParticles[movingStart.index].y
            });
        }

        movingParticle = null;
        movingStart = null;
        draw();
        return;
    }

    if (movedDuringDrag) return;

    const nearest = findNearestParticle(sx, sy, true);

    if (mode === "remove") {
        if (nearest === null) return;

        if (nearest.type === "added") {
            const removedAdded = addedParticles.splice(nearest.index, 1)[0];

            history.push({
                action: "remove_added",
                point: removedAdded,
                index: nearest.index
            });

            draw();
            return;
        }

        const id = String(nearest.particle.particle_id);

        if (removed.has(id)) {
            removed.delete(id);

            history.push({
                action: "unremove",
                particle_id: id
            });
        } else {
            removed.add(id);

            history.push({
                action: "remove",
                particle_id: id
            });
        }

        draw();
        return;
    }

    if (mode === "add") {
        if (nearest !== null && nearest.type === "added") {
            const removedAdded = addedParticles.splice(nearest.index, 1)[0];

            history.push({
                action: "remove_added",
                point: removedAdded,
                index: nearest.index
            });

            draw();
            return;
        }

        const p = screenToImage(sx, sy);

        if (p.x < 0 || p.x >= img.width || p.y < 0 || p.y >= img.height) {
            return;
        }

        const newPoint = {
            x: p.x,
            y: p.y
        };

        addedParticles.push(newPoint);

        history.push({
            action: "add",
            point: newPoint
        });

        draw();
        return;
    }
});

canvas.addEventListener("wheel", function(e) {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const before = screenToImage(mouseX, mouseY);

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    scale *= zoomFactor;

    const after = imageToScreen(before.x, before.y);

    offsetX += mouseX - after.x;
    offsetY += mouseY - after.y;

    draw();
});

function undoLast() {
    if (history.length === 0) return;

    const last = history.pop();

    if (last.action === "remove") {
        removed.delete(last.particle_id);
    }

    if (last.action === "unremove") {
        removed.add(last.particle_id);
    }

    if (last.action === "add") {
        addedParticles.pop();
    }

    if (last.action === "remove_added") {
        addedParticles.splice(last.index, 0, last.point);
    }

    if (last.action === "move_original") {
        const original = getOriginalParticleById(last.particle_id);

        if (
            Math.abs(last.before_x - original.x) < 1e-12 &&
            Math.abs(last.before_y - original.y) < 1e-12
        ) {
            delete moved[last.particle_id];
        } else {
            moved[last.particle_id] = {
                particle_id: last.particle_id,
                old_x: original.x,
                old_y: original.y,
                new_x: last.before_x,
                new_y: last.before_y
            };
        }
    }

    if (last.action === "move_added") {
        addedParticles[last.index].x = last.before_x;
        addedParticles[last.index].y = last.before_y;
    }

    draw();
}

function resetView() {
    scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    offsetX = 0;
    offsetY = 0;
    draw();
}

function makeCombinedRows() {
    let lines = [];

    lines.push("action,particle_id,x,y,new_x,new_y,is_added");

    for (const p of originalParticles) {
        const id = String(p.particle_id);

        if (removed.has(id)) {
            lines.push(
                "remove," + id + "," +
                p.x.toFixed(6) + "," +
                p.y.toFixed(6) + ",,,0"
            );
        }
    }

    for (const id in moved) {
        if (removed.has(id)) continue;

        const m = moved[id];

        lines.push(
            "move," + id + "," +
            m.old_x.toFixed(6) + "," +
            m.old_y.toFixed(6) + "," +
            m.new_x.toFixed(6) + "," +
            m.new_y.toFixed(6) + ",0"
        );
    }

    for (const p of addedParticles) {
        lines.push(
            "add,," +
            p.x.toFixed(6) + "," +
            p.y.toFixed(6) + ",,,1"
        );
    }

    return lines;
}

function downloadCombinedCSV() {
    const lines = makeCombinedRows();
    const csv = lines.join("\n");

    const blob = new Blob([csv], {type: "text/csv"});
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = outputCsvName;

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}
</script>

</body>
</html>
"""

    html = (
        html_template
        .replace("__PARTICLE_DATA__", json.dumps(records))
        .replace("__IMAGE_SRC__", image_src)
        .replace("__POINT_RADIUS__", str(point_radius))
        .replace("__OUTPUT_CSV_NAME__", output_csv_name)
    )

    output_html.write_text(html, encoding="utf-8")

    return str(output_html)



def apply_particle_clicker_edits(
    particles,
    edits_csv,
    x_col="x",
    y_col="y",
    id_col="particle_id",
    save_path=None,
):
    """
    Apply the combined CSV downloaded from make_particle_clicker_html.

    Handles:
    - remove
    - move
    - add

    Input dataframe/file must contain:
        particle_id, x, y

    Output:
        edited dataframe
    """

    if isinstance(particles, (str, Path)):
        df = pd.read_csv(particles)
    else:
        df = particles.copy()

    df = df.reset_index(drop=True)

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))

    if isinstance(edits_csv, (str, Path)):
        edits = pd.read_csv(edits_csv)
    else:
        edits = edits_csv.copy()

    if len(edits) == 0:
        print("No edits found.")
        return df

    edits["action"] = edits["action"].astype(str).str.lower()

    if "manual_added" not in df.columns:
        df["manual_added"] = False

    if "manual_moved" not in df.columns:
        df["manual_moved"] = False

    # -------------------------
    # Remove particles
    # -------------------------
    remove_rows = edits[edits["action"] == "remove"].copy()

    if len(remove_rows) > 0:
        remove_ids = set(
            remove_rows["particle_id"]
            .dropna()
            .astype(str)
        )

        df = df[
            ~df[id_col].astype(str).isin(remove_ids)
        ].copy()

    # -------------------------
    # Move particles
    # -------------------------
    move_rows = edits[edits["action"] == "move"].copy()

    for _, row in move_rows.iterrows():
        if pd.isna(row["particle_id"]):
            continue

        if pd.isna(row["new_x"]) or pd.isna(row["new_y"]):
            continue

        pid = str(row["particle_id"])
        mask = df[id_col].astype(str) == pid

        if mask.any():
            df.loc[mask, x_col] = float(row["new_x"])
            df.loc[mask, y_col] = float(row["new_y"])
            df.loc[mask, "manual_moved"] = True

    # -------------------------
    # Add particles
    # -------------------------
    add_rows = edits[edits["action"] == "add"].copy()

    add_rows = add_rows[
        add_rows["x"].notna() &
        add_rows["y"].notna()
    ].copy()

    if len(add_rows) > 0:
        added_df = pd.DataFrame(columns=df.columns)

        added_df[x_col] = add_rows["x"].to_numpy(dtype=float)
        added_df[y_col] = add_rows["y"].to_numpy(dtype=float)

        current_ids = pd.to_numeric(df[id_col], errors="coerce")

        if current_ids.notna().any():
            max_id = int(current_ids.max())
        else:
            max_id = len(df) - 1

        added_df[id_col] = np.arange(
            max_id + 1,
            max_id + 1 + len(added_df)
        )

        added_df["manual_added"] = True
        added_df["manual_moved"] = False

        for col in df.columns:
            if col not in added_df.columns:
                added_df[col] = np.nan

        added_df = added_df[df.columns]

        df = pd.concat([df, added_df], ignore_index=True)

    df = df.reset_index(drop=True)

    print("Manual removes:", len(remove_rows))
    print("Manual moves:  ", len(move_rows))
    print("Manual adds:   ", len(add_rows))
    print("Final particles:", len(df))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df