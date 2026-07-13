import pandas as pd
import numpy as np
import json
import base64
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
from PIL import Image


def remove_marked_particles(df_det_fitted, manual_edits_csv, id_col="particle_id"):
    """
    Remove particles from df_det_fitted that were marked for removal
    in the manual edits CSV.

    Parameters
    ----------
    df_det_fitted : pandas.DataFrame
        DataFrame returned by refine_all_particles_to_df.

    manual_edits_csv : str or Path or pandas.DataFrame
        CSV file downloaded from the HTML review tool,
        or already-loaded DataFrame.

    id_col : str
        Name of the particle ID column in df_det_fitted.

    Returns
    -------
    df_clean : pandas.DataFrame
        df_det_fitted with removed particles deleted.

    remove_ids : np.ndarray
        Particle IDs that were removed.
    """

    
    # Load manual edits
    if isinstance(manual_edits_csv, pd.DataFrame):
        manual_edits = manual_edits_csv.copy()
    else:
        manual_edits = pd.read_csv(manual_edits_csv)

    if "particle_id" not in manual_edits.columns:
        raise ValueError("manual edits file must contain column 'remove_particle_id'")

    # Get IDs marked for removal
    remove_ids = (
        manual_edits["particle_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    # Remove those IDs
    df_clean = df_det_fitted[
        ~df_det_fitted[id_col].astype(int).isin(remove_ids)
    ].copy()

    print("Before:", len(df_det_fitted))
    print("Removed:", len(remove_ids))
    print("After :", len(df_clean))

    return df_clean


import pandas as pd
import numpy as np


def add_manual_particles(
    df_det_fitted,
    manual_edits_csv,
    id_col="particle_id",
    add_x_col="add_x",
    add_y_col="add_y",
):
    """
    Add manually clicked particles to df_det_fitted.

    Parameters
    ----------
    df_det_fitted : pandas.DataFrame
        Original fitted particle dataframe.

    manual_edits_csv : str or Path or pandas.DataFrame
        CSV file downloaded from the HTML review/correction tool,
        or already-loaded DataFrame.

    id_col : str
        Name of the particle ID column.

    add_x_col, add_y_col : str
        Column names in the manual edits file containing added particle x/y positions.

    Returns
    -------
    df_with_added : pandas.DataFrame
        Original dataframe plus manually added particles.

    added_df : pandas.DataFrame
        DataFrame containing only the added particles.
    """

    # Load manual edits
    if isinstance(manual_edits_csv, pd.DataFrame):
        manual_edits = manual_edits_csv.copy()
    else:
        manual_edits = pd.read_csv(manual_edits_csv)

    # Check columns
    if add_x_col not in manual_edits.columns:
        raise ValueError(f"manual edits file must contain column '{add_x_col}'")

    if add_y_col not in manual_edits.columns:
        raise ValueError(f"manual edits file must contain column '{add_y_col}'")

    # Keep only rows where an added particle exists
    add_rows = manual_edits[
        manual_edits[add_x_col].notna() &
        manual_edits[add_y_col].notna()
    ].copy()

    # If nothing was added
    if len(add_rows) == 0:
        print("Before:", len(df_det_fitted))
        print("Added :", 0)
        print("After :", len(df_det_fitted))
        return df_det_fitted.copy(), pd.DataFrame()

    # Create new dataframe with same columns as df_det_fitted
    added_df = pd.DataFrame(columns=df_det_fitted.columns)

    # Fill x/y positions
    added_df["x"] = add_rows[add_x_col].to_numpy(dtype=float)
    added_df["y"] = add_rows[add_y_col].to_numpy(dtype=float)

    # Give new particle IDs
    if id_col in df_det_fitted.columns:
        max_id = int(df_det_fitted[id_col].max())
        added_df[id_col] = np.arange(
            max_id + 1,
            max_id + 1 + len(added_df)
        )

    # Mark these as manually added, useful later
    added_df["manual_added"] = True

    # For original dataframe, also add this column if it does not exist
    df_original = df_det_fitted.copy()

    if "manual_added" not in df_original.columns:
        df_original["manual_added"] = False

    # Make sure added_df has all same columns as df_original
    for col in df_original.columns:
        if col not in added_df.columns:
            added_df[col] = np.nan

    added_df = added_df[df_original.columns]

    # Combine
    df_with_added = pd.concat(
        [df_original, added_df],
        ignore_index=True
    )

    print("Before:", len(df_det_fitted))
    print("Added :", len(added_df))
    print("After :", len(df_with_added))

    return df_with_added, added_df




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
    Safely parse polygon columns that may be:
    - JSON string
    - Python list
    - NaN
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

def make_remove_particles_html(
    image,
    roi_df,
    output_html,
    output_csv_name="particles_to_remove.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    polygon_col=None,
    point_radius=4,
):
    """
    Make an interactive HTML file for manually marking particles for removal.

    Controls
    --------
    Left click particle : toggle remove / keep
    Mouse wheel         : zoom
    Drag mouse          : pan
    U button            : undo last click
    Download CSV button : saves particles_to_remove.csv

    Output CSV columns
    ------------------
    particle_id, x, y, remove
    """

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    df = roi_df.copy().reset_index(drop=True)

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))

    records = []

    for _, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        rec = {
            "particle_id": row[id_col],
            "x": float(x),
            "y": float(y),
            "polygon": None,
        }

        if polygon_col is not None and polygon_col in df.columns:
            rec["polygon"] = _parse_polygon_value(row[polygon_col])

        records.append(rec)

    image_b64 = _image_to_base64_png(image)

    data_json = json.dumps(records)
    image_src = f"data:image/png;base64,{image_b64}"

    html_template = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Remove particles</title>

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
    gap: 10px;
    align-items: center;
}

button {
    padding: 6px 10px;
    font-size: 14px;
}

#status {
    margin-left: 10px;
}

canvas {
    position: absolute;
    top: 48px;
    left: 0;
    background: black;
    cursor: crosshair;
}
</style>
</head>

<body>

<div id="toolbar">
    <button onclick="downloadCSV()">Download removal CSV</button>
    <button onclick="undoLast()">Undo</button>
    <button onclick="resetView()">Reset view</button>
    <span id="status">Loading...</span>
</div>

<canvas id="canvas"></canvas>

<script>
const particles = __PARTICLE_DATA__;
const outputCsvName = "__OUTPUT_CSV_NAME__";
const imageSrc = "__IMAGE_SRC__";
const pointRadius = __POINT_RADIUS__;

let removed = new Set();
let history = [];

let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

let img = new Image();

let scale = 1.0;
let offsetX = 0;
let offsetY = 0;

let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragOffsetX = 0;
let dragOffsetY = 0;
let movedDuringDrag = false;

img.onload = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 48;

    offsetX = 0;
    offsetY = 0;
    scale = Math.min(canvas.width / img.width, canvas.height / img.height);

    draw();
};

img.src = imageSrc;

window.addEventListener("resize", function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 48;
    draw();
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

    for (const p of particles) {
        const isRemoved = removed.has(String(p.particle_id));

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
            ctx.globalAlpha = isRemoved ? 0.95 : 0.5;
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }

        const s = imageToScreen(p.x, p.y);

        ctx.beginPath();
        ctx.arc(s.x, s.y, pointRadius, 0, 2 * Math.PI);

        if (isRemoved) {
            ctx.fillStyle = "red";
            ctx.strokeStyle = "white";
            ctx.lineWidth = 1.5;
        } else {
            ctx.fillStyle = "lime";
            ctx.strokeStyle = "black";
            ctx.lineWidth = 1.0;
        }

        ctx.fill();
        ctx.stroke();
    }

    document.getElementById("status").innerText =
        "Marked for removal: " + removed.size + " / " + particles.length;
}

function findNearestParticle(screenX, screenY) {
    let best = null;
    let bestDist = Infinity;

    for (const p of particles) {
        const s = imageToScreen(p.x, p.y);
        const dx = s.x - screenX;
        const dy = s.y - screenY;
        const d = Math.sqrt(dx * dx + dy * dy);

        if (d < bestDist) {
            bestDist = d;
            best = p;
        }
    }

    const maxClickDist = Math.max(10, pointRadius * 3);

    if (bestDist <= maxClickDist) {
        return best;
    }

    return null;
}

canvas.addEventListener("mousedown", function(e) {
    isDragging = true;
    movedDuringDrag = false;

    dragStartX = e.clientX;
    dragStartY = e.clientY;

    dragOffsetX = offsetX;
    dragOffsetY = offsetY;
});

canvas.addEventListener("mousemove", function(e) {
    if (!isDragging) return;

    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;

    if (Math.abs(dx) + Math.abs(dy) > 3) {
        movedDuringDrag = true;
    }

    offsetX = dragOffsetX + dx;
    offsetY = dragOffsetY + dy;

    draw();
});

canvas.addEventListener("mouseup", function(e) {
    if (!isDragging) return;

    isDragging = false;

    if (movedDuringDrag) return;

    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    const p = findNearestParticle(sx, sy);

    if (p === null) return;

    const id = String(p.particle_id);

    if (removed.has(id)) {
        removed.delete(id);
        history.push({id: id, action: "keep"});
    } else {
        removed.add(id);
        history.push({id: id, action: "remove"});
    }

    draw();
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
        removed.delete(last.id);
    } else {
        removed.add(last.id);
    }

    draw();
}

function resetView() {
    scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    offsetX = 0;
    offsetY = 0;
    draw();
}

function downloadCSV() {
    let lines = [];
    lines.push("particle_id,x,y,remove");

    for (const p of particles) {
        const id = String(p.particle_id);

        if (removed.has(id)) {
            lines.push(
                p.particle_id + "," +
                p.x + "," +
                p.y + "," +
                "1"
            );
        }
    }

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
        .replace("__PARTICLE_DATA__", data_json)
        .replace("__OUTPUT_CSV_NAME__", output_csv_name)
        .replace("__IMAGE_SRC__", image_src)
        .replace("__POINT_RADIUS__", str(point_radius))
    )

    output_html.write_text(html, encoding="utf-8")

    return str(output_html)

def remove_particles_from_roi_df(
    roi_df_or_csv,
    removal_csv,
    id_col="particle_id",
    x_col="x",
    y_col="y",
    save_path=None,
):
    """
    Remove particles from an ROI dataframe using the CSV exported from
    make_remove_particles_html.

    Parameters
    ----------
    roi_df_or_csv : pandas.DataFrame or str/path
        Original ROI dataframe or path to ROI CSV.

    removal_csv : str/path
        CSV downloaded from the HTML tool.

    id_col : str
        Particle ID column used to match particles.

    x_col, y_col : str
        Coordinate columns. Only used as fallback if id_col is unavailable.

    save_path : str/path or None
        If given, saves final ROI dataframe to this path.

    Returns
    -------
    final_roi_df : pandas.DataFrame
        ROI dataframe with selected particles removed.
    """

    if isinstance(roi_df_or_csv, (str, Path)):
        roi_df = pd.read_csv(roi_df_or_csv)
    else:
        roi_df = roi_df_or_csv.copy()

    removal_df = pd.read_csv(removal_csv)

    removal_df = removal_df[removal_df["remove"].astype(int) == 1].copy()

    if len(removal_df) == 0:
        final_roi_df = roi_df.copy()
    else:
        if id_col in roi_df.columns and id_col in removal_df.columns:
            remove_ids = set(removal_df[id_col].astype(str))
            keep_mask = ~roi_df[id_col].astype(str).isin(remove_ids)
            final_roi_df = roi_df.loc[keep_mask].copy()

        else:
            # fallback: remove by exact x/y coordinate match
            remove_xy = set(
                zip(
                    removal_df[x_col].round(6),
                    removal_df[y_col].round(6),
                )
            )

            roi_xy = list(
                zip(
                    roi_df[x_col].round(6),
                    roi_df[y_col].round(6),
                )
            )

            keep_mask = [xy not in remove_xy for xy in roi_xy]
            final_roi_df = roi_df.loc[keep_mask].copy()

    final_roi_df = final_roi_df.reset_index(drop=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_roi_df.to_csv(save_path, index=False)

    print(f"Original ROI particles: {len(roi_df)}")
    print(f"Removed particles:      {len(roi_df) - len(final_roi_df)}")
    print(f"Final ROI particles:    {len(final_roi_df)}")

    return final_roi_df


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


def make_add_particles_html(
    image,
    roi_df,
    output_html,
    output_csv_name="particles_to_add.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    point_radius=4,
):
    """
    Make an interactive HTML file for manually adding missing particles.

    Controls
    --------
    Left click empty location : add new particle
    Left click added particle : remove that added particle again
    Mouse wheel              : zoom
    Drag mouse               : pan
    Undo button              : undo last add/remove
    Download CSV button      : saves particles_to_add.csv

    Output CSV columns
    ------------------
    add_x, add_y
    """

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    df = roi_df.copy().reset_index(drop=True)

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))

    existing_particles = []

    for _, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        existing_particles.append({
            "particle_id": int(row[id_col]),
            "x": float(x),
            "y": float(y),
        })

    image_b64 = _image_to_base64_png(image)

    existing_json = json.dumps(existing_particles)
    image_src = f"data:image/png;base64,{image_b64}"

    html_template = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Add particles</title>

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
    gap: 10px;
    align-items: center;
}

button {
    padding: 6px 10px;
    font-size: 14px;
}

#status {
    margin-left: 10px;
}

canvas {
    position: absolute;
    top: 48px;
    left: 0;
    background: black;
    cursor: crosshair;
}
</style>
</head>

<body>

<div id="toolbar">
    <button onclick="downloadCSV()">Download added CSV</button>
    <button onclick="undoLast()">Undo</button>
    <button onclick="resetView()">Reset view</button>
    <span id="status">Loading...</span>
</div>

<canvas id="canvas"></canvas>

<script>
const existingParticles = __EXISTING_PARTICLE_DATA__;
const outputCsvName = "__OUTPUT_CSV_NAME__";
const imageSrc = "__IMAGE_SRC__";
const pointRadius = __POINT_RADIUS__;

let addedParticles = [];
let history = [];

let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

let img = new Image();

let scale = 1.0;
let offsetX = 0;
let offsetY = 0;

let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragOffsetX = 0;
let dragOffsetY = 0;
let movedDuringDrag = false;

img.onload = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 48;

    offsetX = 0;
    offsetY = 0;
    scale = Math.min(canvas.width / img.width, canvas.height / img.height);

    draw();
};

img.src = imageSrc;

window.addEventListener("resize", function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 48;
    draw();
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

    // Draw original particles in lime
    for (const p of existingParticles) {
        const s = imageToScreen(p.x, p.y);

        ctx.beginPath();
        ctx.arc(s.x, s.y, pointRadius, 0, 2 * Math.PI);
        ctx.fillStyle = "lime";
        ctx.strokeStyle = "black";
        ctx.lineWidth = 1.0;
        ctx.fill();
        ctx.stroke();
    }

    // Draw added particles in red
    for (let i = 0; i < addedParticles.length; i++) {
        const p = addedParticles[i];
        const s = imageToScreen(p.x, p.y);

        ctx.beginPath();
        ctx.arc(s.x, s.y, pointRadius + 2, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
    }

    document.getElementById("status").innerText =
        "Existing particles: " + existingParticles.length +
        " | Added particles: " + addedParticles.length;
}

function findNearestAddedParticle(screenX, screenY) {
    let bestIndex = -1;
    let bestDist = Infinity;

    for (let i = 0; i < addedParticles.length; i++) {
        const p = addedParticles[i];
        const s = imageToScreen(p.x, p.y);

        const dx = s.x - screenX;
        const dy = s.y - screenY;
        const d = Math.sqrt(dx * dx + dy * dy);

        if (d < bestDist) {
            bestDist = d;
            bestIndex = i;
        }
    }

    const maxClickDist = Math.max(10, pointRadius * 3);

    if (bestDist <= maxClickDist) {
        return bestIndex;
    }

    return -1;
}

canvas.addEventListener("mousedown", function(e) {
    isDragging = true;
    movedDuringDrag = false;

    dragStartX = e.clientX;
    dragStartY = e.clientY;

    dragOffsetX = offsetX;
    dragOffsetY = offsetY;
});

canvas.addEventListener("mousemove", function(e) {
    if (!isDragging) return;

    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;

    if (Math.abs(dx) + Math.abs(dy) > 3) {
        movedDuringDrag = true;
    }

    offsetX = dragOffsetX + dx;
    offsetY = dragOffsetY + dy;

    draw();
});

canvas.addEventListener("mouseup", function(e) {
    if (!isDragging) return;

    isDragging = false;

    if (movedDuringDrag) return;

    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    // If clicking an already-added point, remove it again
    const addedIndex = findNearestAddedParticle(sx, sy);

    if (addedIndex >= 0) {
        const removedPoint = addedParticles.splice(addedIndex, 1)[0];
        history.push({
            action: "remove_added",
            point: removedPoint,
            index: addedIndex
        });
        draw();
        return;
    }

    // Otherwise add a new point at clicked image coordinate
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

    if (last.action === "add") {
        addedParticles.pop();
    }

    if (last.action === "remove_added") {
        addedParticles.splice(last.index, 0, last.point);
    }

    draw();
}

function resetView() {
    scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    offsetX = 0;
    offsetY = 0;
    draw();
}

function downloadCSV() {
    let lines = [];
    lines.push("add_x,add_y");

    for (const p of addedParticles) {
        lines.push(
            p.x.toFixed(6) + "," +
            p.y.toFixed(6)
        );
    }

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
        .replace("__EXISTING_PARTICLE_DATA__", existing_json)
        .replace("__OUTPUT_CSV_NAME__", output_csv_name)
        .replace("__IMAGE_SRC__", image_src)
        .replace("__POINT_RADIUS__", str(point_radius))
    )

    output_html.write_text(html, encoding="utf-8")

    return str(output_html)


def remove_too_close_particles(
    df,
    min_dist_um=0.2,
    um_per_px=0.06,
    x_col="x",
    y_col="y",
    id_col="particle_id",
    prefer_original=True,
):
    """
    Remove duplicate particles that are closer than min_dist_um.

    Coordinates are assumed to be in pixels.
    Distance threshold is given in micrometers.

    If prefer_original=True and the dataframe has a 'manual_added' column,
    then manually added particles are removed first when they overlap
    with original particles.

    Parameters
    ----------
    df : pandas.DataFrame
        Particle dataframe.

    min_dist_um : float
        Minimum allowed distance between particles in micrometers.

    um_per_px : float
        Micrometers per pixel.

    x_col, y_col : str
        Coordinate columns.

    id_col : str
        Particle ID column.

    prefer_original : bool
        If True, keep original particles over manually added particles.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe with too-close duplicates removed.

    removed_df : pandas.DataFrame
        Particles that were removed.

    close_pairs_df : pandas.DataFrame
        All close pairs that were detected.
    """

    df = df.copy().reset_index(drop=True)

    min_dist_px = min_dist_um / um_per_px

    valid = (
        np.isfinite(df[x_col].to_numpy(dtype=float)) &
        np.isfinite(df[y_col].to_numpy(dtype=float))
    )

    valid_indices = np.where(valid)[0]
    points = df.loc[valid, [x_col, y_col]].to_numpy(dtype=float)

    if len(points) < 2:
        print("Not enough valid particles to compare.")
        return df, pd.DataFrame(), pd.DataFrame()

    tree = cKDTree(points)

    # pairs are indices inside the valid-points array
    pairs = list(tree.query_pairs(r=min_dist_px))

    if len(pairs) == 0:
        print("Before:", len(df))
        print("Removed:", 0)
        print("After :", len(df))
        print(f"No pairs closer than {min_dist_um} µm.")
        return df, pd.DataFrame(), pd.DataFrame()

    remove_indices = set()
    close_pair_records = []

    for i_local, j_local in pairs:
        i = valid_indices[i_local]
        j = valid_indices[j_local]

        xi, yi = df.loc[i, [x_col, y_col]]
        xj, yj = df.loc[j, [x_col, y_col]]

        dist_px = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        dist_um = dist_px * um_per_px

        close_pair_records.append({
            "index_1": i,
            "index_2": j,
            "particle_id_1": df.loc[i, id_col] if id_col in df.columns else i,
            "particle_id_2": df.loc[j, id_col] if id_col in df.columns else j,
            "distance_px": dist_px,
            "distance_um": dist_um,
        })

        # Decide which one to remove
        if prefer_original and "manual_added" in df.columns:
            i_manual = bool(df.loc[i, "manual_added"])
            j_manual = bool(df.loc[j, "manual_added"])

            if i_manual and not j_manual:
                remove_indices.add(i)
            elif j_manual and not i_manual:
                remove_indices.add(j)
            else:
                # if both are same type, remove the later one
                remove_indices.add(max(i, j))
        else:
            # default: remove the later one
            remove_indices.add(max(i, j))

    removed_df = df.loc[sorted(remove_indices)].copy()

    df_clean = df.drop(index=sorted(remove_indices)).reset_index(drop=True)

    close_pairs_df = pd.DataFrame(close_pair_records)

    print("Before:", len(df))
    print("Close pairs found:", len(close_pairs_df))
    print("Removed:", len(removed_df))
    print("After :", len(df_clean))
    print(f"Minimum distance used: {min_dist_um} µm = {min_dist_px:.2f} px")

    return df_clean, removed_df, close_pairs_df


def make_edit_particles_html(
    image,
    roi_df,
    output_html,
    output_edits_csv_name="particle_edits.csv",
    output_add_csv_name="particles_to_add.csv",
    output_remove_csv_name="particles_to_remove.csv",
    output_move_csv_name="particles_to_move.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    polygon_col=None,
    point_radius=4,
):
    """
    Interactive HTML editor for particle centers.

    Modes
    -----
    Remove mode:
        click existing particle -> toggle remove/keep

    Add mode:
        click empty location -> add new particle
        click added particle -> remove added particle again

    Move mode:
        drag existing or added particle -> move center

    Navigation
    ----------
    mouse wheel -> zoom
    drag empty region -> pan

    Downloads
    ---------
    - combined edits CSV
    - add-only CSV
    - remove-only CSV
    - move-only CSV
    """

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    df = roi_df.copy().reset_index(drop=True)

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))

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
<title>Edit particles</title>

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
}

canvas {
    position: absolute;
    top: 82px;
    left: 0;
    background: black;
    cursor: crosshair;
}
</style>
</head>

<body>

<div id="toolbar">
    <button id="removeBtn" onclick="setMode('remove')">Remove mode</button>
    <button id="addBtn" onclick="setMode('add')">Add mode</button>
    <button id="moveBtn" onclick="setMode('move')">Move mode</button>

    <button onclick="undoLast()">Undo</button>
    <button onclick="resetView()">Reset view</button>

    <button onclick="downloadCombinedCSV()">Download edits CSV</button>
    <button onclick="downloadAddCSV()">Download add CSV</button>
    <button onclick="downloadRemoveCSV()">Download remove CSV</button>
    <button onclick="downloadMoveCSV()">Download move CSV</button>

    <span id="status">Loading...</span>
</div>

<canvas id="canvas"></canvas>

<script>
const originalParticles = __PARTICLE_DATA__;
const imageSrc = "__IMAGE_SRC__";
const pointRadius = __POINT_RADIUS__;

const combinedCsvName = "__OUTPUT_EDITS_CSV_NAME__";
const addCsvName = "__OUTPUT_ADD_CSV_NAME__";
const removeCsvName = "__OUTPUT_REMOVE_CSV_NAME__";
const moveCsvName = "__OUTPUT_MOVE_CSV_NAME__";

let mode = "remove";

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

let dragStartX = 0;
let dragStartY = 0;
let dragOffsetX = 0;
let dragOffsetY = 0;

let movingParticle = null;
let movingStart = null;

img.onload = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 82;

    scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    offsetX = 0;
    offsetY = 0;

    setMode("remove");
    draw();
};

img.src = imageSrc;

window.addEventListener("resize", function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 82;
    draw();
});

function setMode(newMode) {
    mode = newMode;

    document.getElementById("removeBtn").classList.toggle("active", mode === "remove");
    document.getElementById("addBtn").classList.toggle("active", mode === "add");
    document.getElementById("moveBtn").classList.toggle("active", mode === "move");

    draw();
}

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
        } else if (moved[id] !== undefined) {
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
        if (removed.has(id) && mode !== "remove") continue;

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

    dragStartX = e.clientX;
    dragStartY = e.clientY;

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
                    x: pos.x,
                    y: pos.y,
                    old_x: nearest.particle.x,
                    old_y: nearest.particle.y,
                    id: id
                };
            } else {
                movingStart = {
                    x: nearest.particle.x,
                    y: nearest.particle.y,
                    index: nearest.index
                };
            }
        }
    }
});

canvas.addEventListener("mousemove", function(e) {
    if (!isDragging) return;

    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;

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

    offsetX = dragOffsetX + dx;
    offsetY = dragOffsetY + dy;

    draw();
});

canvas.addEventListener("mouseup", function(e) {
    if (!isDragging) return;
    isDragging = false;

    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    if (mode === "move" && movingParticle !== null) {
        if (movingParticle.type === "original") {
            history.push({
                action: "move_original",
                particle_id: movingStart.id,
                before: {
                    new_x: movingStart.x,
                    new_y: movingStart.y
                },
                after: moved[movingStart.id]
            });
        } else {
            history.push({
                action: "move_added",
                index: movingStart.index,
                before: {
                    x: movingStart.x,
                    y: movingStart.y
                },
                after: {
                    x: addedParticles[movingStart.index].x,
                    y: addedParticles[movingStart.index].y
                }
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
            history.push({action: "unremove", particle_id: id});
        } else {
            removed.add(id);
            history.push({action: "remove", particle_id: id});
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
        if (
            Math.abs(last.before.new_x - originalParticles.find(p => String(p.particle_id) === last.particle_id).x) < 1e-12 &&
            Math.abs(last.before.new_y - originalParticles.find(p => String(p.particle_id) === last.particle_id).y) < 1e-12
        ) {
            delete moved[last.particle_id];
        } else {
            moved[last.particle_id] = {
                particle_id: last.particle_id,
                old_x: originalParticles.find(p => String(p.particle_id) === last.particle_id).x,
                old_y: originalParticles.find(p => String(p.particle_id) === last.particle_id).y,
                new_x: last.before.new_x,
                new_y: last.before.new_y
            };
        }
    }

    if (last.action === "move_added") {
        addedParticles[last.index].x = last.before.x;
        addedParticles[last.index].y = last.before.y;
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

    for (const p of addedParticles) {
        lines.push(
            "add,," +
            p.x.toFixed(6) + "," +
            p.y.toFixed(6) + ",,,1"
        );
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

    return lines;
}

function downloadTextFile(lines, filename) {
    const csv = lines.join("\n");
    const blob = new Blob([csv], {type: "text/csv"});
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}

function downloadCombinedCSV() {
    downloadTextFile(makeCombinedRows(), combinedCsvName);
}

function downloadAddCSV() {
    let lines = [];
    lines.push("add_x,add_y");

    for (const p of addedParticles) {
        lines.push(p.x.toFixed(6) + "," + p.y.toFixed(6));
    }

    downloadTextFile(lines, addCsvName);
}

function downloadRemoveCSV() {
    let lines = [];
    lines.push("particle_id,x,y,remove");

    for (const p of originalParticles) {
        const id = String(p.particle_id);

        if (removed.has(id)) {
            lines.push(
                id + "," +
                p.x.toFixed(6) + "," +
                p.y.toFixed(6) + ",1"
            );
        }
    }

    downloadTextFile(lines, removeCsvName);
}

function downloadMoveCSV() {
    let lines = [];
    lines.push("particle_id,x,y,new_x,new_y");

    for (const id in moved) {
        if (removed.has(id)) continue;

        const m = moved[id];

        lines.push(
            id + "," +
            m.old_x.toFixed(6) + "," +
            m.old_y.toFixed(6) + "," +
            m.new_x.toFixed(6) + "," +
            m.new_y.toFixed(6)
        );
    }

    downloadTextFile(lines, moveCsvName);
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
        .replace("__OUTPUT_EDITS_CSV_NAME__", output_edits_csv_name)
        .replace("__OUTPUT_ADD_CSV_NAME__", output_add_csv_name)
        .replace("__OUTPUT_REMOVE_CSV_NAME__", output_remove_csv_name)
        .replace("__OUTPUT_MOVE_CSV_NAME__", output_move_csv_name)
    )

    output_html.write_text(html, encoding="utf-8")

    return str(output_html)

def apply_particle_edits(
    df,
    edits_csv,
    x_col="x",
    y_col="y",
    id_col="particle_id",
    min_dist_um=0.2,
    um_per_px=0.06,
    prefer_original=True,
    save_path=None,
):
    """
    Apply manual particle edits from make_edit_particles_html.

    Handles:
    - removing particles
    - adding particles
    - moving particles
    - removing duplicate/too-close centers at the end

    Parameters
    ----------
    df : pandas.DataFrame
        Original particle dataframe.

    edits_csv : str or Path or pandas.DataFrame
        Combined edits CSV downloaded from the HTML.

    min_dist_um : float
        Minimum allowed center-to-center distance in micrometers.

    um_per_px : float
        Micrometers per pixel.

    Returns
    -------
    df_clean : pandas.DataFrame
        Edited and duplicate-cleaned dataframe.

    added_df : pandas.DataFrame
        Only added particles.

    removed_df : pandas.DataFrame
        Removed particles from too-close cleanup.

    close_pairs_df : pandas.DataFrame
        Too-close pairs detected after applying edits.
    """

    df_edit = df.copy().reset_index(drop=True)

    if id_col not in df_edit.columns:
        df_edit[id_col] = np.arange(len(df_edit))

    if "manual_added" not in df_edit.columns:
        df_edit["manual_added"] = False

    if isinstance(edits_csv, pd.DataFrame):
        edits = edits_csv.copy()
    else:
        edits = pd.read_csv(edits_csv)

    if len(edits) == 0:
        print("No edits found.")
        return df_edit, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    edits["action"] = edits["action"].astype(str).str.lower()

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

        df_edit = df_edit[
            ~df_edit[id_col].astype(str).isin(remove_ids)
        ].copy()

    # -------------------------
    # Move particles
    # -------------------------
    move_rows = edits[edits["action"] == "move"].copy()

    for _, row in move_rows.iterrows():
        if pd.isna(row["particle_id"]):
            continue

        pid = str(row["particle_id"])

        if pd.isna(row["new_x"]) or pd.isna(row["new_y"]):
            continue

        mask = df_edit[id_col].astype(str) == pid

        if mask.any():
            df_edit.loc[mask, x_col] = float(row["new_x"])
            df_edit.loc[mask, y_col] = float(row["new_y"])

            if "manual_moved" not in df_edit.columns:
                df_edit["manual_moved"] = False

            df_edit.loc[mask, "manual_moved"] = True

    # -------------------------
    # Add particles
    # -------------------------
    add_rows = edits[edits["action"] == "add"].copy()

    add_rows = add_rows[
        add_rows["x"].notna() &
        add_rows["y"].notna()
    ].copy()

    added_df = pd.DataFrame(columns=df_edit.columns)

    if len(add_rows) > 0:
        added_df = pd.DataFrame(columns=df_edit.columns)

        added_df[x_col] = add_rows["x"].to_numpy(dtype=float)
        added_df[y_col] = add_rows["y"].to_numpy(dtype=float)

        max_id = int(pd.to_numeric(df_edit[id_col], errors="coerce").max())

        added_df[id_col] = np.arange(
            max_id + 1,
            max_id + 1 + len(added_df)
        )

        added_df["manual_added"] = True

        if "manual_moved" in df_edit.columns:
            added_df["manual_moved"] = False

        for col in df_edit.columns:
            if col not in added_df.columns:
                added_df[col] = np.nan

        added_df = added_df[df_edit.columns]

        df_edit = pd.concat(
            [df_edit, added_df],
            ignore_index=True
        )

    print("After manual edits:", len(df_edit))
    print("Removed manually:", len(remove_rows))
    print("Moved manually:  ", len(move_rows))
    print("Added manually:  ", len(add_rows))

    # -------------------------
    # Remove too-close duplicates
    # -------------------------
    df_clean, duplicate_removed_df, close_pairs_df = remove_too_close_particles(
        df_edit,
        min_dist_um=min_dist_um,
        um_per_px=um_per_px,
        x_col=x_col,
        y_col=y_col,
        id_col=id_col,
        prefer_original=prefer_original,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(save_path, index=False)

    return df_clean, added_df, duplicate_removed_df, close_pairs_df