
"""
Aerofoil Profile Generator for Blender
=======================================
Generates a boomerang-style aerofoil cross-section and saves a .blend file.

Parameters:
    D          - maximum thickness/depth of the aerofoil (metres)
    W          - chord width (metres)
    slope      - leading edge bevel angle in degrees (measured from horizontal)
    arm_length   - length of each straight arm (mm)
    arm_angle    - interior angle between the two arms (degrees); 180=straight, 90=right-angle
    iterations   - number of CoM convergence iterations

Profile shape:
    - Flat bottom face
    - Straight leading edge bevel (from bottom-front up to top at angle 'slope')
    - Smooth upper surface (cosine profile) from bevel top back to trailing edge
    - Trailing edge tapers to a point / thin edge

Run from Blender's script editor, or from the command line:
    blender --python aerofoil_profile.py
"""

import bpy
import bmesh
import math
import os

# =============================================================================
# PARAMETERS  —  edit these (all dimensions in mm)
# =============================================================================
D            = 12    # max depth / thickness (mm)
W            = 45    # chord width           (mm)
slope        = 40    # leading edge bevel angle (degrees from horizontal)
tail_T       = 2     # trailing edge thickness (mm) — flat blunt tail
crest_W      = 15     # width of flat crest at maximum thickness (mm) needs to make sense with W
arm_length   = 300   # length of each straight arm (mm)
arm_angle      = 120   # interior angle between the two arms (degrees)
iterations     = 5     # CoM convergence iterations
W_compensation = 0.0002   # chord width scale rate (1/mm); widens chord toward arm tips (away from grip)

def _default_output_path():
    """Return a writable output path regardless of how the script is invoked."""
    import tempfile

    # 1. If the current .blend file has been saved, put it alongside it.
    blend_path = bpy.data.filepath
    if blend_path:
        return os.path.join(os.path.dirname(blend_path), "aerofoil_profile.blend")

    # 2. If __file__ points to a real file on disk (run via --python), use that dir.
    try:
        candidate = os.path.abspath(__file__)
        candidate_dir = os.path.dirname(candidate)
        # Make sure it's a real file path, not a Blender text-block name like "\Text"
        if os.path.isfile(candidate):
            return os.path.join(candidate_dir, "aerofoil_profile.blend")
    except (NameError, OSError):
        pass

    # 3. Final fallback: user's temp directory (always writable).
    return os.path.join(tempfile.gettempdir(), "aerofoil_profile.blend")

OUTPUT_BLEND = _default_output_path()
# =============================================================================


def _resample_equal_arclength(pts, n):
    """
    Resample a closed polygon to n equally arc-length-spaced points.
    Duplicate endpoint (first == last) is removed before resampling.
    """
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    m = len(pts)
    # Cumulative arc-length table; final entry closes back to pts[0]
    cum = [0.0]
    for i in range(m):
        p0 = pts[i]
        p1 = pts[(i + 1) % m]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        cum.append(cum[-1] + math.sqrt(dx*dx + dy*dy))
    total = cum[-1]
    result = []
    seg = 0
    for k in range(n):
        target = k * total / n
        # Advance segment pointer (never rewinds — O(m+n) total)
        while seg < m - 1 and cum[seg + 1] < target:
            seg += 1
        seg_len = cum[seg + 1] - cum[seg]
        if seg_len < 1e-12:
            result.append(pts[seg])
        else:
            t  = (target - cum[seg]) / seg_len
            p0 = pts[seg]
            p1 = pts[(seg + 1) % m]
            result.append((p0[0] + t*(p1[0]-p0[0]),
                           p0[1] + t*(p1[1]-p0[1])))
    return result


def generate_aerofoil_profile(D, W, slope_deg, tail_T=2, crest_W=5,
                              n_bevel=4, n_upper=8, n_bottom=8):
    """
    Return a list of (x, y) 2-D points forming a closed aerofoil cross-section.

    Coordinate convention (looking at the cross-section end-on):
        x — along chord, 0 = leading edge nose, W = trailing edge
        y — thickness,   0 = flat bottom face, D = maximum thickness

    The closed loop goes (counter-clockwise when viewed from +Z):
        1. Leading edge bevel  : (0, 0)         → (le_x, D)           straight line
        2. Flat crest          : (le_x, D)       → (le_x+crest_W, D)  horizontal
        3. Upper surface       : (le_x+crest_W, D) → (W, tail_T)      cosine curve
        4. Trailing edge face  : (W, tail_T)     → (W, 0)             vertical
        5. Flat bottom         : (W, 0)          → (0, 0)             straight line
    """
    if slope_deg <= 0 or slope_deg >= 90:
        raise ValueError("slope must be between 0 and 90 degrees (exclusive)")
    if tail_T < 0 or tail_T >= D:
        raise ValueError("tail_T must be between 0 and D")
    if crest_W < 0:
        raise ValueError("crest_W must be >= 0")

    slope_rad = math.radians(slope_deg)
    le_x = D / math.tan(slope_rad)   # x where bevel meets the crest
    crest_end_x = le_x + crest_W     # x where crest ends and descent begins

    if crest_end_x >= W:
        raise ValueError(
            f"Bevel + crest extends beyond chord: {crest_end_x:.4f} >= W={W:.4f}. "
            "Increase W, reduce slope, or reduce crest_W."
        )

    profile = []

    # --- 1. Leading edge bevel: (0, 0) → (le_x, D) ---
    for i in range(n_bevel + 1):
        t = i / n_bevel
        profile.append((le_x * t, D * t))

    # --- 2. Flat crest: (le_x, D) → (crest_end_x, D) — n_upper edges, matching upper surface ---
    if crest_W > 0:
        for i in range(1, n_upper + 1):
            t = i / n_upper
            profile.append((le_x + t * crest_W, D))

    # --- 3. Upper surface descent: (crest_end_x, D) → (W, tail_T) via cosine ---
    # y = tail_T + (D - tail_T)/2 * (1 + cos(π·t)),  t ∈ [0, 1]
    # At t=0: y=D     (leaves crest with horizontal tangent → smooth join)
    # At t=1: y=tail_T (arrives at trailing edge thickness)
    for i in range(1, n_upper + 1):
        t = i / n_upper
        x = crest_end_x + t * (W - crest_end_x)
        y = tail_T + 0.5 * (D - tail_T) * (1.0 + math.cos(math.pi * t))
        profile.append((x, y))

    # --- 4. Trailing edge face: (W, tail_T) → (W, 0) — same node count as bevel ---
    if tail_T > 0:
        for i in range(1, n_bevel + 1):
            t = i / n_bevel
            profile.append((W, tail_T * (1.0 - t)))

    # --- 5. Flat bottom: (W, 0) → (0, 0) ---
    for i in range(1, n_bottom + 1):
        t = i / n_bottom
        profile.append((W * (1.0 - t), 0.0))

    # Redistribute all nodes to equal arc-length spacing around the perimeter.
    # The raw profile has a duplicate endpoint (0,0); resample to len-1 unique pts.
    n_resample = len(profile) - 1
    return _resample_equal_arclength(profile, n_resample)


def build_boomerang_path(arm_length, arm_angle, arm_width, n_arm=15, n_elbow=8):
    """
    Generate frames for a full boomerang-shaped centreline path:
        1. Arm 1  — straight, length=arm_length, along +Y
        2. Elbow  — circular arc of radius=arm_width, turning by (180-arm_angle)°
        3. Arm 2  — straight, length=arm_length, in the post-elbow direction

    arm_angle   : interior angle between the two arms (180° = straight stick)
    arm_width   : chord width of the cross-section; used as the elbow radius so
                  the two arms are naturally separated at the joint with no overlap

    Returns list of ((px,py,pz), (cdx,cdy,cdz), section) frames.
    """
    deflection   = math.radians(180.0 - arm_angle)
    elbow_radius = arm_width          # 1 chord-width radius keeps the arms clear
    frames       = []

    # --- Arm 1: straight along +Y ---
    for i in range(n_arm + 1):
        t = i / n_arm
        frames.append(((0.0, arm_length * t, 0.0), (1.0, 0.0, 0.0), 'arm1'))

    # --- Elbow: circular arc ---
    # Centre of curvature is at (elbow_radius, arm_length, 0) — to the right of
    # the arm1 tip.  At alpha=0 the path is at (0, arm_length), tangent = +Y.
    for i in range(n_elbow + 1):
        alpha     = deflection * i / n_elbow
        px        = elbow_radius * (1.0 - math.cos(alpha))
        py        = arm_length   + elbow_radius * math.sin(alpha)
        chord_dir = (math.cos(alpha), -math.sin(alpha), 0.0)   # 90° CW from tangent
        frames.append(((px, py, 0.0), chord_dir, 'elbow'))

    # --- Arm 2: straight from end of elbow ---
    ex      = elbow_radius * (1.0 - math.cos(deflection))
    ey      = arm_length   + elbow_radius * math.sin(deflection)
    arm2_tx  = math.sin(deflection)
    arm2_ty  = math.cos(deflection)
    arm2_cdx = math.cos(deflection)
    arm2_cdy = -math.sin(deflection)

    for i in range(1, n_arm + 1):
        t = i / n_arm
        frames.append((
            (ex + t * arm_length * arm2_tx,
             ey + t * arm_length * arm2_ty,
             0.0),
            (arm2_cdx, arm2_cdy, 0.0),
            'arm2',
        ))

    return frames


def _frame_toward_com(pos, tangent, com):
    """
    Compute (x_dir, y_dir) for a profile frame such that:
      - y_dir is always global +Z  →  flat bottom stays horizontal
      - x_dir (chord direction) points toward the CoM in the XY plane,
        perpendicular to the path tangent.

    Because the CoM is near the elbow, this naturally produces opposite chord
    orientations on arm1 and arm2 (both bevels face inward toward the elbow)
    with a smooth, collapse-free transition through the elbow itself.
    """
    y_dir = (0.0, 0.0, 1.0)

    # Vector from station to CoM, projected onto XY
    v = (com[0] - pos[0], com[1] - pos[1], 0.0)

    # Remove tangent component so x_dir is perpendicular to the arm
    dot = v[0]*tangent[0] + v[1]*tangent[1]
    vp  = (v[0] - dot*tangent[0], v[1] - dot*tangent[1], 0.0)

    mag = math.sqrt(vp[0]**2 + vp[1]**2)
    if mag < 1e-9:
        # Station is on the CoM — fall back to 90° CCW from tangent
        x_dir = (-tangent[1], tangent[0], 0.0)
    else:
        x_dir = (vp[0]/mag, vp[1]/mag, 0.0)

    return x_dir, y_dir


def extrude_profile_along_frames(profile_2d, frames, name="Boomerang", n_cap=8, com=None, W_compensation=0.0):
    """
    Sweep a 2-D aerofoil cross-section along pre-computed path frames to build a 3-D solid
    with hemispherical rounded caps at each arm tip.

    profile_2d : closed list of (x, y) profile points
    frames     : list of ((px,py,pz), (cdx,cdy,cdz)) from build_boomerang_path()
    n_cap      : number of dome rings per end cap
    com        : (cx, cy, cz) centre of mass of previous iteration.
                 Each cross-section is rotated so its thickness direction points toward com.
                 If None, global +Z is used (first iteration).

    Returns the created Blender object.
    """
    if not profile_2d or not frames:
        raise ValueError("profile_2d and frames must be non-empty")

    n_f      = len(frames)
    n_pts    = len(profile_2d)
    W_chord  = max(p[0] for p in profile_2d)
    cx_c     = sum(p[0] for p in profile_2d) / n_pts
    cy_c     = sum(p[1] for p in profile_2d) / n_pts
    dome_r   = W_chord * 0.5
    tip_off  = cx_c - W_chord * 0.5

    # Arm 2 uses the profile in reverse: leading edge ends up on the inside of the angle.
    # Rotate so that the trailing-edge bottom corner (W_chord, 0) lands at index 0.
    # This aligns arm2's outside-bottom vertex (j=0) with arm1's outside-bottom vertex (j=0)
    # through the blend, giving a twist-free transition regardless of n_bevel/n_bottom.
    rev_unrot   = list(reversed(profile_2d))
    _rev_offset = min(range(len(rev_unrot)),
                      key=lambda k: (rev_unrot[k][0] - W_chord)**2 + rev_unrot[k][1]**2)
    rev_profile = rev_unrot[_rev_offset:] + rev_unrot[:_rev_offset]

    # --- Pre-compute tangent vectors along the path (central differences) ---
    tangents = []
    for i in range(n_f):
        ia = max(0, i - 1)
        ib = min(n_f - 1, i + 1)
        pa, pb = frames[ia][0], frames[ib][0]
        raw = (pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2])
        mag = math.sqrt(raw[0]**2 + raw[1]**2 + raw[2]**2) or 1.0
        tangents.append((raw[0]/mag, raw[1]/mag, raw[2]/mag))

    # --- Elbow frame range for smooth arm1 → arm2 cross-section blend ---
    elbow_indices = [i for i, f in enumerate(frames) if f[2] == 'elbow']
    elbow_start   = elbow_indices[0] if elbow_indices else -1
    elbow_count   = len(elbow_indices)

    # Blend zone: elbow frames plus n_blend_ext frames into each straight arm.
    n_blend_ext   = elbow_count          # extends by one full elbow-length on each side
    blend_zone_lo = max(0,       elbow_start - n_blend_ext)
    blend_zone_hi = min(n_f - 1, elbow_start + elbow_count - 1 + n_blend_ext)
    blend_span    = max(1, blend_zone_hi - blend_zone_lo)

    # Reference point for W_compensation: elbow centroid (grip area).
    # Chord widens with distance from grip → tips become wider.
    if elbow_indices:
        _ref_x = sum(frames[i][0][0] for i in elbow_indices) / elbow_count
        _ref_y = sum(frames[i][0][1] for i in elbow_indices) / elbow_count
    else:
        _ref_x, _ref_y = 0.0, 0.0

    def chord_scale(frame_idx):
        """Chord width multiplier: 1 + W_compensation * distance_from_elbow_centre."""
        p = frames[frame_idx][0]
        dx, dy = p[0] - _ref_x, p[1] - _ref_y
        return 1.0 + W_compensation * math.sqrt(dx*dx + dy*dy)

    def _blend_factor(frame_idx):
        """Cosine-eased blend: 0 = pure arm1 profile, 1 = pure arm2 profile.
        Transitions across the elbow plus n_blend_ext frames on each side."""
        if frame_idx <= blend_zone_lo:
            return 0.0
        if frame_idx >= blend_zone_hi:
            return 1.0
        t = (frame_idx - blend_zone_lo) / blend_span
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def _tip_factor(frame_idx):
        """1.0 for the first 65% of each arm (from tip), cosine blend to 0.0
        over the last 35% approaching the elbow. Elbow frames always 0.0."""
        section = frames[frame_idx][2]
        if section == 'elbow':
            return 0.0
        elif section == 'arm1':
            # t=0 at tip (frame 0), t=1 at elbow junction
            t = frame_idx / max(1, elbow_start - 1)
        else:  # arm2
            arm2_start_idx = elbow_start + elbow_count
            # t=0 at elbow junction, t=1 at tip
            t = 1.0 - (frame_idx - arm2_start_idx) / max(1, n_f - 1 - arm2_start_idx)
        # t is now 0=tip, 1=elbow for both arms
        if t <= 0.65:
            return 1.0
        else:
            blend_t = (t - 0.65) / 0.35   # 0→1 across the last 35%
            return 0.5 * (1.0 + math.cos(math.pi * blend_t))

    # Outward tangents at each arm tip
    start_tan = (-tangents[0][0], -tangents[0][1], -tangents[0][2])
    end_tan   =  tangents[-1]

    def arm1_dirs(frame_idx):
        """At elbow (tf=0): x_dir = toward-CoM.
        At tip (tf=1): x_dir = 90° CCW of toward-CoM (arm1) or 90° CW (arm2),
        so both arms get the same chirality after the profile flip in make_body_ring."""
        pos, chord_dir_raw, section = frames[frame_idx]
        if com is not None:
            vx = com[0] - pos[0]
            vy = com[1] - pos[1]
            mag = math.sqrt(vx*vx + vy*vy)
            if mag < 1e-9:
                rx, ry = tangents[frame_idx][0], tangents[frame_idx][1]
            else:
                rx, ry = vx/mag, vy/mag
            tf = _tip_factor(frame_idx)
            # arm2 uses CW rotation so that after the profile-flip in make_body_ring
            # both arms produce the same aerodynamic chirality at the tips.
            if section == 'arm2':
                px, py = -ry, rx   # 90° CW of toward-CoM
            else:
                px, py = ry, -rx   # 90° CCW of toward-CoM
            bx = (1.0 - tf) * rx + tf * px
            by = (1.0 - tf) * ry + tf * py
            bmag = math.sqrt(bx*bx + by*by) or 1.0
            x_dir = (bx/bmag, by/bmag, 0.0)
            return x_dir, (0.0, 0.0, 1.0)
        cdx, cdy, _ = chord_dir_raw
        return (cdx, cdy, 0.0), (0.0, 0.0, 1.0)

    def arm2_dirs(frame_idx):
        """x_dir away from CoM — arm2: LE on the inside of the boomerang angle."""
        x_dir, y_dir = arm1_dirs(frame_idx)
        return (-x_dir[0], -x_dir[1], -x_dir[2]), y_dir

    def get_frame_dirs(frame_idx):
        """Return (x_dir, y_dir) based on the frame's section tag."""
        return arm2_dirs(frame_idx) if frames[frame_idx][2] == 'arm2' else arm1_dirs(frame_idx)

    def get_prof(frame_idx):
        """Return the appropriate profile list for this frame's section."""
        return rev_profile if frames[frame_idx][2] == 'arm2' else profile_2d

    bm = bmesh.new()
    all_rings = []

    def make_cap_ring(frame_idx, out_tan, t_param):
        """One dome ring scaled and advanced along out_tan."""
        pos          = frames[frame_idx][0]
        x_dir, y_dir = get_frame_dirs(frame_idx)
        prof         = get_prof(frame_idx)
        sc_chord     = chord_scale(frame_idx)
        s = math.cos(t_param * math.pi * 0.5)
        d = math.sin(t_param * math.pi * 0.5) * dome_r * sc_chord
        ring = []
        for (cx, cy) in prof:
            sx = cx_c + (cx - cx_c) * s
            sy = cy_c + (cy - cy_c) * s
            offset = (sx - W_chord * 0.5) * sc_chord
            ring.append(bm.verts.new((
                pos[0] + offset*x_dir[0] + sy*y_dir[0] + d*out_tan[0],
                pos[1] + offset*x_dir[1] + sy*y_dir[1] + d*out_tan[1],
                pos[2] + offset*x_dir[2] + sy*y_dir[2] + d*out_tan[2],
            )))
        return ring

    def make_body_ring(frame_idx):
        """Build one cross-section ring.
        A cosine-eased blend transitions arm1→arm2 profile across the elbow
        plus n_blend_ext frames into each straight arm.
        b=0: pure arm1 (LE outside)  |  b=1: pure arm2 (LE inside).
        The same formula covers all three sections with no special cases.
        """
        pos           = frames[frame_idx][0]
        b             = _blend_factor(frame_idx)
        arm1_x, y_dir = arm1_dirs(frame_idx)
        sc_chord      = chord_scale(frame_idx)
        ring = []
        for j in range(n_pts):
            p1cx, p1cy = profile_2d[j]
            p2cx, p2cy = rev_profile[j]
            eff_off = sc_chord * ((1.0-b)*(p1cx - W_chord*0.5) + b*(W_chord*0.5 - p2cx))
            eff_cy  = (1.0-b)*p1cy + b*p2cy
            ring.append(bm.verts.new((
                pos[0] + eff_off*arm1_x[0] + eff_cy*y_dir[0],
                pos[1] + eff_off*arm1_x[1] + eff_cy*y_dir[1],
                pos[2] + eff_off*arm1_x[2] + eff_cy*y_dir[2],
            )))
        return ring

    # --- Start cap: near-tip rings first ---
    for i in range(n_cap - 1, 0, -1):
        all_rings.append(make_cap_ring(0, start_tan, i / n_cap))

    # Start tip vertex
    pos0 = frames[0][0]
    x0, y0 = get_frame_dirs(0)
    sc0 = chord_scale(0)
    start_tip = bm.verts.new((
        pos0[0] + sc0*tip_off*x0[0] + cy_c*y0[0] + sc0*dome_r*start_tan[0],
        pos0[1] + sc0*tip_off*x0[1] + cy_c*y0[1] + sc0*dome_r*start_tan[1],
        pos0[2] + sc0*tip_off*x0[2] + cy_c*y0[2] + sc0*dome_r*start_tan[2],
    ))

    # --- Main body rings ---
    for idx in range(n_f):
        all_rings.append(make_body_ring(idx))

    # --- End cap ---
    for i in range(1, n_cap):
        all_rings.append(make_cap_ring(n_f - 1, end_tan, i / n_cap))

    # End tip vertex
    posN = frames[-1][0]
    xN, yN = get_frame_dirs(n_f - 1)
    scN = chord_scale(n_f - 1)
    end_tip = bm.verts.new((
        posN[0] + scN*tip_off*xN[0] + cy_c*yN[0] + scN*dome_r*end_tan[0],
        posN[1] + scN*tip_off*xN[1] + cy_c*yN[1] + scN*dome_r*end_tan[1],
        posN