import argparse
import copy
from itertools import combinations
from pathlib import Path

import cv2
import imutils
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_IMAGE = PROJECT_ROOT / "dataset" / "form2.png"
DEFAULT_TEMPLATE_IMAGE = PROJECT_ROOT / "dataset" / "form1.png"
MARKER_TEMPLATE_IMAGE = PROJECT_ROOT / "dataset" / "form_markers.png"
DEFAULT_ANSWER_KEY = PROJECT_ROOT / "answer_key.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
MIN_ALIGNMENT_SCORE = 0.70
# A small photo upscales into a softer edge map, so the absolute score drops
# while the correct rotation still leads the other three.
MIN_ALIGNMENT_FLOOR = 0.48
MIN_ALIGNMENT_MARGIN = 0.12
# Faint red print has fewer edges than the blank form, so cosine stays low
# even when the warp sits on the grid. Coverage asks only whether the ink
# that is visible lands on the template, and the correct rotation must lead.
MIN_EDGE_AGREEMENT = 0.66
MIN_AGREEMENT_MARGIN = 0.08
MIN_COSINE_WITH_AGREEMENT = 0.28
EDGE_AGREEMENT_RADIUS = 5
# Empty bubble interiors sit near 0.03; a pencilled bubble is near 0.95.
MARKED_FILL = 0.45

_REFERENCE_CACHE = {}
_LAYOUT_CACHE = {}


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()

    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate.resolve()

    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        package_candidate = PROJECT_ROOT.parent / path
        if package_candidate.exists():
            return package_candidate.resolve()

    return project_candidate.resolve()


def ensure_bgr(img):
    if img is None:
        raise ValueError("Could not read image")
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def order_points(points):
    points = np.asarray(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def dark_mask(gray):
    """Solid black squares, kept apart from pencil marks and page shadows.

    Otsu and adaptive thresholds join a corner square to the shadow along a
    wrinkled page, so the square is no longer its own contour. A black-hat
    keeps dark blobs about the size of those squares.
    """
    short_side = min(gray.shape[:2])
    kernel_size = max(21, int(short_side * 0.045))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    peak = float(np.percentile(hat, 99))
    threshold = max(25, int(peak * 0.45))
    _, mask = cv2.threshold(hat, threshold, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_marker_candidates(img):
    img = ensure_bgr(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = dark_mask(gray)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    h_img, w_img = gray.shape[:2]
    short_side = min(h_img, w_img)
    min_side = max(6, int(short_side * 0.003))
    max_side = max(24, int(short_side * 0.09))

    candidates = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_side or h < min_side or w > max_side or h > max_side:
            continue

        aspect = w / float(h)
        if not 0.65 <= aspect <= 1.45:
            continue

        rect_area = w * h
        contour_area = cv2.contourArea(cnt)
        if rect_area <= 0:
            continue

        fill_ratio = contour_area / float(rect_area)
        roi = mask[y:y + h, x:x + w]
        dark_ratio = cv2.countNonZero(roi) / float(rect_area)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if dark_ratio < 0.5 or fill_ratio < 0.55:
            continue
        if len(approx) > 6 and fill_ratio < 0.8:
            continue

        candidates.append({
            "box": (x, y, w, h),
            "center": (x + w / 2.0, y + h / 2.0),
            "area": contour_area,
            "score": contour_area * fill_ratio * dark_ratio,
        })

    return candidates


def _large_marker_cluster(candidates):
    """Corner squares are the big fiducials. The smaller ticks sit inside them."""
    if not candidates:
        return []
    largest = max(item["area"] for item in candidates)
    cluster = [item for item in candidates if item["area"] >= largest * 0.40]
    if len(cluster) >= 4:
        return cluster
    return candidates[:40]


def detect_corner_markers(img):
    img = ensure_bgr(img)
    h_img, w_img = img.shape[:2]
    candidates = _large_marker_cluster(find_marker_candidates(img))
    if len(candidates) < 4:
        raise ValueError(f"Need 4 corner markers, found {len(candidates)}")

    best_points = None
    best_score = -1
    image_area = float(w_img * h_img)
    for indexes in combinations(range(len(candidates)), 4):
        points = np.array([candidates[i]["center"] for i in indexes], dtype=np.float32)
        ordered = order_points(points)
        quad_area = abs(cv2.contourArea(ordered.reshape(-1, 1, 2)))
        if quad_area < image_area * 0.005:
            continue

        width_top = np.linalg.norm(ordered[1] - ordered[0])
        width_bottom = np.linalg.norm(ordered[2] - ordered[3])
        height_left = np.linalg.norm(ordered[3] - ordered[0])
        height_right = np.linalg.norm(ordered[2] - ordered[1])
        if min(width_top, width_bottom, height_left, height_right) < min(h_img, w_img) * 0.08:
            continue

        marker_score = sum(candidates[i]["score"] for i in indexes)
        balance = min(width_top, width_bottom) / max(width_top, width_bottom)
        balance *= min(height_left, height_right) / max(height_left, height_right)
        score = quad_area * balance + marker_score * 25

        if score > best_score:
            best_score = score
            best_points = ordered

    if best_points is None:
        raise ValueError("Could not identify the four outer corner markers")

    return best_points


def edge_signature(img):
    gray = cv2.cvtColor(ensure_bgr(img), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1) > 0


def template_similarity(img, reference_img):
    img_edges = edge_signature(img)
    ref_edges = edge_signature(reference_img)
    overlap = np.logical_and(img_edges, ref_edges).sum()
    denom = np.sqrt(img_edges.sum() * ref_edges.sum())
    if denom == 0:
        return 0.0
    return overlap / denom


def edge_agreement(img, reference_img, radius=EDGE_AGREEMENT_RADIUS):
    """Fraction of this photo's edges that land on the template.

    Cosine shrinks when the print is faint because the blank form has many
    more edges. A dilated template asks only whether the edges we do see
    sit on the printed grid.
    """
    img_edges = edge_signature(img)
    count = int(img_edges.sum())
    if count < 50:
        return 0.0
    ref_edges = edge_signature(reference_img)
    kernel = np.ones((radius, radius), np.uint8)
    near_ref = cv2.dilate(ref_edges.astype(np.uint8), kernel) > 0
    return float(np.logical_and(img_edges, near_ref).sum() / count)


def load_reference(reference_path=DEFAULT_REFERENCE_IMAGE):
    reference_path = resolve_path(reference_path)
    cache_key = str(reference_path)
    if cache_key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[cache_key]

    reference_img = cv2.imread(str(reference_path))
    if reference_img is None:
        raise FileNotFoundError(f"Reference form not found: {reference_path}")

    reference_img = ensure_bgr(reference_img)
    reference_markers = detect_corner_markers(reference_img)
    _REFERENCE_CACHE[cache_key] = (reference_img, reference_markers)
    return _REFERENCE_CACHE[cache_key]


def _quad_from_contour(contour):
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return None
    for factor in (0.015, 0.02, 0.03, 0.045):
        approx = cv2.approxPolyDP(contour, factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    if len(hull) < 4:
        return None
    sums = hull.sum(axis=1)
    diff = np.diff(hull, axis=1).reshape(-1)
    quad = np.vstack([
        hull[np.argmin(sums)],
        hull[np.argmin(diff)],
        hull[np.argmax(sums)],
        hull[np.argmax(diff)],
    ]).astype(np.float32)
    if len(np.unique(np.round(quad, 1), axis=0)) < 4:
        return None
    return quad


def _paper_quad_from_mask(mask, image_shape):
    h_img, w_img = image_shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    contour = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    fraction = area / float(h_img * w_img)
    # A scan that already fills the frame is handled by the corner markers.
    # A sheet photographed from farther away can be only a few percent of the frame.
    if fraction < 0.02 or fraction > 0.92:
        return None
    quad = _quad_from_contour(contour)
    if quad is None:
        return None
    rect = order_points(quad)
    width = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
    height = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
    if width < 40 or height < 40:
        return None
    aspect = width / height
    if not 0.40 <= aspect <= 2.5:
        return None
    return rect


def find_paper_quad(img):
    """Largest bright page on a darker background, or None when the sheet fills the frame."""
    gray = cv2.cvtColor(ensure_bgr(img), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h_img, w_img = gray.shape[:2]
    kernel = max(15, int(min(h_img, w_img) * 0.02))
    if kernel % 2 == 0:
        kernel += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    masks = []
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    masks.append(otsu)
    # A light desk sits close to the paper, so Otsu can miss it. Try a stricter bright cut.
    bright_cut = int(np.percentile(blurred, 55))
    _, strict = cv2.threshold(blurred, max(140, bright_cut), 255, cv2.THRESH_BINARY)
    masks.append(strict)

    best = None
    best_area = 0
    for mask in masks:
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
        quad = _paper_quad_from_mask(closed, (h_img, w_img))
        if quad is None:
            continue
        area = abs(cv2.contourArea(quad.reshape(-1, 1, 2)))
        if area > best_area:
            best_area = area
            best = quad
    return best


def warp_quad(img, quad):
    rect = order_points(quad)
    width = int(round(max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))))
    height = int(round(max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))))
    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(
        ensure_bgr(img),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )


def rectify_sheet(img):
    """Pull a photographed page off the desk and view it top-down. Scans pass through."""
    quad = find_paper_quad(img)
    if quad is None:
        return ensure_bgr(img), None
    return warp_quad(img, quad), quad


def template_candidates(reference_path):
    """The requested form, plus the built-in sheets that share the same answer blocks."""
    found = []
    for path in (reference_path, DEFAULT_REFERENCE_IMAGE, MARKER_TEMPLATE_IMAGE):
        resolved = resolve_path(path)
        if resolved.exists() and resolved not in found:
            found.append(resolved)
    return found


def normalize_answer_sheet(img, reference_path=DEFAULT_REFERENCE_IMAGE, debug_dir=None):
    """Warp a photo onto the built-in form it matches.

    Some sheets use four corner squares. Others scatter smaller squares down the
    page. In both cases the outer four squares define the page, and the winning
    template is the blank form whose printed grid lines up after that warp.
    """
    img = ensure_bgr(img)
    best = None
    last_error = None
    for candidate in template_candidates(reference_path):
        try:
            warped, matrix, score = _align_to_reference(img, candidate, debug_dir=None)
        except ValueError as exc:
            last_error = exc
            continue
        if best is None or score > best[2]:
            best = (warped, matrix, score, candidate)
    if best is None:
        raise last_error or ValueError(
            "Could not align the sheet. The outer black squares need to stay visible."
        )
    warped, matrix, score, chosen = best
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "aligned_sheet.jpg"), warped)
    return warped, matrix, score, chosen


def _warp_markers(image, reference_img, reference_markers):
    h_ref, w_ref = reference_img.shape[:2]
    source_markers = detect_corner_markers(image)
    best_warped = None
    best_score = -1.0
    second_score = -1.0
    best_agree = -1.0
    second_agree = -1.0
    best_matrix = None
    for shift in range(4):
        shifted_source = np.roll(source_markers, -shift, axis=0)
        matrix = cv2.getPerspectiveTransform(shifted_source, reference_markers)
        warped = cv2.warpPerspective(
            image,
            matrix,
            (w_ref, h_ref),
            flags=cv2.INTER_LINEAR,
            borderValue=(255, 255, 255),
        )
        score = float(template_similarity(warped, reference_img))
        agree = float(edge_agreement(warped, reference_img))
        if score > best_score:
            second_score = best_score
            second_agree = best_agree
            best_score = score
            best_agree = agree
            best_warped = warped
            best_matrix = matrix
        elif score > second_score:
            second_score = score
            second_agree = agree
    return best_warped, best_matrix, best_score, second_score, best_agree, second_agree


def _align_to_reference(img, reference_path, debug_dir=None):
    img = ensure_bgr(img)
    reference_img, reference_markers = load_reference(reference_path)
    # A wrinkled photo can make the page outline the wrong quadrilateral.
    # Keep the marker warp, and also the outline warp, whichever matches the form better.
    candidates = []
    try:
        candidates.append(_warp_markers(img, reference_img, reference_markers))
    except ValueError:
        pass
    rectified, paper_quad = rectify_sheet(img)
    if paper_quad is not None:
        try:
            candidates.append(_warp_markers(rectified, reference_img, reference_markers))
        except ValueError:
            pass
    if not candidates:
        raise ValueError(
            "Could not see the black square markers. Keep the outer squares in the frame."
        )
    best_warped, best_matrix, best_score, second_score, best_agree, second_agree = max(
        candidates, key=lambda item: item[2]
    )
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "rectified_sheet.jpg"), rectified)
        cv2.imwrite(str(debug_dir / "aligned_sheet.jpg"), best_warped)
        if paper_quad is not None:
            preview = img.copy()
            cv2.polylines(preview, [paper_quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3)
            cv2.imwrite(str(debug_dir / "paper_quad.jpg"), preview)

    aligned = best_score >= MIN_ALIGNMENT_SCORE or (
        best_score >= MIN_ALIGNMENT_FLOOR and best_score - second_score >= MIN_ALIGNMENT_MARGIN
    ) or (
        best_agree >= MIN_EDGE_AGREEMENT
        and best_agree - second_agree >= MIN_AGREEMENT_MARGIN
        and best_score >= MIN_COSINE_WITH_AGREEMENT
    )
    if not aligned:
        raise ValueError(
            f"Alignment confidence too low ({best_score:.3f}). "
            "Make sure all four black corner markers are visible and not cropped."
        )

    return best_warped, best_matrix, best_score


def _ink_channel(image):
    """Darkness channel. Red printed circles count as ink, not only black ones."""
    if len(image.shape) == 2:
        return image
    return image.min(axis=2)


def _drop_labelled_circles(gray, circles):
    """Row-number circles have a glyph in the middle. Answer bubbles are empty rings."""
    kept = []
    for x, y, radius in circles:
        radius = max(3, int(round(radius)))
        xi, yi = int(round(x)), int(round(y))
        patch = gray[max(0, yi - radius):yi + radius, max(0, xi - radius):xi + radius]
        if patch.size < 9:
            continue
        yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
        dist = np.hypot(yy - patch.shape[0] / 2.0, xx - patch.shape[1] / 2.0)
        interior = patch[dist < radius * 0.42]
        ring = patch[(dist >= radius * 0.62) & (dist <= radius * 1.05)]
        if interior.size == 0 or ring.size == 0:
            continue
        interior_mean = float(interior.mean())
        # A pencilled bubble is dark in the middle. An empty bubble is a bright hole
        # inside a darker ring. A printed digit is neither, so it is left out.
        if interior_mean < 110 or interior_mean > float(ring.mean()) + 12:
            kept.append((x, y, radius))
    if not kept:
        return circles
    return np.asarray(kept, dtype=np.float32)


def detect_bubble_centers(image, min_count=80):
    """Circle centres. Hough keeps stacked bubbles whose edges line up into a column."""
    gray = _ink_channel(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    height, width = gray.shape[:2]
    min_radius = max(4, int(round(width * 0.008)))
    max_radius = max(min_radius + 2, int(round(width * 0.018)))
    min_dist = max(8, int(round(width * 0.016)))
    found = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if found is not None and len(found[0]) >= min_count:
        circles = _drop_labelled_circles(gray, found[0])
        if len(circles) >= min_count:
            return circles[:, :2].astype(np.float32), float(np.median(circles[:, 2]) * 2)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        8,
    )
    height, width = gray.shape[:2]
    line_len = max(12, int(round(width * 0.028)))
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    )
    lines = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8), iterations=1)
    ink = cv2.subtract(binary, lines)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_side = max(7, int(round(width * 0.012)))
    max_side = max(min_side + 2, int(round(width * 0.030)))
    centers = []
    widths = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area < 25 or peri <= 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        x, y, bw, bh = cv2.boundingRect(cnt)
        if circularity < 0.65 or not 0.75 <= bw / float(bh) <= 1.35:
            continue
        if not min_side <= bw <= max_side or not min_side <= bh <= max_side:
            continue
        centers.append((x + bw / 2.0, y + bh / 2.0))
        widths.append((bw + bh) / 2.0)

    if len(centers) < min_count:
        raise ValueError(
            f"Found only {len(centers)} answer bubbles. "
            "The sheet grid has to stay visible after alignment."
        )
    return np.asarray(centers, dtype=np.float32), float(np.median(widths))


def _cluster_rows(points, gap):
    order = np.argsort(points[:, 1])
    ordered = points[order]
    rows = []
    bucket = [ordered[0]]
    for point in ordered[1:]:
        if point[1] - np.median([item[1] for item in bucket]) > gap:
            rows.append(np.vstack(bucket))
            bucket = [point]
        else:
            bucket.append(point)
    rows.append(np.vstack(bucket))
    return rows


def _horizontal_pitch(rows):
    diffs = []
    for row in rows:
        xs = np.sort(row[:, 0])
        delta = np.diff(xs)
        delta = delta[(delta > 5) & (delta < 80)]
        if len(delta):
            diffs.append(np.median(delta))
    if not diffs:
        raise ValueError("Could not measure horizontal bubble spacing.")
    return float(np.median(diffs))


def _column_centers(rows, min_hits):
    pitch = _horizontal_pitch(rows)
    xs = np.concatenate([row[:, 0] for row in rows])
    xs.sort()
    groups = [[xs[0]]]
    for value in xs[1:]:
        if value - np.median(groups[-1]) > pitch * 0.55:
            groups.append([value])
        else:
            groups[-1].append(value)
    kept = [float(np.median(group)) for group in groups if len(group) >= min_hits]
    return kept, pitch


def _band_rows(rows, min_gap):
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(np.median(row[:, 1])))
    bands = [[ordered[0]]]
    for row in ordered[1:]:
        previous_y = float(np.median(bands[-1][-1][:, 1]))
        current_y = float(np.median(row[:, 1]))
        if current_y - previous_y > min_gap:
            bands.append([row])
        else:
            bands[-1].append(row)
    return bands


def _row_y(row):
    return float(np.median(row[:, 1]))


def _take_exact_rows(rows, count, target_len):
    ranked = sorted(rows, key=lambda row: (abs(len(row) - target_len), _row_y(row)))
    chosen = sorted(ranked[:count], key=_row_y)
    if len(chosen) != count:
        raise ValueError(f"Expected {count} bubble rows, found {len(rows)}")
    return chosen


def _groups_of(xs, size, n_groups, label):
    if len(xs) != size * n_groups:
        raise ValueError(f"{label}: expected {size * n_groups} columns, found {len(xs)}")
    groups = [xs[index * size:(index + 1) * size] for index in range(n_groups)]
    inside = [float(np.max(np.diff(group))) for group in groups]
    between = [groups[index + 1][0] - groups[index][-1] for index in range(n_groups - 1)]
    if between and min(between) <= max(inside) * 1.15:
        raise ValueError(f"{label}: bubble columns are not separated into {n_groups} blocks")
    return groups


def _grid(column_groups, row_ys):
    """column_groups is a list of x-lists. Returns [group][row][col] -> (x, y)."""
    grid = []
    for columns in column_groups:
        block = []
        for y in row_ys:
            block.append([(float(x), float(y)) for x in columns])
        grid.append(block)
    return grid


def _snap_y(rows, target_y, tolerance):
    best = None
    best_distance = tolerance
    for row in rows:
        distance = abs(_row_y(row) - target_y)
        if distance <= best_distance:
            best = row
            best_distance = distance
    if best is None:
        return target_y
    return _row_y(best)


def build_sheet_layout(reference_bgr):
    """Measure the answer grid from the printed bubbles. No fixed crop coordinates."""
    reference_bgr = ensure_bgr(reference_bgr)
    centers, bubble_width = detect_bubble_centers(reference_bgr)
    height, width = reference_bgr.shape[:2]
    ys = np.sort(centers[:, 1])
    deltas = np.diff(ys)
    deltas = deltas[(deltas >= 6) & (deltas <= 40)]
    if len(deltas) == 0:
        raise ValueError("Could not measure vertical bubble spacing.")
    pitch_y = float(np.median(deltas))
    rows = _cluster_rows(centers, pitch_y * 0.6)

    info_rows = []
    wide_rows = []
    digit_rows = []
    for row in rows:
        xs = np.sort(row[:, 0])
        span = float(xs[-1] - xs[0]) if len(xs) > 1 else 0
        dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 0
        median_x = float(np.median(xs))
        y = _row_y(row)
        # Student id sits on the right. A row that also reaches the left margin is not that grid.
        # Rows can be incomplete when some circles are faint, so the minimum count stays low.
        if 3 <= len(row) <= 14 and xs[0] > width * 0.50 and xs[-1] > width * 0.82:
            info_rows.append(row)
        elif len(row) >= 18 and y > height * 0.45 and dx <= pitch_y * 1.7:
            # Part III has about 24 bubbles in a row. Check it before the wider part I/II rows.
            digit_rows.append(row)
        elif len(row) >= 12 and span > width * 0.55 and dx >= pitch_y * 1.4:
            wide_rows.append(row)

    info_bands = [band for band in _band_rows(info_rows, pitch_y * 3.2) if len(band) >= 10]
    if not info_bands:
        raise ValueError("Could not find the student-id bubble grid.")
    info_rows = max(info_bands, key=len)
    # The boxes above digit 0 are one extra row. The ten digit rows are the lower run.
    if len(info_rows) > 10:
        info_rows = info_rows[-10:]
    info_xs, _ = _column_centers(info_rows, min_hits=max(4, int(0.45 * len(info_rows))))
    if len(info_xs) != 9:
        raise ValueError(f"Student-id block should have 9 columns, found {len(info_xs)}")
    split_at = int(np.argmax(np.diff(info_xs))) + 1
    sbd_xs = info_xs[:split_at]
    made_xs = info_xs[split_at:]
    if len(sbd_xs) != 6 or len(made_xs) != 3:
        raise ValueError("Could not separate the student-id columns from the exam-code columns.")
    info_ys = [_row_y(row) for row in info_rows]

    wide_bands = _band_rows(wide_rows, pitch_y * 2.2)
    wide_bands = [band for band in wide_bands if len(band) >= 4]
    if len(wide_bands) < 2:
        raise ValueError("Could not find part I and part II bubble blocks.")
    wide_bands = sorted(wide_bands, key=lambda band: _row_y(band[0]))
    # Part I is the taller block above part II.
    part_bands = sorted(wide_bands[:2], key=lambda band: _row_y(band[0]))
    part1_rows = _take_exact_rows(part_bands[0], 10, 16)
    part2_rows = _take_exact_rows(part_bands[1], 4, 16)
    part1_xs, _ = _column_centers(part1_rows, min_hits=max(4, int(0.45 * len(part1_rows))))
    part2_xs, _ = _column_centers(part2_rows, min_hits=max(2, int(0.4 * len(part2_rows))))
    part1_groups = _groups_of(part1_xs, 4, 4, "Part I")
    part2_groups = _groups_of(part2_xs, 4, 4, "Part II")
    part1_ys = [_row_y(row) for row in part1_rows]
    part2_ys = [_row_y(row) for row in part2_rows]

    digit_bands = [band for band in _band_rows(digit_rows, pitch_y * 2.2) if len(band) >= 10]
    if not digit_bands:
        raise ValueError("Could not find the part III digit grid.")
    digit_rows = _take_exact_rows(max(digit_bands, key=len), 10, 24)
    digit_xs, _ = _column_centers(digit_rows, min_hits=max(6, int(0.6 * len(digit_rows))))
    digit_groups = _groups_of(digit_xs, 4, 6, "Part III")
    digit_ys = [_row_y(row) for row in digit_rows]
    digit_pitch = float(np.median(np.diff(digit_ys)))
    # The sign row and the decimal row sit one and two steps above digit 0.
    # Sign is printed only in the first column; the decimal point only in the middle two.
    dot_y = _snap_y(rows, digit_ys[0] - digit_pitch, digit_pitch * 0.45)
    sign_y = _snap_y(rows, digit_ys[0] - 2 * digit_pitch, digit_pitch * 0.45)

    radius = max(3, int(round(bubble_width * 0.28)))
    return {
        "width": width,
        "height": height,
        "pitch_y": pitch_y,
        "radius": radius,
        "sbd": _grid([sbd_xs], info_ys)[0],
        "made": _grid([made_xs], info_ys)[0],
        "part1": _grid(part1_groups, part1_ys),
        "part2": _grid(part2_groups, part2_ys),
        "part3_columns": digit_groups,
        "part3_sign_y": sign_y,
        "part3_dot_y": dot_y,
        "part3_digit_ys": digit_ys,
    }


def template_image_for(reference_path):
    """Blank form when it is the same scan geometry, otherwise the alignment reference."""
    reference_path = resolve_path(reference_path)
    reference = cv2.imread(str(reference_path))
    if reference is None:
        raise FileNotFoundError(f"Reference form not found: {reference_path}")
    reference = ensure_bgr(reference)
    template_path = resolve_path(DEFAULT_TEMPLATE_IMAGE)
    if template_path.exists() and template_path != reference_path:
        template = cv2.imread(str(template_path))
        if template is not None and template.shape[:2] == reference.shape[:2]:
            return ensure_bgr(template), template_path
    return reference, reference_path


def get_sheet_layout(reference_path=DEFAULT_REFERENCE_IMAGE):
    reference_path = resolve_path(reference_path)
    template, template_path = template_image_for(reference_path)
    cache_key = str(template_path.resolve())
    if cache_key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[cache_key]
    layout = build_sheet_layout(template)
    _LAYOUT_CACHE[cache_key] = layout
    return layout


def bubble_fill(gray, x, y, radius):
    xi = int(round(x))
    yi = int(round(y))
    y0 = max(0, yi - radius)
    x0 = max(0, xi - radius)
    patch = gray[y0:yi + radius, x0:xi + radius]
    if patch.size == 0:
        return 0.0
    return float((patch < 120).mean())


def estimate_grid_shift(gray, layout):
    """Slide the measured grid a few pixels so it sits on this photo's bubbles."""
    try:
        detected, _ = detect_bubble_centers(gray, min_count=40)
    except ValueError:
        return 0.0, 0.0

    samples = []
    for row in layout["sbd"]:
        samples.extend(row[::3])
    for block in layout["part1"]:
        for row in block[::2]:
            samples.extend(row)
    samples = np.asarray(samples, dtype=np.float32)
    deltas = []
    for point in samples:
        distance = np.hypot(detected[:, 0] - point[0], detected[:, 1] - point[1])
        nearest = int(np.argmin(distance))
        if distance[nearest] <= 8:
            deltas.append(detected[nearest] - point)
    if len(deltas) < 25:
        return 0.0, 0.0
    shift = np.median(np.asarray(deltas), axis=0)
    if abs(shift[0]) > 6 or abs(shift[1]) > 6:
        return 0.0, 0.0
    return float(shift[0]), float(shift[1])


def _shifted(point, shift):
    return point[0] + shift[0], point[1] + shift[1]


def read_sheet(aligned_bgr, layout):
    # Minimum channel, so blue pen and red print both count as dark ink.
    gray = _ink_channel(aligned_bgr)
    if gray.shape[:2] != (layout["height"], layout["width"]):
        raise ValueError(
            f"Aligned sheet is {gray.shape[1]}x{gray.shape[0]}, "
            f"reference grid is {layout['width']}x{layout['height']}."
        )
    shift = estimate_grid_shift(gray, layout)
    radius = layout["radius"]
    marked_points = []

    def fill_at(point):
        x, y = _shifted(point, shift)
        return bubble_fill(gray, x, y, radius)

    def is_marked(point):
        value = fill_at(point)
        if value >= MARKED_FILL:
            marked_points.append(_shifted(point, shift))
        return value >= MARKED_FILL

    part1 = defaultdict(list)
    letters = "ABCD"
    for group_index, block in enumerate(layout["part1"]):
        for row_index, choices in enumerate(block):
            question = group_index * 10 + row_index + 1
            for letter, point in zip(letters, choices):
                if is_marked(point):
                    part1[question].append(letter)

    part2 = OrderedDict((question, defaultdict(list)) for question in range(1, 9))
    sub_labels = "abcd"
    for group_index, block in enumerate(layout["part2"]):
        left_question = group_index * 2 + 1
        right_question = group_index * 2 + 2
        for row_index, choices in enumerate(block):
            label = sub_labels[row_index]
            # Each row is Đúng/Sai for the odd question, then Đúng/Sai for the even one.
            if is_marked(choices[0]):
                part2[left_question][label].append(True)
            if is_marked(choices[1]):
                part2[left_question][label].append(False)
            if is_marked(choices[2]):
                part2[right_question][label].append(True)
            if is_marked(choices[3]):
                part2[right_question][label].append(False)

    symbols = "-.0123456789"
    part3 = defaultdict(list)
    part3_cells = {}
    for question, columns in enumerate(layout["part3_columns"], start=1):
        digits = []
        row_ys = _part3_row_ys(layout, question - 1)
        for column_index, x in enumerate(columns):
            chosen = ""
            for symbol, y in zip(symbols, row_ys):
                # The form only prints a minus in column 0 and a decimal point in columns 1 and 2.
                if symbol == "-" and column_index != 0:
                    continue
                if symbol == "." and column_index not in (1, 2):
                    continue
                point = (x, y)
                if is_marked(point):
                    chosen = symbol
                    break
            digits.append(chosen)
        part3[question].append("".join(digits))
        part3_cells[question] = digits

    def columns_of(row_points):
        width_n = len(row_points[0])
        return [[row[col] for row in row_points] for col in range(width_n)]

    def read_columns(row_points):
        chars = []
        for column in columns_of(row_points):
            fills = [fill_at(point) for point in column]
            choice = int(np.argmax(fills))
            # An empty column has no dark bubble. Guessing its darkest cell
            # turns a blank exam-code digit into 0.
            chars.append(str(choice) if max(fills) >= MARKED_FILL else "")
            if max(fills) >= MARKED_FILL:
                marked_points.append(_shifted(column[choice], shift))
        return {index + 1: [char] for index, char in enumerate(chars)}

    return part1, part2, part3, read_columns(layout["sbd"]), read_columns(layout["made"]), marked_points, shift, part3_cells


def grade_part1(df, student_answer_dict):
    score = 0
    answer_dict = defaultdict(list)
    for _, row in df.iterrows():
        question_number = int(row["Question"])
        part = row["Section"]
        if 1 <= question_number <= 40 and part == "I":
            answer_dict[question_number].append(row["Answer"])
    for key, value in answer_dict.items():
        if value == student_answer_dict[key]:
            score += 1
    return f"{score}/40"


def grade_part2(df, student_answer_dict):
    score = 0
    answer_dict = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        question_number = int(row["Question"])
        part = row["Section"]
        if 1 <= question_number <= 8 and part == "II":
            answer_dict[question_number][row["Sub-question"]].append(row["Answer"])
    for key, value in answer_dict.items():
        for sub_key, correct_ans in value.items():
            if correct_ans == list(map(str, student_answer_dict[key][sub_key])):
                score += 1
    return f"{score}/32"


def grade_part3(df, student_answer_dict):
    score = 0
    answer_dict = defaultdict(list)
    for _, row in df.iterrows():
        question_number = int(row["Question"])
        part = row["Section"]
        if 1 <= question_number <= 6 and part == "III":
            answer_dict[question_number].append(row["Answer"])

    for key, value in answer_dict.items():
        if value == student_answer_dict.get(key, []):
            score += 1
    return f"{score}/6"


def _draw_check(img, x, y, size, thickness=2):
    size = max(6, int(size))
    x, y = int(round(x)), int(round(y))
    start = (x, y + size // 5)
    corner = (x + int(size * 0.35), y + int(size * 0.55))
    end = (x + size, y - int(size * 0.35))
    green = (30, 150, 40)
    cv2.line(img, start, corner, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.line(img, corner, end, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.line(img, start, corner, green, thickness, cv2.LINE_AA)
    cv2.line(img, corner, end, green, thickness, cv2.LINE_AA)


def _mark_ring_radius(layout):
    """Ring that stays inside its own bubble. A larger one reaches the next row and the two rings interlock."""
    pitch = float(layout["pitch_y"])
    return max(4, min(int(layout["radius"]) + 1, int(round(pitch * 0.26))))


def _draw_red_ring(img, x, y, radius):
    cv2.circle(
        img,
        (int(round(x)), int(round(y))),
        max(4, int(radius)),
        (0, 0, 220),
        2,
        cv2.LINE_AA,
    )


def _draw_red_text(img, text, x, y, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(round(x)), int(round(y))
    cv2.rectangle(img, (x - 1, y - height - 1), (x + width + 1, y + baseline), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 220), thickness, cv2.LINE_AA)


def _key_maps(df):
    part1 = defaultdict(list)
    part2 = defaultdict(lambda: defaultdict(list))
    part3 = defaultdict(list)
    for _, row in df.iterrows():
        section = str(row["Section"]).strip()
        question = int(row["Question"])
        answer = "" if pd.isna(row["Answer"]) else str(row["Answer"]).strip()
        if section == "I":
            part1[question].append(answer)
        elif section == "II":
            sub = "" if pd.isna(row["Sub-question"]) else str(row["Sub-question"]).strip().lower()
            part2[question][sub].append(answer)
        elif section == "III":
            part3[question].append(answer)
    return part1, part2, part3


def _align_part3_answer(answer, column_count):
    """Place a written answer onto the four columns: sign, two middle digits, last digit."""
    assigned = [""] * column_count
    index = 0
    for column in range(column_count):
        if index >= len(answer):
            break
        symbol = answer[index]
        if column == 0 and symbol == "-":
            assigned[column] = symbol
            index += 1
            continue
        if column in (1, 2) and symbol == ".":
            assigned[column] = symbol
            index += 1
            continue
        if symbol in "0123456789":
            assigned[column] = symbol
            index += 1
            continue
        index += 1
    return assigned


def _part2_choice_point(layout, question, sub_label, marked_true, shift):
    group = (question - 1) // 2
    row = "abcd".index(sub_label)
    column = 0 if marked_true else 1
    if question % 2 == 0:
        column += 2
    return _shifted(layout["part2"][group][row][column], shift)


def mark_correct_and_wrong(img, layout, shift, part1, part2, part3, df, part3_cells=None):
    """Green check beside a right answer. Red mark shows the right answer when it is wrong."""
    key1, key2, key3 = _key_maps(df)
    part3_cells = part3_cells or {}
    radius = _mark_ring_radius(layout)
    # The check and the red letter share the gap between rows. Taller marks overlap the next question.
    check_size = max(6, int(round(float(layout["pitch_y"]) * 0.36)))
    text_scale = float(np.clip(img.shape[1] / float(layout["width"]) * 0.24, 0.22, 0.28))

    for question, correct in key1.items():
        block = (question - 1) // 10
        row = (question - 1) % 10
        if block >= len(layout["part1"]) or row >= len(layout["part1"][block]):
            continue
        choices = [_shifted(point, shift) for point in layout["part1"][block][row]]
        anchor = choices[0]
        if part1[question] == correct:
            _draw_check(img, anchor[0] - check_size * 1.7, anchor[1] - check_size * 0.2, check_size)
            continue
        _draw_red_text(img, "".join(correct), anchor[0] - check_size * 1.8, anchor[1] + check_size * 0.35, text_scale)
        for letter in correct:
            index = "ABCD".find(letter.upper())
            if 0 <= index < len(choices):
                _draw_red_ring(img, choices[index][0], choices[index][1], radius)

    for question, subs in key2.items():
        for sub_label, correct in subs.items():
            if sub_label not in "abcd" or not correct:
                continue
            is_true = correct[0] == "True"
            point = _part2_choice_point(layout, question, sub_label, is_true, shift)
            pair_left = _part2_choice_point(layout, question, sub_label, True, shift)
            student = list(map(str, part2[question][sub_label]))
            if student == correct:
                _draw_check(img, pair_left[0] - check_size * 1.35, pair_left[1] - check_size * 0.15, check_size * 0.8)
            else:
                _draw_red_ring(img, point[0], point[1], radius)

    symbols = "-.0123456789"
    for question, correct in key3.items():
        if question < 1 or question > len(layout["part3_columns"]):
            continue
        columns = layout["part3_columns"][question - 1]
        row_ys = _part3_row_ys(layout, question - 1)
        answer = "".join(correct)
        student_cols = part3_cells.get(question, [""] * len(columns))
        whole_correct = part3.get(question, []) == correct
        header = _shifted((columns[0], row_ys[0] - layout["pitch_y"] * 2.15), shift)
        if whole_correct:
            _draw_check(img, header[0], header[1], check_size * 1.15, thickness=3)
        else:
            _draw_red_text(img, answer, header[0], header[1] + check_size * 0.35, text_scale)
        expected_cols = _align_part3_answer(answer, len(columns))
        for column_index, symbol in enumerate(expected_cols):
            if symbol not in symbols:
                continue
            point = _shifted((columns[column_index], row_ys[symbols.index(symbol)]), shift)
            student_symbol = student_cols[column_index] if column_index < len(student_cols) else ""
            if student_symbol == symbol:
                _draw_check(img, point[0] + radius * 0.15, point[1] - radius * 2.3, max(8, int(radius * 1.7)))
            elif not whole_correct:
                _draw_red_ring(img, point[0], point[1], radius)


def _put_label(img, text, origin, scale):
    height, width = img.shape[:2]
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    x = int(np.clip(x, 8, max(8, width - 90)))
    y = int(np.clip(y, 22, height - 8))
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def annotate_result_image(img, layout, diemp1, diemp2, diemp3, sbd_str, made_str, shift=(0.0, 0.0), part1=None, part2=None, part3=None, answer_key=None, part3_cells=None):
    img = img.copy()
    if answer_key is not None and part1 is not None:
        mark_correct_and_wrong(img, layout, shift, part1, part2, part3, answer_key, part3_cells)
    scale = max(0.4, img.shape[1] / float(layout["width"]) * 0.5)
    pitch = layout["pitch_y"]
    sbd_origin = layout["sbd"][0][0]
    made_origin = layout["made"][0][0]
    # Sit each score in the blank band above that section, on the right, clear of the printed title.
    part1_origin = layout["part1"][3][0][0]
    part2_origin = layout["part2"][3][0][0]
    part3_x = layout["part3_columns"][4][0]
    part3_y = layout["part3_sign_y"]
    _put_label(img, sbd_str, (sbd_origin[0], sbd_origin[1] - pitch * 3.4), scale)
    _put_label(img, made_str, (made_origin[0], made_origin[1] - pitch * 3.4), scale)
    _put_label(img, diemp1, (part1_origin[0], part1_origin[1] - pitch * 1.85), scale)
    _put_label(img, diemp2, (part2_origin[0], part2_origin[1] - pitch * 2.9), scale)
    _put_label(img, diemp3, (part3_x, part3_y - pitch * 3.05), scale)
    return img


def draw_marked_bubbles(img, points, radius):
    canvas = img.copy()
    for x, y in points:
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius + 3, (0, 0, 255), 1)
    return canvas


def _part3_row_ys(layout, question_index):
    per_question = layout.get("part3_question_ys")
    if per_question and 0 <= question_index < len(per_question):
        return per_question[question_index]
    return [layout["part3_sign_y"], layout["part3_dot_y"], *layout["part3_digit_ys"]]


def _filled_mark_blobs(image):
    """Centres of pencil or pen fills. Empty red rings stay out."""
    ink = _ink_channel(image)
    _, dark = cv2.threshold(ink, 105, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    blobs = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < 40 or area > 400 or width > 24 or height > 24:
            continue
        if not 0.55 <= width / float(max(height, 1)) <= 1.7:
            continue
        blobs.append(centroids[index])
    if not blobs:
        return np.zeros((0, 2), np.float32)
    return np.asarray(blobs, np.float32)


def _column_mark_matches(rows, blobs, x_gate, y_gate):
    """Pair filled blobs to one column, top to bottom, without jumping a row."""
    matches = []
    if len(blobs) == 0:
        return matches
    for column_index in range(len(rows[0])):
        column_x = float(np.median([row[column_index][0] for row in rows]))
        column_blobs = sorted(
            (blob for blob in blobs if abs(blob[0] - column_x) <= x_gate),
            key=lambda blob: blob[1],
        )
        template_y = [row[column_index][1] for row in rows]
        row_index = 0
        for blob in column_blobs:
            if row_index >= len(template_y):
                break
            while (
                row_index + 1 < len(template_y)
                and abs(template_y[row_index + 1] - blob[1]) <= abs(template_y[row_index] - blob[1])
            ):
                row_index += 1
            if abs(template_y[row_index] - blob[1]) <= y_gate:
                matches.append((row_index, column_index, blob))
                row_index += 1
    return matches


def _refine_choice_rows(rows, blobs, pitch):
    """Shift each answer row onto the fills. Wrinkles move a row by several pixels."""
    if len(blobs) == 0 or len(rows) < 2:
        return rows
    gap = float(np.median(np.diff([point[0] for point in rows[0]])))
    matches = _column_mark_matches(rows, blobs, gap * 0.46, pitch * 0.85)
    buckets = defaultdict(list)
    for row_index, column_index, blob in matches:
        origin = rows[row_index][column_index]
        buckets[row_index].append((blob[0] - origin[0], blob[1] - origin[1]))
    if len(buckets) < 2:
        return rows
    offsets = {row_index: np.median(np.asarray(values), axis=0) for row_index, values in buckets.items()}
    known = sorted(offsets)
    full = {}
    for row_index in range(len(rows)):
        if row_index in offsets:
            full[row_index] = offsets[row_index]
            continue
        earlier = [index for index in known if index < row_index]
        later = [index for index in known if index > row_index]
        if not earlier:
            full[row_index] = offsets[later[0]]
        elif not later:
            full[row_index] = offsets[earlier[-1]]
        else:
            left, right = earlier[-1], later[0]
            weight = (row_index - left) / float(right - left)
            full[row_index] = offsets[left] * (1 - weight) + offsets[right] * weight
    refined = []
    for row_index, row in enumerate(rows):
        dx, dy = full[row_index]
        refined.append([(point[0] + dx, point[1] + dy) for point in row])
    return refined


def _printed_part3_points(columns, row_ys):
    """Bubbles that are actually printed: minus only in column 0, decimal in columns 1–2."""
    symbols = "-.0123456789"
    points = []
    for column_index, x in enumerate(columns):
        for symbol_index, y in enumerate(row_ys):
            symbol = symbols[symbol_index]
            if symbol == "-" and column_index != 0:
                continue
            if symbol == "." and column_index not in (1, 2):
                continue
            points.append((float(x), float(y)))
    return np.asarray(points, np.float32)


def _best_circle_translation(points, circles, search=22.0, tolerance=4.0):
    """Shift that lands the most grid points on detected circles.

    Wrinkles move a whole answer block by about one bubble, and a one-row
    slide still hits many circles because the grid repeats. The nearer shift
    wins when both explain the same circles.
    """
    points = np.asarray(points, np.float32)
    circles = np.asarray(circles, np.float32)
    if len(points) < 8 or len(circles) < 8:
        return 0.0, 0.0, 0, 99.0
    min_xy = points.min(axis=0) - (search + 6)
    max_xy = points.max(axis=0) + (search + 6)
    local = circles[
        (circles[:, 0] >= min_xy[0])
        & (circles[:, 0] <= max_xy[0])
        & (circles[:, 1] >= min_xy[1])
        & (circles[:, 1] <= max_xy[1])
    ]
    if len(local) < 8:
        local = circles
    shifts_x = np.arange(-search, search + 0.01, 1.0, dtype=np.float32)
    shifts_y = np.arange(-search, search + 0.01, 1.0, dtype=np.float32)
    grid = np.stack(np.meshgrid(shifts_x, shifts_y, indexing="xy"), axis=-1).reshape(-1, 2)
    inliers = np.empty(len(grid), np.int32)
    medians = np.empty(len(grid), np.float32)
    chunk = 250
    for start in range(0, len(grid), chunk):
        shifts = grid[start:start + chunk]
        moved = points[None, :, :] + shifts[:, None, :]
        delta = moved[:, :, None, :] - local[None, None, :, :]
        nearest = np.hypot(delta[..., 0], delta[..., 1]).min(axis=2)
        inliers[start:start + len(shifts)] = (nearest <= tolerance).sum(axis=1)
        medians[start:start + len(shifts)] = np.median(nearest, axis=1)
    norms = np.hypot(grid[:, 0], grid[:, 1])
    best = int(np.lexsort((medians, norms, -inliers))[0])
    return float(grid[best, 0]), float(grid[best, 1]), int(inliers[best]), float(medians[best])


def _tighten_part3_rows(columns, row_ys, circles):
    """Nudge each symbol row onto its circles, by at most a few pixels."""
    symbols = "-.0123456789"
    tightened = []
    dxs = []
    for symbol_index, y in enumerate(row_ys):
        symbol = symbols[symbol_index]
        row_points = []
        for column_index, x in enumerate(columns):
            if symbol == "-" and column_index != 0:
                continue
            if symbol == "." and column_index not in (1, 2):
                continue
            row_points.append((x, y))
        deltas = []
        for x, yy in row_points:
            distance = np.hypot(circles[:, 0] - x, circles[:, 1] - yy)
            nearest = int(np.argmin(distance))
            if distance[nearest] <= 5:
                deltas.append(circles[nearest] - np.array([x, yy], np.float32))
        if len(deltas) >= 2:
            deltas = np.asarray(deltas, np.float32)
            med = np.median(deltas, axis=0)
            spread = float(np.median(np.hypot(deltas[:, 0] - med[0], deltas[:, 1] - med[1])))
            if spread <= 1.6 and max(abs(float(med[0])), abs(float(med[1]))) <= 4:
                tightened.append(float(y) + float(med[1]))
                dxs.append(float(med[0]))
                continue
        tightened.append(float(y))
    dx = float(np.median(dxs)) if dxs else 0.0
    if abs(dx) > 4:
        dx = 0.0
    return tightened, dx


def _snap_part3_question(columns, base_ys, circles, blobs, pitch):
    """Put one Part III question on the circles. Fall back to pen fills if circles do not fit."""
    points = _printed_part3_points(columns, base_ys)
    dx, dy, inliers, median_distance = _best_circle_translation(points, circles)
    enough = inliers >= max(18, int(round(0.6 * len(points)))) and median_distance <= 3.5
    if enough:
        shifted_columns = [float(x) + dx for x in columns]
        shifted_ys = [float(y) + dy for y in base_ys]
        tightened_ys, extra_dx = _tighten_part3_rows(shifted_columns, shifted_ys, circles)
        return [x + extra_dx for x in shifted_columns], tightened_ys
    rows = [[(float(x), float(y)) for x in columns] for y in base_ys]
    refined = _refine_choice_rows(rows, blobs, pitch)
    return (
        [refined[0][index][0] for index in range(len(columns))],
        [refined[row_index][0][1] for row_index in range(len(refined))],
    )


def _red_print_mask(image):
    """Printed bubble rings are red. Blue pen and the black squares stay out of the mask."""
    if image.ndim != 3:
        return np.zeros(image.shape[:2], np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, np.array((0, 70, 50), np.uint8), np.array((12, 255, 255), np.uint8))
    high = cv2.inRange(hsv, np.array((168, 70, 50), np.uint8), np.array((180, 255, 255), np.uint8))
    blue, green, red = cv2.split(image)
    redder = (
        (red.astype(np.int16) > green.astype(np.int16) + 20)
        & (red.astype(np.int16) > blue.astype(np.int16) + 20)
        & (red > 60)
    )
    mask = cv2.bitwise_and(cv2.bitwise_or(low, high), redder.astype(np.uint8) * 255)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _red_ring_kernel(radius):
    size = 2 * radius + 1
    yy, xx = np.ogrid[:size, :size]
    dist = np.hypot(yy - radius, xx - radius)
    kernel = np.zeros((size, size), np.float32)
    kernel[(dist >= radius * 0.62) & (dist <= radius * 1.12)] = 1.0
    total = float(kernel.sum())
    if total:
        kernel /= total
    return kernel


def _snap_row_to_red(row, red, kernel, radius, limit):
    """Pull each choice onto its own red ring, by a few pixels and never onto the next row."""
    snapped = []
    for x, y in row:
        xi, yi = int(round(x)), int(round(y))
        best = (0, 0, -1.0)
        base = 0.0
        for dy in range(-limit, limit + 1):
            for dx in range(-limit, limit + 1):
                y0 = yi + dy - radius
                x0 = xi + dx - radius
                patch = red[y0:y0 + kernel.shape[0], x0:x0 + kernel.shape[1]]
                if patch.shape != kernel.shape:
                    continue
                score = float((patch * kernel).sum())
                if dx == 0 and dy == 0:
                    base = score
                # The window edge is the neighbouring bubble. Leave that peak alone.
                if abs(dx) == limit or abs(dy) == limit:
                    continue
                if score > best[2]:
                    best = (dx, dy, score)
        dx, dy, score = best
        if score >= 0.12 and score >= base + 0.04:
            snapped.append((float(x) + dx, float(y) + dy))
        else:
            snapped.append((float(x), float(y)))
    return snapped


def _snap_rows_to_red_print(rows, image, pitch):
    red = _red_print_mask(image)
    if int(red.sum()) < 500:
        return rows
    radius = max(5, int(round(float(pitch) * 0.40)))
    limit = max(2, int(round(float(pitch) * 0.28)))
    kernel = _red_ring_kernel(radius)
    red_f = red.astype(np.float32) / 255.0
    return [_snap_row_to_red(row, red_f, kernel, radius, limit) for row in rows]


def _separate_crowded_rows(rows, template_rows, pitch):
    """Keep one question per printed row.

    A wrinkle can drag two grid rows onto the same bubble, so the red ring of
    one question sits inside the next. The row that has left its template
    place moves back until the gap is a real row again.
    """
    if len(rows) < 2:
        return rows
    min_gap = float(pitch) * 0.82

    def median_y(row):
        return float(np.median([point[1] for point in row]))

    ys = [median_y(row) for row in rows]
    template_ys = [median_y(row) for row in template_rows]
    for _ in range(len(rows)):
        changed = False
        for index in range(len(ys) - 1):
            gap = ys[index + 1] - ys[index]
            if gap >= min_gap:
                continue
            need = min_gap - gap
            drift_here = abs(ys[index] - template_ys[index])
            drift_next = abs(ys[index + 1] - template_ys[index + 1])
            if drift_next >= drift_here:
                ys[index + 1] += need
                if index + 2 < len(ys):
                    ys[index + 1] = min(ys[index + 1], ys[index + 2] - min_gap)
            else:
                ys[index] -= need
                if index > 0:
                    ys[index] = max(ys[index], ys[index - 1] + min_gap)
            changed = True
        if not changed:
            break
    separated = []
    for row, new_y in zip(rows, ys):
        dy = new_y - median_y(row)
        separated.append([(point[0], point[1] + dy) for point in row])
    return separated


def _snap_choice_rows(rows, circles, max_shift):
    """Center a row on its circles when every choice agrees. Never jump to the next row."""
    if len(circles) == 0:
        return rows
    snapped = []
    for row in rows:
        deltas = []
        for x, y in row:
            distance = np.hypot(circles[:, 0] - x, circles[:, 1] - y)
            nearest = int(np.argmin(distance))
            if distance[nearest] <= max_shift:
                deltas.append(circles[nearest] - np.array([x, y], np.float32))
        if len(deltas) >= max(2, (len(row) + 1) // 2):
            deltas = np.asarray(deltas, np.float32)
            med = np.median(deltas, axis=0)
            spread = float(np.median(np.hypot(deltas[:, 0] - med[0], deltas[:, 1] - med[1])))
            if spread <= 1.6 and 0.8 <= float(np.hypot(med[0], med[1])) <= max_shift:
                snapped.append([(point[0] + float(med[0]), point[1] + float(med[1])) for point in row])
                continue
        snapped.append(row)
    return snapped


def refine_layout_to_marks(layout, aligned_bgr):
    """Nudge a template grid onto the bubbles of a wrinkled photo.

    Pen fills place Part I, Part II, and the id grids row by row. Part III is
    mostly empty circles, so each question is translated onto those circles.
    Printed rings are red, so each Part I and Part II choice then recenters on that red.
    """
    blobs = _filled_mark_blobs(aligned_bgr)
    pitch = float(layout["pitch_y"])
    layout = copy.deepcopy(layout)
    template_part1 = copy.deepcopy(layout["part1"])
    template_part2 = copy.deepcopy(layout["part2"])
    if len(blobs) >= 8:
        for key in ("part1", "part2"):
            layout[key] = [_refine_choice_rows(block, blobs, pitch) for block in layout[key]]
        layout["sbd"] = _refine_choice_rows(layout["sbd"], blobs, pitch)
        layout["made"] = _refine_choice_rows(layout["made"], blobs, pitch)
    try:
        circles, _ = detect_bubble_centers(aligned_bgr, min_count=40)
    except ValueError:
        circles = np.zeros((0, 2), np.float32)
    if len(circles) >= 40:
        max_shift = min(6.0, pitch * 0.36)
        for key in ("part1", "part2"):
            layout[key] = [_snap_choice_rows(block, circles, max_shift) for block in layout[key]]
            layout[key] = [_snap_rows_to_red_print(block, aligned_bgr, pitch) for block in layout[key]]
        layout["sbd"] = _snap_choice_rows(layout["sbd"], circles, max_shift)
        layout["made"] = _snap_choice_rows(layout["made"], circles, max_shift)
        base_ys = [layout["part3_sign_y"], layout["part3_dot_y"], *layout["part3_digit_ys"]]
        columns_out = []
        question_ys = []
        for columns in layout["part3_columns"]:
            snapped_columns, snapped_ys = _snap_part3_question(columns, base_ys, circles, blobs, pitch)
            columns_out.append(snapped_columns)
            question_ys.append(snapped_ys)
        layout["part3_columns"] = columns_out
        layout["part3_question_ys"] = question_ys
    elif len(blobs) >= 8:
        base_ys = [layout["part3_sign_y"], layout["part3_dot_y"], *layout["part3_digit_ys"]]
        columns_out = []
        question_ys = []
        for columns in layout["part3_columns"]:
            rows = [[(float(x), float(y)) for x in columns] for y in base_ys]
            refined = _refine_choice_rows(rows, blobs, pitch)
            columns_out.append([refined[0][index][0] for index in range(len(columns))])
            question_ys.append([refined[row_index][0][1] for row_index in range(len(refined))])
        layout["part3_columns"] = columns_out
        layout["part3_question_ys"] = question_ys
    layout["part1"] = [
        _separate_crowded_rows(block, template, pitch)
        for block, template in zip(layout["part1"], template_part1)
    ]
    layout["part2"] = [
        _separate_crowded_rows(block, template, pitch)
        for block, template in zip(layout["part2"], template_part2)
    ]
    return layout


def get_full_answers(link_img, align=True, reference_path=DEFAULT_REFERENCE_IMAGE, debug_dir=None, return_aligned=False):
    img_path = resolve_path(link_img)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Input image not found: {img_path}")

    img = ensure_bgr(img)
    aligned_img = img
    if align:
        aligned_img, _, align_score, reference_path = normalize_answer_sheet(
            img, reference_path, debug_dir=debug_dir
        )
        print(f"Alignment score: {align_score:.3f}")
        print(f"Form template: {Path(reference_path).name}")
    else:
        align_score = 1.0

    layout = get_sheet_layout(reference_path)
    if align_score < 0.85:
        # The page is the right form, but wrinkles leave the printed grid a few
        # pixels off the pen fills. Follow the fills instead of rebuilding it.
        layout = refine_layout_to_marks(layout, aligned_img)
    answers_i, answers_ii, answers_iii, answers_sbd, answers_made, marked_points, shift, part3_cells = read_sheet(
        aligned_img, layout
    )
    print(answers_i)
    print(answers_ii)
    print(answers_iii)
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = draw_marked_bubbles(aligned_img, marked_points, layout["radius"])
        cv2.imwrite(str(debug_dir / "marked_bubbles.jpg"), overlay)
        print(f"Grid shift: ({shift[0]:.2f}, {shift[1]:.2f})")

    if return_aligned:
        return answers_i, answers_ii, answers_iii, answers_sbd, answers_made, aligned_img, layout, shift, part3_cells
    return answers_i, answers_ii, answers_iii, answers_sbd, answers_made


def load_answer_key(path):
    """Read a comma or semicolon CSV. Excel on a Vietnamese computer often uses semicolons."""
    path = resolve_path(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Section", "Question", "Sub-question", "Answer"}
    if not required.issubset(frame.columns):
        frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def grade_image(
    image_path,
    answer_key_path=DEFAULT_ANSWER_KEY,
    reference_path=DEFAULT_REFERENCE_IMAGE,
    output_path=None,
    aligned_output_path=None,
    debug_dir=None,
    align=True,
):
    part1, part2, part3, sbd, made, aligned_img, layout, shift, part3_cells = get_full_answers(
        image_path,
        align=align,
        reference_path=reference_path,
        debug_dir=debug_dir,
        return_aligned=True,
    )
    df = load_answer_key(answer_key_path)
    diemp1 = grade_part1(df, part1)
    diemp2 = grade_part2(df, part2)
    diemp3 = grade_part3(df, part3)
    sbd_str = "".join(val[0] for val in sbd.values())
    made_str = "".join(val[0] for val in made.values())
    import handwriting

    identity = handwriting.read_sheet_header(aligned_img, layout)

    result_img = annotate_result_image(
        aligned_img,
        layout,
        diemp1,
        diemp2,
        diemp3,
        sbd_str,
        made_str,
        shift=shift,
        part1=part1,
        part2=part2,
        part3=part3,
        answer_key=df,
        part3_cells=part3_cells,
    )

    if aligned_output_path is not None:
        aligned_output_path = resolve_path(aligned_output_path)
        aligned_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(aligned_output_path), aligned_img)

    if output_path is not None:
        output_path = resolve_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result_img)

    scores = {
        "part1": diemp1,
        "part2": diemp2,
        "part3": diemp3,
        "student_id": sbd_str,
        "exam_code": made_str,
        **identity,
    }
    return scores, result_img


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Autograde a scanned or photographed answer sheet.")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_REFERENCE_IMAGE), help="Path to the answer sheet image.")
    parser.add_argument("--answer-key", default=str(DEFAULT_ANSWER_KEY), help="Path to answer_key.csv.")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_IMAGE), help="Clean reference form for alignment.")
    parser.add_argument("--output", help="Path to save the graded, aligned result image.")
    parser.add_argument("--aligned-output", help="Path to save the aligned image before drawing scores.")
    parser.add_argument("--debug-dir", help="Directory for alignment and bubble debug images.")
    parser.add_argument("--no-align", action="store_true", help="Skip page detection and perspective alignment.")
    parser.add_argument("--show", action="store_true", help="Show the graded image with OpenCV after processing.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    image_path = resolve_path(args.image)
    output_path = resolve_path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"{image_path.stem}_graded.jpg"
    aligned_output_path = (
        resolve_path(args.aligned_output) if args.aligned_output else DEFAULT_OUTPUT_DIR / f"{image_path.stem}_aligned.jpg"
    )

    scores, result_img = grade_image(
        image_path,
        answer_key_path=args.answer_key,
        reference_path=args.reference,
        output_path=output_path,
        aligned_output_path=aligned_output_path,
        debug_dir=resolve_path(args.debug_dir) if args.debug_dir else None,
        align=not args.no_align,
    )

    print("Scores:", scores)
    print(f"Saved aligned image: {aligned_output_path}")
    print(f"Saved graded image: {output_path}")

    if args.show:
        cv2.imshow("Your score", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
