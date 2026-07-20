import argparse
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.append(str(PACKAGE_PARENT))

import pandas as pd
import cv2
import imutils
import numpy as np
from collections import defaultdict, OrderedDict
try:
    from AUTOGRADE.model.modelCNN import CNN_Model
except ModuleNotFoundError as exc:
    if exc.name != "AUTOGRADE":
        raise
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from model.modelCNN import CNN_Model

DEFAULT_REFERENCE_IMAGE = PROJECT_ROOT / "dataset" / "form2.png"
DEFAULT_ANSWER_KEY = PROJECT_ROOT / "answer_key.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
CROPPED_ROOT = DEFAULT_OUTPUT_DIR / "cropped_images" / "form"
MODEL_WEIGHT_PATH = PROJECT_ROOT / "model" / "weight.h5"
MIN_ALIGNMENT_SCORE = 0.70

_REFERENCE_CACHE = {}
_LAYOUT_CACHE = {}
_MODEL_CACHE = None


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
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def dark_mask(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )
    mask = cv2.bitwise_or(otsu, adaptive)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / float(area_a + area_b - inter)


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


def detect_corner_markers(img):
    img = ensure_bgr(img)
    h_img, w_img = img.shape[:2]
    candidates = sorted(find_marker_candidates(img), key=lambda item: item["score"], reverse=True)[:40]
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


def normalize_answer_sheet(img, reference_path=DEFAULT_REFERENCE_IMAGE, debug_dir=None):
    img = ensure_bgr(img)
    reference_img, reference_markers = load_reference(reference_path)
    h_ref, w_ref = reference_img.shape[:2]
    source_markers = detect_corner_markers(img)

    best_warped = None
    best_score = -1
    best_matrix = None
    for shift in range(4):
        shifted_source = np.roll(source_markers, -shift, axis=0)
        matrix = cv2.getPerspectiveTransform(shifted_source, reference_markers)
        warped = cv2.warpPerspective(
            img,
            matrix,
            (w_ref, h_ref),
            flags=cv2.INTER_LINEAR,
            borderValue=(255, 255, 255),
        )
        score = template_similarity(warped, reference_img)
        if score > best_score:
            best_score = score
            best_warped = warped
            best_matrix = matrix

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "aligned_sheet.jpg"), best_warped)

    if best_score < MIN_ALIGNMENT_SCORE:
        raise ValueError(
            f"Alignment confidence too low ({best_score:.3f}). "
            "Make sure all four black corner markers are visible and not cropped."
        )

    return best_warped, best_matrix, best_score


def select_unique_boxes(boxes, count):
    unique = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(box_iou(box, other) < 0.4 for other in unique):
            unique.append(box)
        if len(unique) == count:
            break
    return sorted(unique, key=lambda item: item[0])


def detect_layout_boxes(img):
    img = ensure_bgr(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    img_canny = cv2.Canny(blurred, 10, 50)
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    merged_boxes = merge_overlapping_contours(cnts)

    h_img, w_img = gray_img.shape[:2]
    boxes = [(x, y, w, h) for x, y, w, h in merged_boxes if w * h > 1000]

    def center(box):
        x, y, w, h = box
        return (x + w / 2.0) / w_img, (y + h / 2.0) / h_img

    sbd_candidates = []
    p1_candidates = []
    p2_candidates = []
    p3_candidates = []
    for box in boxes:
        x, y, w, h = box
        cx, cy = center(box)
        rel_w = w / w_img
        rel_h = h / h_img

        if cy < 0.32 and rel_w > 0.45 and rel_h > 0.1:
            sbd_candidates.append(box)
        elif 0.30 <= cy <= 0.56 and 0.12 <= rel_w <= 0.28 and rel_h > 0.12:
            p1_candidates.append(box)
        elif 0.52 <= cy <= 0.70 and 0.12 <= rel_w <= 0.28 and 0.06 <= rel_h <= 0.18:
            p2_candidates.append(box)
        elif cy > 0.65 and rel_w > 0.55 and rel_h > 0.16:
            p3_candidates.append(box)

    layout = {
        "sbd": max(sbd_candidates, key=lambda item: item[2] * item[3], default=None),
        "part1": select_unique_boxes(p1_candidates, 4),
        "part2": select_unique_boxes(p2_candidates, 4),
        "part3": max(p3_candidates, key=lambda item: item[2] * item[3], default=None),
    }

    missing = []
    if layout["sbd"] is None:
        missing.append("student id/code block")
    if len(layout["part1"]) != 4:
        missing.append("part I blocks")
    if len(layout["part2"]) != 4:
        missing.append("part II blocks")
    if layout["part3"] is None:
        missing.append("part III block")
    if missing:
        raise ValueError("Could not detect layout: " + ", ".join(missing))

    return layout, cnts, merged_boxes


def get_reference_layout(reference_path=DEFAULT_REFERENCE_IMAGE):
    reference_path = resolve_path(reference_path)
    cache_key = str(reference_path)
    if cache_key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[cache_key]

    reference_img, _ = load_reference(reference_path)
    layout, _, _ = detect_layout_boxes(reference_img)
    _LAYOUT_CACHE[cache_key] = layout
    return layout


def crop_by_box(gray_img, box):
    x, y, w, h = box
    return gray_img[y:y + h, x:x + w]


def save_crop(relative_path, crop):
    path = CROPPED_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)


def ratio_index(size, ratio):
    return int(round(size * ratio))


def detect_info_table_boxes(pic):
    gray = pic if len(pic.shape) == 2 else cv2.cvtColor(ensure_bgr(pic), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 10, 50)
    cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    h_img, w_img = gray.shape[:2]
    candidates = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        rel_x = x / float(w_img)
        rel_w = w / float(w_img)
        rel_h = h / float(h_img)

        if y <= ratio_index(h_img, 0.08) and rel_x >= 0.55 and rel_h >= 0.78 and 0.1 <= rel_w <= 0.26:
            candidates.append((x, y, w, h))

    tables = select_unique_boxes(candidates, 2)
    if len(tables) != 2:
        return None
    return tables


def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def merge_overlapping_contours(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = []

    for box in bounding_boxes:
        x, y, w, h = box
        merged = False
        for idx, merged_box in enumerate(merged_boxes):
            x_m, y_m, w_m, h_m = merged_box
            if not (x > x_m + w_m or x + w < x_m or y > y_m + h_m or y + h < y_m):  # Check overlap
                # Merge boxes
                new_x = min(x, x_m)
                new_y = min(y, y_m)
                new_w = max(x + w, x_m + w_m) - new_x
                new_h = max(y + h, y_m + h_m) - new_y
                merged_boxes[idx] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        if not merged:
            merged_boxes.append((x, y, w, h))

    return merged_boxes

def crop_image(img, reference_path=DEFAULT_REFERENCE_IMAGE, debug_dir=None):
    img = ensure_bgr(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    img_canny = cv2.Canny(blurred, 10, 50)

    # Find contours
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    merged_boxes = merge_overlapping_contours(cnts)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        output = img.copy()
        cv2.drawContours(output, cnts, -1, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "all_contours.jpg"), output)
        for (x, y, w, h) in merged_boxes:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "highlighted_contours.jpg"), output)

    layout = get_reference_layout(reference_path)

    ans_blocks_pI = [crop_by_box(gray_img, box) for box in layout["part1"]]
    ans_blocks_pII = [crop_by_box(gray_img, box) for box in layout["part2"]]
    ans_blocks_pIII = [crop_by_box(gray_img, layout["part3"])]
    ans_blocks_sbd = [crop_by_box(gray_img, layout["sbd"])]

    for idx, box in enumerate(layout["part1"]):
        save_crop(Path("I") / f"PHANI_p_{idx}.jpg", img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
    for idx, box in enumerate(layout["part2"]):
        save_crop(Path("II") / f"PHANII_block_{idx}.jpg", img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
    box = layout["part3"]
    save_crop(Path("III") / "PHANIII.jpg", img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
    box = layout["sbd"]
    save_crop(Path("sbd") / "sbd.jpg", img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])

    return ans_blocks_pI, ans_blocks_pII, ans_blocks_pIII, ans_blocks_sbd
    
def remove_trash_images():
    for folder in CROPPED_ROOT.iterdir() if CROPPED_ROOT.exists() else []:
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
        
def process_ans_blocks(ans_blocks, part=1):
    list_answers = []
    if part == 1:
        x = 0   
        for i, pic in enumerate(ans_blocks):
            x = 0
            header_h = ratio_index(pic.shape[0], 0.11)
            new_pic = pic[header_h:, :]
            h = new_pic.shape[0]
            w = new_pic.shape[1]
            h_new = h // 10
            for j in range(10):
                cropped = new_pic[x: x + h_new, :]
                x += h_new
                list_answers.append(cropped)
        #         cv2.imshow(f"pic_{i}_{j}", cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    elif part == 2:
        x = 0
        for i, pic in enumerate(ans_blocks):
            x = 0
            header_h = ratio_index(pic.shape[0], 0.29)
            new_pic = pic[header_h:, :]
            h = new_pic.shape[0]
            w = new_pic.shape[1]
            h_new = h // 4
            for j in range(4):
                cropped = new_pic[x: x + h_new, :]
                x += h_new
                list_answers.append(cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    elif part == 3:
        x = 0
        y = 0
        # dem = 0
        for i, pic in enumerate(ans_blocks):
            w = pic.shape[1]
            new_w = w // 6
            x = 0
            y = 0
            for j in range(6):
                header_h = ratio_index(pic.shape[0], 0.14)
                new_pic = pic[header_h:, y: y + new_w]
                label_w = ratio_index(new_pic.shape[1], 0.24)
                new_pic = new_pic[:, label_w:]
                w_new = new_pic.shape[1] // 4
                y2 = 0
                for k in range(4):
                    cropped = new_pic[:, y2:y2 + w_new]
                    y2 += w_new
                    list_answers.append(cropped)
                    # cv2.imshow(f"pic_{i}_{j}_{k}", cropped)
                y += new_w
    elif part == 4:
        pic = ans_blocks[0]
        info_table_boxes = detect_info_table_boxes(pic)
        if info_table_boxes is not None:
            sbd_table = crop_by_box(pic, info_table_boxes[0])
            top = ratio_index(sbd_table.shape[0], 0.13)
            left = ratio_index(sbd_table.shape[1], 0.16)
            right = sbd_table.shape[1] - max(1, ratio_index(sbd_table.shape[1], 0.02))
            pic_sbd = sbd_table[top:, left:right]
        else:
            top = ratio_index(pic.shape[0], 0.13)
            right_table_x = ratio_index(pic.shape[1], 0.64)
            new_pic = pic[top:, right_table_x:]
            sbd_left = ratio_index(new_pic.shape[1], 0.10)
            sbd_right = new_pic.shape[1] - ratio_index(new_pic.shape[1], 0.39)
            pic_sbd = new_pic[:, sbd_left:sbd_right]
        w_new = pic_sbd.shape[1] // 6
        y = 0
        for i in range(6):
            cropped = pic_sbd[:, y: y + w_new]
            y += w_new
            list_answers.append(cropped)
        #     cv2.imshow(f"pic_{i}", cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # list_answers.append(pic_sbd)
    else:
        pic = ans_blocks[0]
        info_table_boxes = detect_info_table_boxes(pic)
        if info_table_boxes is not None:
            made_table = crop_by_box(pic, info_table_boxes[1])
            top = ratio_index(made_table.shape[0], 0.13)
            left = ratio_index(made_table.shape[1], 0.27)
            right = made_table.shape[1] - max(1, ratio_index(made_table.shape[1], 0.02))
            made_pic = made_table[top:, left:right]
        else:
            top = ratio_index(pic.shape[0], 0.13)
            right_table_x = ratio_index(pic.shape[1], 0.64)
            new_pic = pic[top:, right_table_x:]
            made_left = ratio_index(new_pic.shape[1], 0.71)
            made_right = new_pic.shape[1] - max(1, ratio_index(new_pic.shape[1], 0.01))
            made_pic = new_pic[:, made_left:made_right]
        w_new = made_pic.shape[1] // 3
        y = 0
        for i in range(3):
            cropped = made_pic[: , y: y + w_new]
            y += w_new
            list_answers.append(cropped)
        #     cv2.imshow(f"pic_{i}", cropped)
        # # cv2.imshow("pic_sbd", pic_sbd)
        # # cv2.imshow("made_pic", made_pic)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # list_answers.append(made_pic)
    return list_answers
def process_list_ans(list_answers, part = 1):
    if part == 1:
        buble_choice = []
        for i, pic in enumerate(list_answers):
            label_w = ratio_index(pic.shape[1], 0.18)
            new_pic = pic[:, label_w:]
            new_width = new_pic.shape[1] // 4
            y = 0
            for j in range(4):
                choice = new_pic[ : , y: y + new_width]
                choice = cv2.threshold(choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                choice = cv2.resize(choice, (28, 28), interpolation= cv2.INTER_AREA)
                choice = choice.reshape((28, 28, 1))
                # cv2.imshow(f"pic_{i}_{j}", choice)
                y += new_width
                buble_choice.append(choice)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     # cv2.imshow(f"pic_{i}", buble_choice)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    elif part == 2:
        buble_choice = []
        for i, pic in enumerate(list_answers):
            label_w = ratio_index(pic.shape[1], 0.18)
            new_pic = pic[:, label_w:]
            # cv2.imshow(f"pic_{i}", new_pic)
            new_width = new_pic.shape[1] // 4
            y = 0
            for j in range(4):
                choice = new_pic[ : , y: y + new_width]
                choice = cv2.threshold(choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                choice = cv2.resize(choice, (28, 28), interpolation= cv2.INTER_AREA)
                choice = choice.reshape((28, 28, 1))
                # cv2.imshow(f"pic_{i}_{j}", choice)
                y += new_width
                buble_choice.append(choice)
    elif part == 3:
        buble_choice = []
        for i, pic in enumerate(list_answers):
            new_pic = pic
            # cv2.imshow(f"pic_{i}", new_pic)
            new_height = new_pic.shape[0] // 12
            x = 0
            for j in range(12):
                choice = new_pic[ x : x + new_height, :]
                choice = cv2.threshold(choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                choice = cv2.resize(choice, (28, 28), interpolation= cv2.INTER_AREA)
                choice = choice.reshape((28, 28, 1))
                # cv2.imshow(f"pic_{i}_{j}", choice)
                x += new_height
                buble_choice.append(choice)
        #     break
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        buble_choice = []
        for i, pic in enumerate(list_answers):
            new_pic = pic
            # cv2.imshow(f"pic_{i}", new_pic)
            new_height = new_pic.shape[0] // 10
            x = 0
            for j in range(10):
                choice = new_pic[ x : x + new_height, :]
                choice = cv2.threshold(choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                choice = cv2.resize(choice, (28, 28), interpolation= cv2.INTER_AREA)
                choice = choice.reshape((28, 28, 1))
                # cv2.imshow(f"pic_{i}_{j}", choice)
                x += new_height
                buble_choice.append(choice)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # break
    return buble_choice
def map_answer_p1(idx):
    if idx % 4 == 0:
        answer_circle = 'A'
    elif idx % 4 == 1:
        answer_circle = 'B'
    elif idx % 4 == 2:
        answer_circle = 'C'
    else:
        answer_circle = 'D'
    return answer_circle

def map_answer_p2(idx):
    if idx %2 == 0:
        answer_circle = True
    else:
        answer_circle = False
    return answer_circle

def map_answer_p3(idx):
    mapping_answer_circle = "-.0123456789"
    return mapping_answer_circle[idx%12]
def map_answer_sbd_and_made(idx):
    mapping_answer_circle = "0123456789"
    return mapping_answer_circle[idx%10]


def get_choice_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = CNN_Model(str(MODEL_WEIGHT_PATH)).build_model(rt=True)
    return _MODEL_CACHE


def get_answers(list_answers, part = 1):
    if part == 1:
        results = defaultdict(list)
        model = get_choice_model()
        list_answers = np.array(list_answers)
        # print("model = " , model)
        scores = model.predict_on_batch(list_answers / 255.0)
        for idx, score in enumerate(scores):
            question = idx // 4

            # score [unchoiced_cf, choiced_cf]
            if score[1] > 0.9:  # choiced confidence score > 0.9
                chosed_answer = map_answer_p1(idx)
                results[question + 1].append(chosed_answer)
        return results
    elif part == 2:
        results = defaultdict(list)
        model = get_choice_model()
        list_answers = np.array(list_answers)
        scores = model.predict_on_batch(list_answers / 255.0)
        
        for idx, score in enumerate(scores):
            question = idx // 4
            if score[1] > 0.9:  # Confidence score > 0.9
                chosed_answer = map_answer_p2(idx)
                results[question + 1].append(chosed_answer)

        odd_questions, even_questions = [], []
        for key, val in results.items():
            odd_questions.append(val[0])
            even_questions.append(val[1])

        new_results = defaultdict(lambda: defaultdict(list))

        number, idx = 1, 0
        for i in range(len(odd_questions)):
            idx += 1
            if idx == 5:
                idx = 1
                number += 2
            type_question = ['a', 'b', 'c', 'd'][idx - 1]
            new_results[number][type_question].append(odd_questions[i])

        number, idx = 2, 0
        for i in range(len(even_questions)):
            idx += 1
            if idx == 5:
                idx = 1
                number += 2
            type_question = ['a', 'b', 'c', 'd'][idx - 1]
            new_results[number][type_question].append(even_questions[i])

        sorted_results = OrderedDict(sorted(new_results.items(), key=lambda x: x[0]))

        return sorted_results
    elif part == 3:
        results = defaultdict(list)
        model = get_choice_model()
        list_answers = np.array(list_answers)
        # print("model = " , model)
        scores = model.predict_on_batch(list_answers / 255.0)
        for idx, score in enumerate(scores):
            question = idx // 12

            # score [unchoiced_cf, choiced_cf]
            if score[1] > 0.9:  # choiced confidence score > 0.9
                chosed_answer = map_answer_p3(idx)
                results[question + 1].append(chosed_answer)
        start, end = 1, 5
        new_results = defaultdict(list)
        ans = ""
        for i in range(1, 7):
            ans=""
            for j in range(start, end):
                ans = ans + results[j][0]
            new_results[i].append(ans)
            start += 4
            end += 4
        return new_results
    elif part == 4:
        results = defaultdict(list)
        list_answers = np.array(list_answers)
        for question, start in enumerate(range(0, len(list_answers), 10), start=1):
            choices = list_answers[start:start + 10]
            if len(choices) < 10:
                continue
            fill_scores = [np.count_nonzero(choice) / float(choice.size) for choice in choices]
            chosed_idx = int(np.argmax(fill_scores))
            results[question].append(map_answer_sbd_and_made(chosed_idx))
        # start, end = 1, 7
        # new_results = defaultdict(list)
        # ans = ""
        # for i in range(1, 2):
        #     ans=""
        #     for j in range(start, end):
        #         ans = ans + results[j][0]
        #     new_results[i].append(ans)
        #     start += 4
        #     end += 4
        return results

def get_full_answers(link_img, align=True, reference_path=DEFAULT_REFERENCE_IMAGE, debug_dir=None, return_aligned=False):
    img_path = resolve_path(link_img)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Input image not found: {img_path}")

    img = ensure_bgr(img)
    aligned_img = img
    if align:
        aligned_img, _, align_score = normalize_answer_sheet(img, reference_path, debug_dir=debug_dir)
        print(f"Alignment score: {align_score:.3f}")

    cropped_blocksI, cropped_blocksII, cropped_blocksIII, cropped_blocks_sbd = crop_image(
        aligned_img,
        reference_path=reference_path,
        debug_dir=debug_dir,
    )
    list_answers_pI = process_ans_blocks(cropped_blocksI, 1)
    list_answers_pII = process_ans_blocks(cropped_blocksII, 2)
    list_answers_pIII = process_ans_blocks(cropped_blocksIII, 3)
    list_answers_sbd = process_ans_blocks(cropped_blocks_sbd, 4)
    list_answers_made = process_ans_blocks(cropped_blocks_sbd, 5)
    
    list_answers_pI =process_list_ans(list_answers_pI, 1)
    list_answers_pII = process_list_ans(list_answers_pII, 2)
    list_answers_pIII = process_list_ans(list_answers_pIII, 3)
    list_answers_sbd = process_list_ans(list_answers_sbd, 4)
    list_answers_made = process_list_ans(list_answers_made, 4)
    
    answersI = get_answers(list_answers_pI, 1)
    answersII = get_answers(list_answers_pII, 2)
    answersIII = get_answers(list_answers_pIII, 3)
    answerssbd = get_answers(list_answers_sbd, 4)
    answersmade = get_answers(list_answers_made, 4)
    print(answersI)
    print(answersII)
    print(answersIII)
    # print(answerssbd)
    # print(answersmade)
    if return_aligned:
        return answersI, answersII, answersIII, answerssbd, answersmade, aligned_img
    return answersI, answersII, answersIII, answerssbd, answersmade


def grade_part1(df, student_answer_dict):
    score = 0
    answer_dict = defaultdict(list)
    for _, row in df.iterrows():
        question_number = int(row["Question"])
        part = row["Section"]
        if 1 <= question_number <= 40 and part =="I":
            answer_dict[question_number].append(row["Answer"])
    for key, value in answer_dict.items():
        if value == student_answer_dict[key]:
            score+=1
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


def annotate_result_image(img, diemp1, diemp2, diemp3, sbd_str, made_str):
    img = img.copy()
    cv2.putText(img, diemp1, (400, 280), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, diemp2, (400, 490), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, diemp3, (400, 610), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sbd_str, (420, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, made_str, (540, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def grade_image(
    image_path,
    answer_key_path=DEFAULT_ANSWER_KEY,
    reference_path=DEFAULT_REFERENCE_IMAGE,
    output_path=None,
    aligned_output_path=None,
    debug_dir=None,
    align=True,
):
    remove_trash_images()
    p1, p2, p3, sbd, made, aligned_img = get_full_answers(
        image_path,
        align=align,
        reference_path=reference_path,
        debug_dir=debug_dir,
        return_aligned=True,
    )
    df = pd.read_csv(resolve_path(answer_key_path))
    diemp1 = grade_part1(df, p1)
    diemp2 = grade_part2(df, p2)
    diemp3 = grade_part3(df, p3)
    sbd_str = ''.join([''.join(val) for val in sbd.values()])
    made_str = ''.join([''.join(val) for val in made.values()])

    result_img = annotate_result_image(aligned_img, diemp1, diemp2, diemp3, sbd_str, made_str)

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
    }
    return scores, result_img


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Autograde a scanned or photographed answer sheet.")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_REFERENCE_IMAGE), help="Path to the answer sheet image.")
    parser.add_argument("--answer-key", default=str(DEFAULT_ANSWER_KEY), help="Path to answer_key.csv.")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_IMAGE), help="Clean reference form for alignment/layout.")
    parser.add_argument("--output", help="Path to save the graded, aligned result image.")
    parser.add_argument("--aligned-output", help="Path to save the aligned image before drawing scores.")
    parser.add_argument("--debug-dir", help="Directory for contour/alignment debug images.")
    parser.add_argument("--no-align", action="store_true", help="Skip marker detection and perspective alignment.")
    parser.add_argument("--show", action="store_true", help="Show the graded image with OpenCV after processing.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    image_path = resolve_path(args.image)
    output_path = resolve_path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"{image_path.stem}_graded.jpg"
    aligned_output_path = resolve_path(args.aligned_output) if args.aligned_output else DEFAULT_OUTPUT_DIR / f"{image_path.stem}_aligned.jpg"

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


if __name__ == '__main__':
    main()
