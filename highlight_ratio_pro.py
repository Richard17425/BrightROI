import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --------------------------
# Args
# --------------------------
parser = argparse.ArgumentParser(
    description="Compute highlight-pixel ratio inside a user-selected ROI (thresholding + interactive ROI + eraser)"
)
parser.add_argument("image", type=str, help="Path to input image")
parser.add_argument("--thresh", type=int, default=200, help="Threshold value (default: 200)")
parser.add_argument("--invert", action="store_true", help="If set, count dark pixels (<= thresh) instead of bright ones")
parser.add_argument("--init-width", type=int, default=1200, help="Initial window width (default: 1200)")
parser.add_argument("--init-height", type=int, default=800, help="Initial window height (default: 800)")
args = parser.parse_args()

# --------------------------
# Load image & preprocess
# --------------------------
img = cv2.imread(args.image, cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit(f"Failed to read image: {args.image}")
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholded image in IMAGE coordinates (for counting)
# > thresh => 255 (bright), else 0; if invert, <= thresh => 255
if args.invert:
    _, bin_img = cv2.threshold(gray, args.thresh, 255, cv2.THRESH_BINARY_INV)
else:
    _, bin_img = cv2.threshold(gray, args.thresh, 255, cv2.THRESH_BINARY)

# --------------------------
# Window & view management
# --------------------------
WIN = "ROI (m: toggle poly/rect | wheel: zoom | mid/right drag: pan | e: eraser | Enter: calc | r: reset | z: undo | q: quit)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WIN, args.init_width, args.init_height)

# View state (image -> display transform)
scale = min(1.0, min(args.init_width / w, args.init_height / h))  # initial fit
min_scale, max_scale = 0.1, 10.0
ox, oy = 0.0, 0.0  # top-left offset (display px)
dragging = False
last_mouse = (0, 0)

# ROI state
roi_mode = "poly"   # 'poly' or 'rect'
points_img = []     # polygon points in IMAGE coords
rect_start_img = None
rect_end_img = None
rect_dragging = False

# Eraser state (works AFTER you have a ROI)
eraser_mode = False
brush = 20  # radius in IMAGE pixels
erase_mask = np.zeros((h, w), dtype=np.uint8)  # 255 where user erased
erase_strokes = []  # stack of stroke binary masks for undo

# --------------------------
# Helpers
# --------------------------
def disp_to_img(pt_disp):
    """Display (window) -> image coordinates (int)."""
    x_disp, y_disp = pt_disp
    x_img = (x_disp - ox) / scale
    y_img = (y_disp - oy) / scale
    return int(round(x_img)), int(round(y_img))

def img_to_disp(pt_img):
    """Image -> display coordinates (int)."""
    x_img, y_img = pt_img
    x_disp = x_img * scale + ox
    y_disp = y_img * scale + oy
    return int(round(x_disp)), int(round(y_disp))

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def fit_center(view_w, view_h):
    """Center the image in the view for current scale."""
    global ox, oy
    scaled_w = w * scale
    scaled_h = h * scale
    ox = (view_w - scaled_w) * 0.5
    oy = (view_h - scaled_h) * 0.5

def get_view_size():
    """Try to query the current drawable area; fallback to initial size."""
    try:
        _, _, ww, hh = cv2.getWindowImageRect(WIN)
        return max(200, ww), max(200, hh)
    except Exception:
        return args.init_width, args.init_height

def render_base(view_w, view_h):
    """Return display image with original image scaled & positioned."""
    canvas = np.zeros((int(view_h), int(view_w), 3), dtype=np.uint8)
    scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    Hs, Ws = scaled.shape[:2]
    x0 = int(round(ox)); y0 = int(round(oy))
    x1 = x0 + Ws; y1 = y0 + Hs
    cx0 = max(0, x0); cy0 = max(0, y0)
    cx1 = min(view_w, x1); cy1 = min(view_h, y1)
    if cx1 > cx0 and cy1 > cy0:
        sx0 = cx0 - x0; sy0 = cy0 - y0
        sx1 = sx0 + (cx1 - cx0); sy1 = sy0 + (cy1 - cy0)
        canvas[cy0:cy1, cx0:cx1] = scaled[sy0:sy1, sx0:sx1]
    return canvas

def draw_overlays(disp_img):
    """Draw ROI overlays and eraser overlay onto display image."""
    # ROI overlays
    if roi_mode == "poly":
        pts_disp = [img_to_disp(p) for p in points_img]
        for i, p in enumerate(pts_disp):
            cv2.circle(disp_img, p, 3, (0, 255, 255), -1)
            if i > 0:
                cv2.line(disp_img, pts_disp[i - 1], pts_disp[i], (0, 255, 255), 1)
        if len(pts_disp) >= 3:
            cv2.polylines(disp_img, [np.array(pts_disp, dtype=np.int32)], True, (0, 255, 255), 1)
    else:
        if rect_start_img is not None and rect_end_img is not None:
            p0 = img_to_disp(rect_start_img)
            p1 = img_to_disp(rect_end_img)
            cv2.rectangle(disp_img, p0, p1, (0, 255, 255), 2)

    # Eraser overlay (red translucent)
    if erase_mask.any():
        # Build a display-sized overlay of erased regions
        erased = cv2.resize(erase_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        Hs, Ws = erased.shape[:2]
        x0 = int(round(ox)); y0 = int(round(oy))
        overlay = disp_img.copy()
        # Compose within bounds
        cx0 = max(0, x0); cy0 = max(0, y0)
        cx1 = min(disp_img.shape[1], x0 + Ws); cy1 = min(disp_img.shape[0], y0 + Hs)
        if cx1 > cx0 and cy1 > cy0:
            sx0 = cx0 - x0; sy0 = cy0 - y0
            sx1 = sx0 + (cx1 - cx0); sy1 = sy0 + (cy1 - cy0)
            region = (erased[sy0:sy1, sx0:sx1] == 255)
            overlay[cy0:cy1, cx0:cx1][region] = (0, 0, 255)  # red
            disp_img = cv2.addWeighted(overlay, 0.35, disp_img, 0.65, 0)
    return disp_img

def draw_hud(disp_img, view_w, view_h):
    """HUD text."""
    mode_txt = f"mode={roi_mode} | thresh={args.thresh} | scale={scale:.2f}x | eraser={'ON' if eraser_mode else 'off'} | brush={brush}px"
    hint_txt = "wheel: zoom | mid/right drag: pan | m: ROI | e: eraser | [ ]: brush | u: undo | z: undo point | r: reset | Enter: calc | q: quit"
    cv2.putText(disp_img, mode_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 220, 20), 2, cv2.LINE_AA)
    cv2.putText(disp_img, hint_txt, (10, view_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return disp_img

def show_fit(win_name, image, max_w, max_h):
    """Fit image into (max_w, max_h) with preserved aspect ratio; resizable window."""
    ih, iw = image.shape[:2]
    sc = min(max_w / iw, max_h / ih, 1.0)
    disp = cv2.resize(image, (int(iw * sc), int(ih * sc)), interpolation=cv2.INTER_AREA) if sc < 1.0 else image
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(win_name, disp)
    cv2.resizeWindow(win_name, disp.shape[1], disp.shape[0])

def colorize_binary(single_channel):
    """Make single-channel images easier to see (binary->green/gray, grayscale->JET)."""
    u = np.unique(single_channel)
    if single_channel.dtype == np.uint8 and (np.array_equal(u, [0]) or np.array_equal(u, [255]) or np.array_equal(u, [0, 255])):
        vis = np.dstack([single_channel*0, single_channel, single_channel*0]).copy()
        vis[single_channel == 0] = (40, 40, 40)
        return vis
    else:
        return cv2.applyColorMap(single_channel, cv2.COLORMAP_JET)

def build_roi_mask():
    """Return mask (uint8 0/255) in IMAGE coords for current ROI; None if not ready."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if roi_mode == "poly":
        if len(points_img) < 3: 
            return None
        poly = np.array(points_img, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [poly], 255)  # polygon area = 255  :contentReference[oaicite:1]{index=1}
    else:
        if rect_start_img is None or rect_end_img is None:
            return None
        x0, y0 = rect_start_img; x1, y1 = rect_end_img
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))
        x_min = clamp(x_min, 0, w - 1); x_max = clamp(x_max, 0, w - 1)
        y_min = clamp(y_min, 0, h - 1); y_max = clamp(y_max, 0, h - 1)
        if x_max <= x_min or y_max <= y_min: 
            return None
        mask[y_min:y_max+1, x_min:x_max+1] = 255
    return mask

# --------------------------
# Mouse callback
# --------------------------
def on_mouse(event, x, y, flags, param):
    """Handle zoom/pan/ROI drawing and eraser painting."""
    global dragging, last_mouse, scale, ox, oy
    global rect_dragging, rect_start_img, rect_end_img, points_img
    global eraser_mode, brush, erase_mask, erase_strokes

    # Zoom around cursor
    if event == cv2.EVENT_MOUSEWHEEL:  # get delta for wheel and h-wheel  :contentReference[oaicite:2]{index=2}
        delta = cv2.getMouseWheelDelta(flags)
        if delta != 0:
            factor = 1.25 if delta > 0 else 0.8
            new_scale = clamp(scale * factor, min_scale, max_scale)
            if new_scale != scale:
                # anchor: keep image point under cursor fixed
                img_x_before, img_y_before = disp_to_img((x, y))
                scale = new_scale
                ox = x - img_x_before * scale
                oy = y - img_y_before * scale
        return

    # Middle/right button pan
    if event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        dragging = True
        last_mouse = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - last_mouse[0]; dy = y - last_mouse[1]
        ox += dx; oy += dy
        last_mouse = (x, y)
    elif event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
        dragging = False

    # Eraser painting (left-drag) – only if ROI exists
    roi_ready = build_roi_mask() is not None
    if eraser_mode and roi_ready:
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)):
            cx, cy = disp_to_img((x, y))
            if 0 <= cx < w and 0 <= cy < h:
                # paint a circle on a temporary stroke mask
                stroke = np.zeros_like(erase_mask)
                cv2.circle(stroke, (cx, cy), brush, 255, -1)  # draw disk  :contentReference[oaicite:3]{index=3}
                # Only allow erase inside current ROI
                roi = build_roi_mask()
                stroke = cv2.bitwise_and(stroke, roi)         # limit by ROI  :contentReference[oaicite:4]{index=4}
                erase_mask = cv2.bitwise_or(erase_mask, stroke)
                erase_strokes.append(stroke)
        return

    # ROI interactions with left button (when not in eraser mode)
    if not eraser_mode:
        if roi_mode == "poly":
            if event == cv2.EVENT_LBUTTONDOWN:
                points_img.append(disp_to_img((x, y)))
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                rect_dragging = True
                rect_start_img = disp_to_img((x, y))
                rect_end_img = rect_start_img
            elif event == cv2.EVENT_MOUSEMOVE and rect_dragging:
                rect_end_img = disp_to_img((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                rect_dragging = False
                rect_end_img = disp_to_img((x, y))

cv2.setMouseCallback(WIN, on_mouse)  # register mouse callback  :contentReference[oaicite:5]{index=5}

# Fit image initially
vw, vh = get_view_size()
fit_center(vw, vh)

# --------------------------
# Main loop
# --------------------------
while True:
    vw, vh = get_view_size()
    disp = render_base(vw, vh)
    disp = draw_overlays(disp)
    disp = draw_hud(disp, vw, vh)
    cv2.imshow(WIN, disp)

    key = cv2.waitKey(15) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        raise SystemExit(0)
    elif key == ord('m'):
        # Toggle ROI mode and clear shapes
        roi_mode = "rect" if roi_mode == "poly" else "poly"
        points_img = []
        rect_start_img = rect_end_img = None
        erase_mask[:] = 0
        erase_strokes.clear()
    elif key == ord('z'):
        if roi_mode == "poly" and points_img:
            points_img.pop()
        elif roi_mode == "rect":
            rect_start_img = rect_end_img = None
    elif key == ord('r'):
        points_img = []
        rect_start_img = rect_end_img = None
        erase_mask[:] = 0
        erase_strokes.clear()
    elif key == ord('e'):
        # Toggle eraser only if ROI exists
        eraser_mode = not eraser_mode if build_roi_mask() is not None else False
    elif key == ord('u'):
        # Undo last stroke
        if erase_strokes:
            last = erase_strokes.pop()
            erase_mask = cv2.bitwise_and(erase_mask, cv2.bitwise_not(last))
    elif key == ord('['):
        brush = max(1, brush - 2)
    elif key == ord(']'):
        brush = min(256, brush + 2)
    elif key in (13, 10):  # Enter -> compute with current ROI & erase_mask
        roi = build_roi_mask()
        if roi is None:
            print("Select a ROI first (polygon with >=3 points, or a rectangle).")
            continue

        # Effective ROI = ROI minus erased regions (inside ROI)
        effective_roi = cv2.bitwise_and(roi, cv2.bitwise_not(erase_mask))

        roi_area = cv2.countNonZero(effective_roi)
        if roi_area == 0:
            print("Effective ROI area is zero (maybe fully erased). Please reselect/erase less.")
            continue

        bin_in_roi = cv2.bitwise_and(bin_img, bin_img, mask=effective_roi)
        highlight_pixels = cv2.countNonZero(bin_in_roi)
        ratio = highlight_pixels / float(roi_area)

        # Print summary
        print("========== Result ==========")
        print(f"Image: {os.path.basename(args.image)}")
        print(f"Mode: {'dark (<= thresh)' if args.invert else 'bright (> thresh)'}")
        print(f"Threshold: {args.thresh}")
        print(f"ROI pixels (effective): {roi_area}")
        print(f"{'Dark' if args.invert else 'Highlight'} pixels (effective): {highlight_pixels}")
        print(f"Ratio: {ratio:.4%}")

        # Visualize result overlay (image space)
        vis = img.copy()
        # Draw ROI outline
        if roi_mode == "poly":
            poly = np.array(points_img, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
        else:
            x0, y0 = rect_start_img; x1, y1 = rect_end_img
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Paint counted highlights in green; paint erased regions in red overlay
        overlay = vis.copy()
        overlay[bin_in_roi == 255] = (0, 255, 0)
        overlay[erase_mask == 255] = (0, 0, 255)
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        cv2.putText(vis, f"ratio={ratio:.2%} ({highlight_pixels}/{roi_area})",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)

        # Result windows with fitting
        vw, vh = get_view_size()
        show_fit("Result (image space)", vis, vw, vh)

        bin_vis = colorize_binary(bin_img)
        show_fit("Binary (thresholded image)", bin_vis, vw, vh)

        mask_vis = colorize_binary(effective_roi)  # show effective ROI actually used
        show_fit("ROI mask (effective)", mask_vis, vw, vh)

        # Histogram inside EFFECTIVE ROI
        # roi_vals = gray[effective_roi == 255].ravel()
        # if roi_vals.size > 0:
        #     plt.figure("ROI intensity histogram")
        #     plt.clf()
        #     plt.hist(roi_vals, bins=256, range=(0, 255), alpha=0.85)
        #     plt.axvline(args.thresh, linestyle="--", linewidth=2, label=f"thresh={args.thresh}")
        #     plt.title(f"ROI gray histogram | N={roi_vals.size} | ratio={ratio:.2%}")
        #     plt.xlabel("Gray level (0-255)")
        #     plt.ylabel("Count")
        #     plt.legend(loc="best")
        #     plt.tight_layout()
        #     plt.show(block=False)
