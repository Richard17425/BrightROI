import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --------------------------
# Args
# --------------------------
parser = argparse.ArgumentParser(
    description="Compute highlight-pixel ratio inside a user-selected ROI (thresholding + interactive ROI)"
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

# Thresholding (binary mask in image coordinates)
# > thresh => 255 (bright), else 0; if invert, <= thresh => 255
if args.invert:
    _, bin_img = cv2.threshold(gray, args.thresh, 255, cv2.THRESH_BINARY_INV)
else:
    _, bin_img = cv2.threshold(gray, args.thresh, 255, cv2.THRESH_BINARY)

# --------------------------
# Window & view management
# --------------------------
WIN = "ROI (m: mode, wheel: zoom, middle/right drag: pan, Enter: done, r: reset, z: undo, q: quit)"
# Allow manual resizing, keep aspect if possible (behavior depends on backend)
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # WINDOW_NORMAL makes the window resizable
cv2.resizeWindow(WIN, args.init_width, args.init_height)       # initial size

# View state (image -> display transform)
scale = min(1.0, min(args.init_width / w, args.init_height / h))  # initial fit
min_scale, max_scale = 0.1, 10.0
ox, oy = 0.0, 0.0  # top-left offset (in display pixels)
dragging = False
last_mouse = (0, 0)

# ROI state
roi_mode = "poly"   # 'poly' or 'rect'
points_img = []     # polygon points in image coordinates
rect_start_img = None
rect_end_img = None
rect_dragging = False

# -------------- Helpers: coordinate transforms --------------
def show_fit(win_name, img, max_w, max_h):
    """
    Show an image fitted inside (max_w, max_h) while keeping aspect ratio.
    Creates a WINDOW_NORMAL|WINDOW_KEEPRATIO window and resizes it.
    """
    ih, iw = img.shape[:2]
    scale = min(max_w / iw, max_h / ih, 1.0)
    if scale < 1.0:
        disp = cv2.resize(img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = img

    # Create/ensure a resizable window and set its client size
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(win_name, disp)
    cv2.resizeWindow(win_name, disp.shape[1], disp.shape[0])

def disp_to_img(pt_disp):
    """Display (window canvas) -> image coordinates (int)."""
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

# -------------- Helpers: window size (best-effort) --------------
def get_view_size():
    """
    Try to query the current drawable image area inside the window.
    Fallback to the initial size if backend doesn't support it.
    """
    try:
        # getWindowImageRect returns (x, y, width, height)
        # Available in HighGUI (Qt/Win32) backends on most builds.
        _, _, ww, hh = cv2.getWindowImageRect(WIN)  # may raise if not available
        return max(200, ww), max(200, hh)
    except Exception:
        return args.init_width, args.init_height

# -------------- Rendering --------------
def render_base(view_w, view_h):
    """
    Return the current display image with the original image scaled & positioned
    according to (scale, ox, oy). This does NOT draw ROI overlays yet.
    """
    canvas = np.zeros((int(view_h), int(view_w), 3), dtype=np.uint8)
    scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    Hs, Ws = scaled.shape[:2]

    # Paste scaled image onto canvas at (ox, oy) with clipping
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
    """
    Draw ROI overlays (polygon points/lines or rectangle) onto display image.
    """
    # Draw polygon mode overlays
    if roi_mode == "poly":
        pts_disp = [img_to_disp(p) for p in points_img]
        for i, p in enumerate(pts_disp):
            cv2.circle(disp_img, p, 3, (0, 255, 255), -1)  # vertex
            if i > 0:
                cv2.line(disp_img, pts_disp[i - 1], pts_disp[i], (0, 255, 255), 1)
        if len(pts_disp) >= 3:
            cv2.polylines(disp_img, [np.array(pts_disp, dtype=np.int32)], True, (0, 255, 255), 1)
    else:
        # rectangle mode overlays
        if rect_start_img is not None and rect_end_img is not None:
            p0 = img_to_disp(rect_start_img)
            p1 = img_to_disp(rect_end_img)
            cv2.rectangle(disp_img, p0, p1, (0, 255, 255), 2)
    return disp_img

def draw_hud(disp_img, view_w, view_h):
    """
    Draw heads-up text with instructions and current state.
    """
    mode_txt = f"mode={roi_mode} | thresh={args.thresh} | scale={scale:.2f}x"
    hint_txt = "wheel: zoom | middle/right drag: pan | m: toggle ROI | z: undo | r: reset | Enter: done | q: quit"
    cv2.putText(disp_img, mode_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)
    cv2.putText(disp_img, hint_txt, (10, view_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return disp_img

# -------------- Mouse callbacks --------------
def on_mouse(event, x, y, flags, param):
    """
    Handle left-clicks (add/drag ROI), middle/right drag to pan,
    and mouse wheel to zoom around the cursor.
    """
    global dragging, last_mouse, scale, ox, oy
    global rect_dragging, rect_start_img, rect_end_img, points_img

    # Zoom with mouse wheel (keep cursor-anchored point fixed)
    if event == cv2.EVENT_MOUSEWHEEL:
        # On most backends, getMouseWheelDelta(flags) returns positive for forward (zoom in), negative for backward.
        delta = cv2.getMouseWheelDelta(flags)
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        new_scale = clamp(scale * factor, min_scale, max_scale)

        # Anchor: image-point under cursor before zoom
        img_x_before, img_y_before = disp_to_img((x, y))
        # Update scale
        if new_scale != scale:
            scale = new_scale
            # Keep the same image point under the cursor after zoom:
            ox = x - img_x_before * scale
            oy = y - img_y_before * scale

    # Start panning with middle or right button
    elif event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        dragging = True
        last_mouse = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            dx = x - last_mouse[0]
            dy = y - last_mouse[1]
            ox += dx
            oy += dy
            last_mouse = (x, y)
        elif roi_mode == "rect" and rect_dragging:
            # Update rectangle current corner
            rect_end_img = disp_to_img((x, y))

    elif event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
        dragging = False

    # Left button interactions for ROI
    if roi_mode == "poly":
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add a vertex in image coordinates
            points_img.append(disp_to_img((x, y)))
    else:
        # Rectangle mode
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_dragging = True
            rect_start_img = disp_to_img((x, y))
            rect_end_img = rect_start_img
        elif event == cv2.EVENT_LBUTTONUP:
            rect_dragging = False
            rect_end_img = disp_to_img((x, y))

cv2.setMouseCallback(WIN, on_mouse)

# Fit image to center initially
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
        # Toggle ROI mode
        roi_mode = "rect" if roi_mode == "poly" else "poly"
        # Clear current shapes when switching mode
        points_img = []
        rect_start_img = rect_end_img = None
    elif key == ord('z'):
        if roi_mode == "poly" and points_img:
            points_img.pop()
        elif roi_mode == "rect":
            rect_start_img = rect_end_img = None
    elif key == ord('r'):
        points_img = []
        rect_start_img = rect_end_img = None
    elif key in (13, 10):  # Enter/Return
        # Build ROI mask in image coordinates
        mask = np.zeros((h, w), dtype=np.uint8)

        if roi_mode == "poly":
            if len(points_img) < 3:
                print("Need at least 3 points for a polygon. Keep selecting...")
                continue
            poly = np.array(points_img, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly], 255)  # polygon area = 255
        else:
            if rect_start_img is None or rect_end_img is None:
                print("Draw a rectangle by dragging with left mouse button.")
                continue
            x0, y0 = rect_start_img
            x1, y1 = rect_end_img
            x_min, x_max = sorted((x0, x1))
            y_min, y_max = sorted((y0, y1))
            x_min = clamp(x_min, 0, w - 1)
            x_max = clamp(x_max, 0, w - 1)
            y_min = clamp(y_min, 0, h - 1)
            y_max = clamp(y_max, 0, h - 1)
            if x_max <= x_min or y_max <= y_min:
                print("Rectangle has zero area. Draw again.")
                continue
            mask[y_min:y_max+1, x_min:x_max+1] = 255

        # Compute ratios on the thresholded image
        roi_area = cv2.countNonZero(mask)
        if roi_area == 0:
            print("ROI area is zero. Please reselect.")
            continue
        bin_in_roi = cv2.bitwise_and(bin_img, bin_img, mask=mask)
        highlight_pixels = cv2.countNonZero(bin_in_roi)
        ratio = highlight_pixels / float(roi_area)

        # Print summary
        print("========== Result ==========")
        print(f"Image: {os.path.basename(args.image)}")
        print(f"Mode: {'dark (<= thresh)' if args.invert else 'bright (> thresh)'}")
        print(f"Threshold: {args.thresh}")
        print(f"ROI pixels: {roi_area}")
        print(f"{'Dark' if args.invert else 'Highlight'} pixels: {highlight_pixels}")
        print(f"Ratio: {ratio:.4%}")

        # Visualize: outline ROI + green overlay for counted pixels
        vis = img.copy()
        if roi_mode == "poly":
            cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
        else:
            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        overlay = vis.copy()
        overlay[bin_in_roi == 255] = (0, 255, 0)
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        cv2.putText(
            vis,
            f"ratio={ratio:.2%} ({highlight_pixels}/{roi_area})",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )
        # ----- ROI histogram (grayscale distribution inside the mask) -----


        # roi_vals = gray[mask == 255].ravel()      # all grayscale values inside ROI
        # if roi_vals.size > 0:
        #     plt.figure("ROI intensity histogram")
        #     # 0..255, 256 bins (8-bit)
        #     plt.hist(roi_vals, bins=256, range=(0, 255), alpha=0.85)
        #     plt.axvline(args.thresh, linestyle="--", linewidth=2, label=f"thresh={args.thresh}")
        #     plt.title(f"ROI gray histogram | N={roi_vals.size} | ratio={ratio:.2%}")
        #     plt.xlabel("Gray level (0-255)")
        #     plt.ylabel("Count")
        #     plt.legend(loc="best")
        #     plt.tight_layout()
        #     # 非阻塞显示，避免卡住 OpenCV 主循环
        #     plt.show(block=False)


        # Show results in separate windows (auto-resizable)

        vis_disp = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        bin_disp = cv2.resize(bin_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        mask_disp = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # 仍然用 show_fit，但这里的图本来就被缩放到与交互视图一致的比例
        show_fit("Result (image space)", vis_disp, vw, vh)
        show_fit("Binary (thresholded image)", bin_disp, vw, vh)
        show_fit("ROI mask (image space)", mask_disp, vw, vh)

        # cv2.imshow("Result (image space)", vis)
        # cv2.imshow("Binary (thresholded image)", bin_img)
        # cv2.imshow("ROI mask (image space)", mask)
