import cv2
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures # parallel processing

from mosse import MOSSE
from csk import CSK
from kcf import KCF

def parse_groundtruth(gt_path):
    rects = []
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) >= 4:
                rects.append(tuple(map(lambda x: int(round(float(x))), parts[:4])))
    return rects

def iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def load_otb_sequence(base_dir, seq_name):
    seq_dir = os.path.join(base_dir, seq_name)
    img_dir = os.path.join(seq_dir, 'img')

    gt_files = glob.glob(os.path.join(seq_dir, 'groundtruth_rect*.txt'))
    exact_gt = os.path.join(seq_dir, 'groundtruth_rect.txt')
    candidates = [exact_gt] + [f for f in gt_files if os.path.abspath(f) != os.path.abspath(exact_gt)]
    candidates = [f for f in candidates if os.path.isfile(f)]

    if not os.path.isdir(img_dir) or not candidates:
        raise FileNotFoundError(f"Sequence {seq_name} incomplete in {base_dir}")

    files = sorted([f for p in ['*.jpg', '*.jpeg', '*.png', '*.bmp'] 
                    for f in glob.glob(os.path.join(img_dir, p))])

    rects = []
    for gt_file in candidates:
        rects = parse_groundtruth(gt_file)
        if rects: break

    if not files or not rects:
        raise FileNotFoundError(f"Sequence {seq_name} missing images or valid GT")
    
    return files, rects

def plot_iou(ious, seq_name, save_path=None):
    plt.figure()
    plt.plot(ious, label='IoU')
    plt.xlabel('Frame')
    plt.ylabel('IoU')
    plt.title(f'{seq_name} - IoU per frame')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_sequence(seq_name, base_otb, output_dir=None, show=True, verbose=True, 
                      out_frame_tolerance=-1, zero_iou_tolerance=-1, tracker_type='MOSSE'):
    """
    evaluate one single seq. tolerance: 连续多少帧出界（或IoU=0）时停止评估，-1停用
    """
    files, gt_rects = load_otb_sequence(base_otb, seq_name)

    first_img = cv2.imread(files[0])
    frame_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY).astype('float32') / 255.0
    
    if tracker_type == 'MOSSE':
        tracker = MOSSE(frame_gray, gt_rects[0])
    elif tracker_type == 'CSK':
        tracker = CSK(frame_gray, gt_rects[0])
    elif tracker_type == 'KCF':
        tracker = KCF(frame_gray, gt_rects[0])
    else:
        if verbose: print(f"Unknown tracker {tracker_type}, defaulting to MOSSE")
        tracker = MOSSE(frame_gray, gt_rects[0])

    dists, ious = [], []
    out_frame_counter = 0
    zero_iou_counter = 0
    stop_frame_at = None

    n_frames = min(len(files), len(gt_rects))

    for idx in range(n_frames):
        img = cv2.imread(files[idx])
        if img is None: continue
        
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32') / 255.0
        x, y, w, h = tracker._update(frame_gray)
        pred = (x, y, w, h)

        # check out frame
        h_img, w_img = frame_gray.shape
        is_out = (x + w <= 0) or (x >= w_img) or (y + h <= 0) or (y >= h_img)
        
        if is_out:
            out_frame_counter += 1
        else:
            out_frame_counter = 0
        
        if out_frame_tolerance != -1 and out_frame_counter >= out_frame_tolerance:
            stop_frame_at = idx
            if verbose: print(f"{seq_name}: Stopped at frame {idx} (Out of view for {out_frame_tolerance} frames)")
            break

        # caculate iou / distance
        gt = gt_rects[idx]
        cp, cg = (x + w/2, y + h/2), (gt[0] + gt[2]/2, gt[1] + gt[3]/2)
        curr_iou = iou(pred, gt)
        
        dists.append(np.hypot(cp[0]-cg[0], cp[1]-cg[1]))
        ious.append(curr_iou)

        # check iou=0
        if curr_iou <= 0.0:
            zero_iou_counter += 1
        else:
            zero_iou_counter = 0
            
        if zero_iou_tolerance != -1 and zero_iou_counter >= zero_iou_tolerance:
            stop_frame_at = idx
            if verbose: print(f"{seq_name}: Stopped at frame {idx} (IoU=0 for {zero_iou_tolerance} frames)")
            break

        # visualization
        if show:
            cv2.rectangle(img, (int(gt[0]), int(gt[1])), (int(gt[0]+gt[2]), int(gt[1]+gt[3])), (0,0,255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(img, f"Frame {idx}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(seq_name, img)
            if cv2.waitKey(30) & 0xFF == ord('q'): break

    if show: cv2.destroyAllWindows()
    if not dists: return None

    dists_arr, ious_arr = np.array(dists), np.array(ious)
    prec20 = np.mean(dists_arr <= 20.0) * 100.0
    avg_iou = np.mean(ious_arr)

    if verbose:
        print(f"{seq_name}: Precision @20px: {prec20:.2f}% | Average IoU: {avg_iou:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_iou(ious_arr, seq_name, os.path.join(output_dir, f'iou_{seq_name}.png'))

    return prec20, avg_iou, ious_arr, dists_arr, stop_frame_at

def _worker(args):
    seq, base, out, out_tol, iou_tol, tracker_type = args
    return evaluate_sequence(seq, base, output_dir=out,
                            show=False, verbose=False, 
                            out_frame_tolerance=out_tol,
                            zero_iou_tolerance=iou_tol,
                            tracker_type=tracker_type)

def run_batch(base_otb, output_dir, tracker_type='MOSSE'):
    
    seqs = [d for d in sorted(os.listdir(base_otb)) if os.path.isdir(os.path.join(base_otb, d))]
    workers = os.cpu_count()
    print(f"Evaluating {len(seqs)} sequences with {workers} workers using {tracker_type}...")
    
    OUT_TOL = -1
    IOU_TOL = -1

    args_list = [(s, base_otb, output_dir, OUT_TOL, IOU_TOL, tracker_type) for s in seqs]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, a): a[0] for a in args_list}
        for fut in concurrent.futures.as_completed(futures):
            seq = futures[fut]
            try:
                res = fut.result()
                if res:
                    p20, m_iou, _, _, stop_at = res
                    note = f" | Stopped:{stop_at}" if stop_at is not None else ""
                    print(f"{seq}: Precision @20px: {p20:.2f}% | Average IoU: {m_iou:.4f}{note}")
                else:
                    print(f"{seq}: failed (No results returned)")
            except Exception as e:
                print(f"{seq}: failed ({e})")

def run_single(seq_name, base_otb, tracker_type='MOSSE'):

    res = evaluate_sequence(seq_name, base_otb,
                            show=True,
                            out_frame_tolerance=-1,
                            zero_iou_tolerance=-1,
                            tracker_type=tracker_type)
    if not res: return
    
    p20, m_iou, ious, _, stop_at = res
    if stop_at is not None:
        print(f"{seq_name}: Tracking stopped at frame {stop_at}")

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'iou_plot_{seq_name}_{tracker_type}.png')
    plot_iou(ious, f"{seq_name} ({tracker_type})", out_path)
    print(f"Saved plot to {out_path}")
    
    img = cv2.imread(out_path)
    if img is not None:
        cv2.imshow('IoU Plot', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    base_otb = os.path.join(os.path.dirname(__file__), 'OTB2015')

    if len(sys.argv) > 1:
        mode = sys.argv[1].strip()
        tracker_type = 'MOSSE'
        if len(sys.argv) > 2:
            tracker_type = sys.argv[2].strip().upper()
    else:
        mode = input("Sequence name (0 for all): ").strip()
        if mode != '0' and not os.path.isdir(os.path.join(base_otb, mode)):
            print(f"Error: Sequence '{mode}' not found.")
            return
        tracker_type = input("Tracker Algorithm (MOSSE/CSK/KCF): ").strip().upper() or 'MOSSE'
    
    if mode == '0':
        run_batch(base_otb, os.path.join(os.path.dirname(__file__), 'output'), tracker_type)
    else:
        run_single(mode, base_otb, tracker_type)

if __name__ == "__main__":
    main()