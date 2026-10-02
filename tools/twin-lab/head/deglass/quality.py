"""Numeric-only output checks and private debug composites; no image display."""
import cv2
import numpy as np
from core import pupil_centers

MAX_EYE_MASK_FRACTION = 0.12


def quality_stats(original: np.ndarray, result: np.ndarray, mask: np.ndarray,
                  protected: np.ndarray, landmarks: np.ndarray | None) -> dict:
    selected = mask>0
    flags = []
    stats = {"mask_pixels":int(selected.sum()), "inside_luminance_mean":None,
             "ring_luminance_mean":None,"ring_luminance_std":None,
             "inside_near_white_fraction":None,"eye_mask_fraction":None,
             "max_eye_mask_fraction":MAX_EYE_MASK_FRACTION,
             "plausible_luminance_range":None}
    if np.any(selected & (protected>0)):
        flags.append("protected_anatomy_overlap")
    if not np.any(selected):
        return {**stats,"flags":flags,"checks_passed":not flags}
    inside = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)[selected].astype(float)
    inner = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    outer = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    ring = (outer>0)&(inner==0)&(protected==0)
    stats["inside_luminance_mean"] = round(float(inside.mean()),3)
    stats["inside_near_white_fraction"] = round(float(np.mean(inside>=245)),6)
    if np.any(ring):
        values = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)[ring].astype(float)
        mean,std = float(values.mean()),float(values.std())
        tolerance = max(30.0,2*std)
        low,high = max(0,mean-tolerance),min(245 if mean<230 else 255,mean+tolerance)
        stats.update(ring_pixels=int(ring.sum()),ring_luminance_mean=round(mean,3),
                     ring_luminance_std=round(std,3),
                     plausible_luminance_range=[round(low,3),round(high,3)])
        if not low <= inside.mean() <= high:
            flags.append("inside_luminance_outside_surrounding_range")
        if ((inside.mean()>=245 and inside.mean()-mean>15)
                or (np.mean(inside>=245)>.25 and np.mean(values>=245)<.1)):
            flags.append("near_white_inpainting_outlier")
    else:
        flags.append("no_surrounding_ring_samples")
    if landmarks is not None:
        pupils = pupil_centers(landmarks)
        reference = max(float(np.linalg.norm(pupils[0]-pupils[1])),
                        float(np.linalg.norm(landmarks[10]-landmarks[152]))*.35)
        x0,x1 = np.min(pupils[:,0])-reference*.6,np.max(pupils[:,0])+reference*.6
        y = float(pupils[:,1].mean())
        y0,y1 = y-reference*.6,y+reference*.65
        h,w = mask.shape
        roi = selected[max(0,int(y0)):min(h,int(y1)+1),max(0,int(x0)):min(w,int(x1)+1)]
        fraction = float(roi.mean()) if roi.size else 1.0
        stats["eye_mask_fraction"] = round(fraction,6)
        if fraction>MAX_EYE_MASK_FRACTION:
            flags.append("eye_mask_area_outlier")
    return {**stats,"flags":flags,"checks_passed":not flags}


def debug_composite(original: np.ndarray, result: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = original.copy()
    selected = mask>0
    overlay[selected] = np.rint(original[selected]*.35+np.array([0,0,255])*.65).astype(np.uint8)
    panels = [original.copy(),overlay,result.copy()]
    for panel,label in zip(panels,("Original","Mask overlay (red)","Result")):
        cv2.putText(panel,label,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5,cv2.LINE_AA)
        cv2.putText(panel,label,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    return np.concatenate(panels,axis=1)
