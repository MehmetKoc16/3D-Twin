"""Synthetic-only fixtures: never read personal photos or their derived outputs."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import EYES, BROWS, detect_glasses, inpaint, private_path
from measure import estimate, profile_arm


def synthetic(colour=(45, 45, 45), size=512):
    image = np.full((512, 512, 3), (170, 190, 210), np.uint8)
    truth = np.zeros((512, 512), np.uint8)
    points = np.full((478, 2), [256, 260], np.float32)
    points[234], points[454] = [105, 245], [390, 245]
    points[10], points[152] = [256, 100], [256, 420]
    for i, x in enumerate((180, 306)):
        points[468 if i == 0 else 473] = [x, 220]
        points[list(EYES[i])] = [[x-20,220], [x-10,214], [x+10,214],
                                 [x+20,220], [x+10,226], [x-10,226]]
        points[list(BROWS[i])] = [[x-25+j*5, 168+(j%2)*3] for j in range(10)]
        cv2.ellipse(image, (x,220), (19,5), 0, 0, 360, (75,75,75), -1)
        cv2.line(image, (x-25,169), (x+25,169), (55,55,55), 4)
        cv2.circle(truth, (x,230), 46, 255, 3)
    cv2.line(truth, (226,220), (260,220), 255, 3)
    cv2.line(truth, (105,220), (134,220), 255, 3)
    cv2.line(truth, (352,220), (390,220), 255, 3)
    cv2.line(truth, (230,231), (234,240), 255, 2)
    cv2.line(truth, (256,231), (252,240), 255, 2)
    clean = image.copy()
    image[truth > 0] = colour
    if size != 512:
        clean = cv2.resize(clean, (size,size), interpolation=cv2.INTER_NEAREST)
        # Keep wire pixel thickness constant at higher resolution, matching
        # the current thin-wire requirement rather than doubling to 10px.
        factor = size/512
        truth = np.zeros((size,size),np.uint8)
        for x in (180,306):
            cv2.circle(truth,(round(x*factor),round(230*factor)),round(46*factor),255,3)
        for start,end,width in (((226,220),(260,220),3),((105,220),(134,220),3),
                                 ((352,220),(390,220),3),((230,231),(234,240),2),
                                 ((256,231),(252,240),2)):
            cv2.line(truth,tuple(round(v*factor) for v in start),
                     tuple(round(v*factor) for v in end),255,width)
        image = clean.copy()
        image[truth>0] = colour
        points *= size / 512
    return image, clean, truth, points


@pytest.mark.parametrize("colour,size", [((45,45,45),512), ((245,245,245),512), ((45,45,45),1024)])
def test_mask_recall_precision_and_anatomy(colour, size):
    image, _, truth, landmarks = synthetic(colour, size)
    result = detect_glasses(image, landmarks)
    selected = result.mask > 0
    target = (truth > 0) & (result.protected == 0)
    recall = np.count_nonzero(selected & target) / np.count_nonzero(target)
    exact_precision = np.count_nonzero(selected & target) / max(1, selected.sum())
    # Intentional dilation is evaluated against a small tolerated boundary too.
    margin = max(2, size // 256)
    tolerated = cv2.dilate(truth, np.ones((2*margin+1,2*margin+1),np.uint8)) > 0
    precision = np.count_nonzero(selected & tolerated) / max(1, selected.sum())
    assert recall > 0.90, recall
    assert exact_precision > 0.32, exact_precision
    assert precision > 0.90, precision
    assert not np.any(selected & (result.protected > 0))
    assert len(result.rings) == 2


@pytest.mark.parametrize("method", ["telea", "ns"])
def test_inpainting_removes_line_and_preserves_eyes(method):
    image, clean, truth, landmarks = synthetic()
    result = detect_glasses(image, landmarks)
    output = inpaint(image, result.mask, method)
    target = (truth > 0) & (result.protected == 0)
    residual = np.mean(np.min(output[target], axis=1) < 100)
    assert residual < 0.04, residual
    assert np.mean(np.abs(output[target].astype(float) - clean[target])) < 12
    assert np.array_equal(output[result.mask == 0], image[result.mask == 0])


def test_known_metric_geometry_and_ipd_scaling():
    image, _, _, landmarks = synthetic()
    detection = detect_glasses(image, landmarks)
    result = estimate(image, landmarks, detection)
    # 126 px IPD = 63 mm => 0.5 mm/px; radius 46 px, bridge 34 px.
    assert result["lens_outer_radius_m"] == pytest.approx(0.023, abs=0.002)
    assert result["bridge_width_m"] == pytest.approx(0.017, abs=0.003)
    assert result["frame_width_at_temples_m"] == pytest.approx(0.109, abs=0.004)
    assert result["lens_vertical_offset_m"] == pytest.approx(0.005, abs=0.002)
    assert 0.0005 <= result["frame_thickness_m"] <= 0.004
    assert result["temple_arm_length_m"] == 0.14
    scaled = estimate(image, landmarks, detection, 70)
    assert scaled["lens_outer_radius_m"] / result["lens_outer_radius_m"] == pytest.approx(70/63, rel=0.001)


def test_no_glasses_no_face_back_and_failed_measurement():
    image, clean, _, points = synthetic()
    assert not np.any(detect_glasses(clean, points).mask)
    for view in ("front", "back", "profile_nose_left"):
        result = detect_glasses(image, None, view)
        assert not np.any(result.mask)
        assert np.array_equal(inpaint(image, result.mask), image)
    with pytest.raises(ValueError, match="Two supported"):
        estimate(clean, points, detect_glasses(clean, points))


@pytest.mark.parametrize("ipd", [0, -63, 100, float("nan"), float("inf")])
def test_invalid_ipd(ipd):
    image, _, _, points = synthetic()
    with pytest.raises(ValueError, match="IPD"):
        estimate(image, points, detect_glasses(image, points), ipd)


def test_cli_path_privacy():
    with pytest.raises(ValueError, match="user-data"):
        private_path(Path("docs/leak.jpg"))


def test_projected_profile_rims_use_vertical_face_scale():
    image, _, truth, points = synthetic()
    profile = cv2.resize(image, (256,512))
    target = cv2.resize(truth, (256,512), interpolation=cv2.INTER_NEAREST)
    points[:,0] *= 0.5
    result = detect_glasses(profile, points, "profile_nose_left", 126/320, 40/126)
    eligible = (target>0) & (result.protected==0)
    assert len(result.rings) == 2
    assert np.mean(result.mask[eligible]>0) > 0.85


def test_visible_profile_arm_and_default_colour_statistic():
    image, _, _, points = synthetic()
    result = detect_glasses(image, points)
    result.core_mask[:] = 0
    cv2.line(result.core_mask,(370,220),(500,220),255,2)
    assert profile_arm(image,points,result,40,0.0005) == pytest.approx(0.09,abs=0.003)
    result.core_mask[:] = 0
    assert profile_arm(image,points,result,40,0.0005) is None
    measured = estimate(image,points,result)
    expected = np.rint(np.median(image[result.mask>0][:,::-1],axis=0)).astype(int).tolist()
    assert measured["frame_colour_rgb"] == expected
