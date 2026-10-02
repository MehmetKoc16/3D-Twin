"""Regression tests use drawn geometry only, including offline model smoke tests."""
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import detect_glasses
from measure import estimate
from models import CACHE,AccessorySegmenter,LamaInpainter
from test_deglass import synthetic
from quality import quality_stats


def polygon_scene():
    _,clean,_,points = synthetic()
    truth = np.zeros(clean.shape[:2],np.uint8)
    angles = np.arange(8)*np.pi/4+np.pi/8
    for x in (180,306):
        vertices = np.rint(np.column_stack((x+46*np.cos(angles),230+46*np.sin(angles)))).astype(np.int32)
        cv2.polylines(truth,[vertices],True,255,3)
    cv2.line(truth,(223,220),(263,220),255,3)
    cv2.ellipse(truth,(234,241),(4,8),-20,0,360,255,2)
    cv2.ellipse(truth,(252,241),(4,8),20,0,360,255,2)
    cv2.line(truth,(105,220),(137,220),255,3)
    cv2.line(truth,(349,220),(390,220),255,3)
    image = clean.copy()
    image[truth>0] = (45,45,45)
    accessory = cv2.GaussianBlur(truth.astype(np.float32)/255,(7,7),0)
    return image,truth,points,accessory


def test_polygon_upper_rims_pads_and_hair_guard():
    image,truth,points,accessory = polygon_scene()
    hair = np.zeros(truth.shape,np.uint8)
    hair[180:235,75:125] = 255
    accessory[hair>0] = 1
    result = detect_glasses(image,points,accessory_probability=accessory,hair_mask=hair)
    target = (truth>0)&(result.protected==0)
    assert np.mean(result.mask[target]>0)>.98
    assert not np.any(result.mask[result.protected>0])
    assert not np.any(result.mask[hair>0])
    assert np.mean(result.mask[236:250,230:257]>0)>.2
    measured = estimate(image,points,result,lens_shape="rounded-polygon")
    assert measured["lens_shape"]=="rounded-polygon"
    assert 6<=measured["lens_corner_count_approx"]<=12
    assert .8<=measured["lens_roundness"]<=1
    assert measured["lens_outer_radius_m"]==pytest.approx(.023,abs=.004)


@pytest.fixture(scope="module")
def lama():
    if not (CACHE/"lama_fp32.onnx").exists():
        pytest.skip("Offline Big-LaMa model is not installed; run setup_models.py")
    return LamaInpainter()


def test_lama_removes_synthetic_polygon_and_preserves_unmasked(lama):
    image,truth,points,accessory = polygon_scene()
    detection = detect_glasses(image,points,accessory_probability=accessory)
    output = lama.inpaint(image,detection.mask)
    target = (truth>0)&(detection.protected==0)
    assert np.mean(np.min(output[target],axis=1)<100)<.05
    assert quality_stats(image,output,detection.mask,detection.protected,points)["checks_passed"]
    assert np.array_equal(output[detection.mask==0],image[detection.mask==0])
    assert np.array_equal(lama.inpaint(image,np.zeros_like(truth)),image)


def test_lama_tiling_covers_non_square_image(lama):
    image = np.full((300,750,3),180,np.uint8)
    mask = np.zeros(image.shape[:2],np.uint8)
    cv2.line(mask,(40,150),(710,150),255,5)
    image[mask>0]=30
    output = lama.inpaint(image,mask)
    assert output.shape==image.shape
    assert np.array_equal(output[mask==0],image[mask==0])
    assert np.mean(np.min(output[mask>0],axis=1)<100)<.05
    assert abs(float(output[mask>0].mean())-180)<30


def test_lama_export_range_and_rgb_order(lama):
    image = np.full((512,512,3),(80,150,210),np.uint8)
    mask = np.zeros(image.shape[:2],np.uint8)
    cv2.line(mask,(180,250),(330,250),255,3)
    output = lama.inpaint(image,mask)
    means = output[mask>0].mean(axis=0)
    assert np.all(np.abs(means-np.array([80,150,210]))<25),means
    assert means[0]<means[1]<means[2]
    assert output[mask>0].mean()<230
    assert lama.last_diagnostics["output_range"]=="0..255 (scaled inside export)"
    assert lama.last_diagnostics["unmasked_identity_max_error"]<1


def test_multiclass_model_offline_synthetic_smoke():
    if not (CACHE/"selfie_multiclass_256x256.tflite").exists():
        pytest.skip("Offline segmentation model not installed")
    segmenter = AccessorySegmenter()
    try:
        image,_,_,points = synthetic()
        accessory,hair = segmenter.segment(image,points)
        assert accessory.shape==hair.shape==image.shape[:2]
        assert np.isfinite(accessory).all()
        assert 0<=accessory.min()<=accessory.max()<=1
        assert set(np.unique(hair)).issubset({0,255})
    finally:
        segmenter.close()
