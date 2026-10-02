"""Detect the previous white-output and accessory-area-fill regressions."""
import numpy as np
import cv2
from quality import quality_stats,debug_composite
from core import detect_glasses
from test_deglass import synthetic


def test_white_output_and_wrong_dark_range_are_rejected():
    image,_,_,points = synthetic()
    mask = np.zeros(image.shape[:2],np.uint8)
    cv2.line(mask,(150,250),(320,250),255,5)
    guard = np.zeros_like(mask)
    white = image.copy()
    white[mask>0] = 255
    qa = quality_stats(image,white,mask,guard,points)
    assert "near_white_inpainting_outlier" in qa["flags"]
    assert "inside_luminance_outside_surrounding_range" in qa["flags"]
    dark = image.copy()
    dark[mask>0] = 1
    assert "inside_luminance_outside_surrounding_range" in quality_stats(image,dark,mask,guard,points)["flags"]


def test_area_fill_is_flagged_even_if_luminance_is_plausible():
    image,_,_,points = synthetic()
    mask = np.zeros(image.shape[:2],np.uint8)
    mask[185:275,135:350] = 255
    qa = quality_stats(image,image,mask,np.zeros_like(mask),points)
    assert "eye_mask_area_outlier" in qa["flags"]
    assert not qa["checks_passed"]


def test_uniform_accessory_probability_cannot_fill_skin():
    image,clean,_,points = synthetic()
    accessory = np.ones(image.shape[:2],np.float32)
    result = detect_glasses(clean,points,accessory_probability=accessory)
    assert not np.any(result.mask)
    result = detect_glasses(image,points,accessory_probability=accessory)
    qa = quality_stats(image,clean,result.mask,result.protected,points)
    assert "eye_mask_area_outlier" not in qa["flags"]
    assert qa["eye_mask_fraction"]<.12


def test_debug_panels_and_empty_mask_statistics():
    image,_,_,points = synthetic()
    mask = np.zeros(image.shape[:2],np.uint8)
    mask[250,250] = 255
    debug = debug_composite(image,image,mask)
    assert debug.shape==(512,1536,3)
    assert np.array_equal(debug[250,250],image[250,250])
    assert debug[250,512+250,2]>debug[250,512+250,0]
    assert np.array_equal(debug[250,1024+250],image[250,250])
    qa = quality_stats(image,image,np.zeros_like(mask),np.zeros_like(mask),None)
    assert qa["checks_passed"]
    assert qa["inside_luminance_mean"] is None
