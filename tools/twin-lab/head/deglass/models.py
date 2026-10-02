"""Offline model inference. Setup downloads are deliberately separate."""
from pathlib import Path
import cv2
import numpy as np

CACHE = Path(__file__).resolve().parent / ".cache"


class AccessorySegmenter:
    def __init__(self, model: Path = CACHE / "selfie_multiclass_256x256.tflite"):
        import mediapipe as mp
        self.mp = mp
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
            mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                output_category_mask=True, output_confidence_masks=True))

    def segment(self, image: np.ndarray, landmarks: np.ndarray | None) -> tuple[np.ndarray,np.ndarray]:
        h,w = image.shape[:2]
        accessories, hair = np.zeros((h,w),np.float32), np.zeros((h,w),np.uint8)
        regions = [(0,0,w,h)]
        if landmarks is not None:
            # A tight head crop resolves thin accessories better than the full portrait.
            low, high = landmarks.min(axis=0),landmarks.max(axis=0)
            margin = float(high[0]-low[0])*.2
            regions.append((max(0,int(low[0]-margin)),max(0,int(low[1]-margin)),
                            min(w,int(high[0]+margin)),min(h,int(high[1]+margin))))
        for x0,y0,x1,y1 in regions:
            rgb = np.ascontiguousarray(cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB))
            result = self.segmenter.segment(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb))
            probabilities = [m.numpy_view().copy() for m in result.confidence_masks]
            if len(probabilities) != 6:
                raise ValueError("Expected six-class selfie segmentation model")
            size = (x1-x0,y1-y0)
            accessory = cv2.resize(probabilities[5],size,interpolation=cv2.INTER_LINEAR)
            hair_probability = cv2.resize(probabilities[1],size,interpolation=cv2.INTER_LINEAR)
            category = cv2.resize(result.category_mask.numpy_view(),size,interpolation=cv2.INTER_NEAREST)
            accessories[y0:y1,x0:x1] = np.maximum(accessories[y0:y1,x0:x1],accessory)
            hair[y0:y1,x0:x1] |= ((category==1)|(hair_probability>=.35)).astype(np.uint8)*255
        # The guard remains strict across segmentation scales.
        hair = cv2.dilate(hair,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        return accessories,hair

    def close(self) -> None:
        self.segmenter.close()


class LamaInpainter:
    def __init__(self, model: Path = CACHE / "lama_fp32.onnx"):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model),options,providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        by_name = {item.name:item for item in inputs}
        outputs = {item.name:item for item in self.session.get_outputs()}
        for name,channels in (("image",3),("mask",1)):
            item = by_name.get(name)
            if item is None or item.shape[1:]!=[channels,512,512] or item.type!="tensor(float)":
                raise ValueError("Unexpected Big-LaMa input contract")
        output = outputs.get("output")
        if output is None or output.shape[1:]!=[3,512,512] or output.type!="tensor(float)":
            raise ValueError("Unexpected Big-LaMa output contract")
        self.image_name,self.mask_name,self.output_name = "image","mask","output"
        # Carve's export includes clamp(prediction * 255) in the graph.
        # The reference PyTorch generator returns 0..1; this ONNX output is 0..255.
        self.last_diagnostics = {}

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Overlapping 512px context tiles; only final-mask pixels are replaced."""
        if not np.any(mask):
            return image.copy()
        h,w = mask.shape
        ys,xs = np.where(mask>0)
        x_start,y_start = max(0,int(xs.min())-128),max(0,int(ys.min())-128)
        x_stop,y_stop = min(w,int(xs.max())+129),min(h,int(ys.max())+129)
        def origins(start,stop,limit):
            start = min(start,max(0,limit-512))
            last = min(max(start,stop-512),max(0,limit-512))
            return sorted(set(list(range(start,last+1,256))+[last]))
        sums = np.zeros_like(image,dtype=np.float32)
        weights = np.zeros((h,w),np.float32)
        window = np.maximum(.05,np.hanning(512)).astype(np.float32)
        window = window[:,None]*window[None,:]
        raw_min,raw_max,identity_error = 255.0,0.0,0.0
        tile_count = 0
        for y in origins(y_start,y_stop,h):
            for x in origins(x_start,x_stop,w):
                tile = image[y:min(y+512,h),x:min(x+512,w)]
                local_mask = mask[y:min(y+512,h),x:min(x+512,w)]
                if not np.any(local_mask):
                    continue
                th,tw = local_mask.shape
                # 512 is a multiple of the reference model's pad modulus (8).
                padded = cv2.copyMakeBorder(tile,0,512-th,0,512-tw,cv2.BORDER_REFLECT)
                padded_mask = cv2.copyMakeBorder(local_mask,0,512-th,0,512-tw,cv2.BORDER_CONSTANT)
                rgb = cv2.cvtColor(padded,cv2.COLOR_BGR2RGB).astype(np.float32)/255
                values = self.session.run([self.output_name],{
                    self.image_name: np.ascontiguousarray(rgb.transpose(2,0,1)[None]),
                    self.mask_name: (padded_mask[None,None]>0).astype(np.float32)})[0]
                if values.shape != (1,3,512,512) or not np.isfinite(values).all():
                    raise ValueError("Invalid Big-LaMa output")
                raw_min = min(raw_min,float(values.min()))
                raw_max = max(raw_max,float(values.max()))
                if raw_min < -.01 or raw_max > 255.01:
                    raise ValueError("Big-LaMa output violates the 0..255 export contract")
                prediction_rgb = values[0].transpose(1,2,0)
                context = padded_mask==0
                if np.any(context):
                    error = float(np.max(np.abs(prediction_rgb[context]-rgb[context]*255)))
                    identity_error = max(identity_error,error)
                    if error>1:
                        raise ValueError("Big-LaMa unmasked identity check failed: range/layout/output mismatch")
                output = np.clip(prediction_rgb,0,255)[:,:,::-1]
                tile_count += 1
                weight = window[:th,:tw]*(local_mask>0)
                sums[y:y+th,x:x+tw] += output[:th,:tw]*weight[:,:,None]
                weights[y:y+th,x:x+tw] += weight
        selected = mask>0
        if np.any(weights[selected]<=0):
            raise ValueError("Big-LaMa tiles did not cover the mask")
        output = image.copy()
        output[selected] = np.rint(sums[selected]/weights[selected,None]).astype(np.uint8)
        self.last_diagnostics = {"output_tensor":self.output_name,"layout":"NCHW RGB",
                                 "input_range":"0..1","mask_range":"binary 0/1",
                                 "output_range":"0..255 (scaled inside export)",
                                 "raw_min":round(raw_min,4),"raw_max":round(raw_max,4),
                                 "unmasked_identity_max_error":round(identity_error,6),
                                 "tiles":tile_count}
        return output
