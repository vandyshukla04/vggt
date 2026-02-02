import torch
from glob import glob
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import predictions_to_glb

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load images from the rhinos folder (excluding grounded-sam subfolder)
image_dir = "/home/shuklva/CUT3R/examples/wd_data/rhinos/rhin-11"
image_names = sorted(glob(f"{image_dir}/*.jpg") + glob(f"{image_dir}/*.png") + glob(f"{image_dir}/*.jpeg"))
image_names = image_names[:20]  # Limit to 20 frames to avoid OOM
print(f"Using {len(image_names)} images")
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

# Convert pose encoding to extrinsic/intrinsic matrices
extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic

# Convert tensors to numpy
for key in predictions.keys():
    if isinstance(predictions[key], torch.Tensor):
        predictions[key] = predictions[key].cpu().numpy().squeeze(0)

# Save as GLB file
print("Saving to GLB...")
scene = predictions_to_glb(predictions, conf_thres=50.0)
output_path = "output_rhinos.glb"
scene.export(output_path)
print(f"Saved to {output_path}")