import os
import glob
import numpy as np
from PIL import Image
import qai_hub as hub
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import to_tensor
from model import tensor_to_uint8

# Constants
BASE_MODEL_PATH = "/home/som/RSOTA_venv/models"
TEST_DATASET_PATH = os.path.join(BASE_MODEL_PATH, "ATSyn_static/static_new/test_static")
ONNX_MODELS = {
    "blur": os.path.join(BASE_MODEL_PATH, "blur.onnx"),
    "tilt": os.path.join(BASE_MODEL_PATH, "tilt.onnx"),
    "turb": os.path.join(BASE_MODEL_PATH, "turb.onnx"),
}
DEVICE = hub.Device("QCS8550 (Proxy)")
INPUT_SHAPE = (1, 3, 192, 192)

# Normalization config
train_mean = torch.tensor([0.4750, 0.4592, 0.4289])
train_std = torch.tensor([0.2499, 0.2490, 0.2715])
inv_normalize = transforms.Normalize(
    mean=(-train_mean / train_std).tolist(),
    std=(1.0 / train_std).tolist()
)

for degradation, model_path in ONNX_MODELS.items():
    print(f"\n=== Processing model for: {degradation.upper()} ===")

    # --- Gather all (image, gt) pairs for this degradation ---
    valid_samples = []
    scene_dirs = sorted(glob.glob(os.path.join(TEST_DATASET_PATH, "*")))
    for scene_dir in scene_dirs:
        degradation_dir = os.path.join(scene_dir, degradation)
        gt_path = os.path.join(scene_dir, "gt.jpg")
        if not os.path.isdir(degradation_dir) or not os.path.exists(gt_path):
            continue
        for img_path in sorted(glob.glob(os.path.join(degradation_dir, "*.png"))):
            valid_samples.append((img_path, gt_path))

    if not valid_samples:
        print(f"No valid samples found for {degradation}, skipping.")
        continue
    print(f"Found {len(valid_samples)} samples for {degradation}")

    # --- Step A: Compile original ONNX to static ONNX ---
    print("Compiling original ONNX to static shape...")
    compile_static = hub.submit_compile_job(
        model=model_path,
        device=DEVICE,
        options="--target_runtime onnx",
        input_specs={"input": INPUT_SHAPE},
    )
    compile_static.wait()
    if compile_static.get_status().code != "SUCCESS":
        print("Static compile failed.")
        continue
    static_model = compile_static.get_target_model()
    print("Static ONNX ready:", static_model.model_id)

    # --- Step B: Quantize the static ONNX (INT8) ---
    print("Quantizing static ONNX (INT8)...")
    C, H, W = INPUT_SHAPE[1:]
    calibration_dataset = {"input": []}
    for img_path, _ in valid_samples[:50]:
        img = Image.open(img_path).resize((W, H))
        np_img = np.array(img, dtype=np.float32) / 255.0          # H×W×C
        chw = np.transpose(np_img, (2, 0, 1))                     # C×H×W
        arr = chw[None, ...]                                      # 1×C×H×W
        calibration_dataset["input"].append(arr)

    quantize_job = hub.submit_quantize_job(
        static_model,                         # compiled static ONNX model
        calibration_dataset,                  # dict: input name -> list of numpy arrays
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name=f"{degradation}_quant"
    )
    quantize_job.wait()
    print(f"Quantize Job URL: https://app.aihub.qualcomm.com/jobs/{quantize_job.job_id}")
    if quantize_job.get_status().code != "SUCCESS":
        print("Quantization failed:", quantize_job.get_status())
        continue
    quantized_model = quantize_job.get_target_model()
    print("Quantized model ready:", quantized_model.model_id)

    # --- Step C: Re-compile quantized ONNX to QNN binary ---
    print("Compiling quantized ONNX to QNN binary...")
    compile_qnn = hub.submit_compile_job(
        model=quantized_model,
        device=DEVICE,
        options="--target_runtime onnx",
        input_specs={"input": INPUT_SHAPE},
    )
    compile_qnn.wait()
    if compile_qnn.get_status().code != "SUCCESS":
        print("QNN compile failed.")
        continue
    target_model = compile_qnn.get_target_model()
    print("QNN binary ready:", target_model.model_id)

    # --- Step D: Profile the compiled QNN model ---
    print("Profiling model...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=DEVICE
    )
    profile_job.wait()
    print(f"Profile Job URL: https://app.aihub.qualcomm.com/jobs/{profile_job.job_id}")
    if profile_job.get_status().code != "SUCCESS":
        print("Profiling failed.")
        # But we’ll continue to inference anyway

    # --- Step E: Inference & Metrics on all samples ---
    psnr_scores = []
    ssim_scores = []

    for img_path, gt_path in valid_samples:
        # prepare input
        img = Image.open(img_path).resize((W, H))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]  # 1×C×H×W

        inf_job = hub.submit_inference_job(
            model=target_model,
            device=DEVICE,
            inputs={"input": [arr]}
        )
        inf_job.wait()
        out = inf_job.download_output_data()
        if out is None or "output_0" not in out:
            print(f"Inference failed for {img_path}")
            continue

        pred_tensor = torch.from_numpy(out["output_0"][0])
        pred_img = tensor_to_uint8(pred_tensor, inv_normalize)

        # load GT
        gt_img = Image.open(gt_path).resize((W, H))
        gt_tensor = to_tensor(gt_img)    # C×H×W in [0,1]
        gt_uint8 = tensor_to_uint8(gt_tensor, inv_normalize)

        psnr_scores.append(psnr(gt_uint8, pred_img, data_range=255))
        ssim_scores.append(ssim(gt_uint8, pred_img, data_range=255, channel_axis=-1))

    # --- Step F: Plot & Save Metrics ---
    os.makedirs("results", exist_ok=True)
    df = {
        "PSNR mean": np.mean(psnr_scores),
        "PSNR std":  np.std(psnr_scores),
        "SSIM mean": np.mean(ssim_scores),
        "SSIM std":  np.std(ssim_scores),
    }
    print(f"Results summary for {degradation}:", df)

    plt.figure(figsize=(10, 6))
    plt.suptitle(f"{degradation.upper()} PSNR/SSIM Distribution")

    plt.subplot(1, 2, 1)
    plt.boxplot(psnr_scores, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.title("PSNR")
    plt.ylabel("dB")
    plt.grid(True, axis="y")

    plt.subplot(1, 2, 2)
    plt.boxplot(ssim_scores, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.title("SSIM")
    plt.ylabel("Index")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(f"results/{degradation}_psnr_ssim.png")
    plt.show()

