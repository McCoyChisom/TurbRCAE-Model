import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import qai_hub as hub

from model import ATSynDataset, tensor_to_uint8

print("Current working directory:", os.getcwd())
print("Using transforms from:", transforms.__file__)

# === Config ===
degradations_and_models = {
    "blur": "mqk1wy11m",  # Confirmed Model ID for blur.onnx 
    "tilt": "mnll47lon",  # Confirmed Model ID for tilt.onnx
    "turb": "mq3p87prn",  # Confirmed Model ID for turb.onnx
}

dataset_path = "/home/som/RSOTA_venv/models/ATSyn_static/static_new/test_static"
PATCH_SIZE = 192
BATCH_SIZE = 1
NUM_WORKERS = 1

train_mean = torch.tensor([0.4750, 0.4592, 0.4289])
train_std = torch.tensor([0.2499, 0.2490, 0.2715])

normalize = transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
inv_normalize = transforms.Normalize(
    mean=(-train_mean / train_std).tolist(), std=(1.0 / train_std).tolist()
)
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    normalize,
])

post_process_fn = lambda image: tensor_to_uint8(image, inv_normalize)

for degradation_type, onnx_model_id in degradations_and_models.items():
    print(f"\n===== Running inference for: {degradation_type.upper()} =====")

    test_set = ATSynDataset(
        dataset_path,
        transform,
        (PATCH_SIZE, PATCH_SIZE),
        split="test",
        degradation_type=degradation_type,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    hub_model = hub.get_model(onnx_model_id)

    psnr_scores = []
    ssim_scores = []

    for inputs, targets in test_loader:
        inputs = inputs.numpy()
        #predictions = hub_model.run({"input": inputs})["output"]
        predictions = hub_model.run_on(inputs)["output"]  # This line makes the model run on the Qai_Hub

        predictions = torch.from_numpy(predictions)
        targets = targets

        for pred, target in zip(predictions, targets):
            pred_img = post_process_fn(pred)
            target_img = post_process_fn(target)

            psnr_scores.append(psnr(pred_img, target_img, data_range=255))
            ssim_scores.append(ssim(pred_img, target_img, data_range=255, channel_axis=-1))

    # === Plot Results ===
    plt.figure(figsize=(10, 6))
    plt.suptitle(f"QAI Hub Inference Results: {degradation_type}")

    plt.subplot(1, 2, 1)
    plt.boxplot(ssim_scores, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.title("SSIM Distribution")
    plt.ylabel("SSIM")
    plt.grid(True, axis="y")

    plt.subplot(1, 2, 2)
    plt.boxplot(psnr_scores, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.title("PSNR Distribution")
    plt.ylabel("PSNR (dB)")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()

    print("\n=== Evaluation Summary ===")
    print(f"Avg PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f}")
    print(f"Avg SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")

    # Save results to CSV
    results_df = pd.DataFrame({
        "PSNR": psnr_scores,
        "SSIM": ssim_scores,
    })
    csv_save_path = f"{degradation_type}_qaihub_results.csv"
    results_df.to_csv(csv_save_path, index=False)
    print(f"Saved results to: {csv_save_path}")





#import os
#import torch
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#from torch.utils.data import DataLoader
#from torchvision.transforms import v2 as transforms
#from skimage.metrics import peak_signal_noise_ratio as psnr
#from skimage.metrics import structural_similarity as ssim
#import qai_hub as hub

#from model import ATSynDataset, tensor_to_uint8

#print("Using transforms from:", transforms.__file__)

# === Configuration ===
#degradations_and_models = {
#    "blur": "mq3p8r03n",
#    "tilt": "mqezrkk7q",
#    "turb": "mn03zyy8n",
#}

#dataset_path = "/home/som/RSOTA_venv/models/ATSyn_static/static_new/test_static" 
#dataset_path = "../datasets/static_new/test_static/"
#PATCH_SIZE = 192
#BATCH_SIZE = 1
#NUM_WORKERS = 1

# Normalization constants
#train_mean = torch.tensor([0.4750, 0.4592, 0.4289])
#train_std = torch.tensor([0.2499, 0.2490, 0.2715])

#normalize = transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
#inv_normalize = transforms.Normalize(
#    mean=(-train_mean / train_std).tolist(),
#    std=(1.0 / train_std).tolist()
#)

#transform = transforms.Compose([
#    transforms.ToImage(),
#    transforms.ConvertImageDtype(torch.float32),
#    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
#    normalize,
#])

#post_process_fn = lambda image: tensor_to_uint8(image, inv_normalize)

# === Inference Loop ===
#for degradation_type, onnx_model_id in degradations_and_models.items():
#    print(f"\n===== Running inference for: {degradation_type.upper()} =====")

#    test_set = ATSynDataset(
#        dataset_path,
 #       transform,
 #       (PATCH_SIZE, PATCH_SIZE),
  
  #      split="test",
   #     degradation_type=degradation_type,
   # )

   # test_loader = DataLoader(
   #     test_set,
    #    batch_size=BATCH_SIZE,
     #   shuffle=False,
    #    num_workers=NUM_WORKERS,
     #   pin_memory=True,
    #)

    # Load compiled ONNX model from QAI Hub
    #hub_model = hub.Model(model_id=onnx_model_id)
    #onnx_model_id = "mq3p8r03n"
    #onnx_model_id = "mqezrkk7q"
    #onnx_model_id = "jgok8l1xp"
    #hub_model = hub.get_model(onnx_model_id)

   # psnr_scores = []
   # ssim_scores = []

   # for inputs, targets in test_loader:
   #     inputs_np = inputs.numpy()
    #    predictions = hub_model.run({"input": inputs_np})["output"]
     #   predictions = torch.from_numpy(predictions)

      #  for pred, target in zip(predictions, targets):
       #     pred_img = post_process_fn(pred)
        #    target_img = post_process_fn(target)

         #   psnr_scores.append(psnr(pred_img, target_img, data_range=255))
          #  ssim_scores.append(ssim(pred_img, target_img, data_range=255, channel_axis=-1))

    # Save results to CSV
   # results_df = pd.DataFrame({"PSNR": psnr_scores, "SSIM": ssim_scores})
   # csv_path = f"{degradation_type}_qaihub_results.csv"
   # results_df.to_csv(csv_path, index=False)
   # print(f"Saved results to {csv_path}")

    # === Plot Results ===
   # plt.figure(figsize=(10, 6))
   # plt.suptitle(f"QAI Hub Inference Results: {degradation_type}")

   # plt.subplot(1, 2, 1)
   # plt.boxplot(ssim_scores, patch_artist=True, boxprops=dict(facecolor="lightblue"))
   # plt.title("SSIM Distribution")
   # plt.ylabel("SSIM")
   # plt.grid(True, axis="y")

   # plt.subplot(1, 2, 2)
   # plt.boxplot(psnr_scores, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
   # plt.title("PSNR Distribution")
   # plt.ylabel("PSNR (dB)")
   # plt.grid(True, axis="y")

   # plt.tight_layout()
   # plt.show()

    #print("\n=== Evaluation Summary ===")
    #print(f"Avg PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f}")
    #print(f"Avg SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")













#from model import ATSynDataset
#from utils import tensor_to_uint8
#import os
#import torch
#import numpy as np
#import matplotlib.pyplot as plt
#import cv2
#import qai_hub as hub
#import json

#from torchvision.transforms import v2 as transforms
#print("Using transforms from:", transforms.__file__)
#from torch.utils.data import DataLoader
#from torchvision import transforms
#from torchvision.transforms import v2 as transforms
#from skimage.metrics import peak_signal_noise_ratio as psnr
#from skimage.metrics import structural_similarity as ssim
#import qai_hub as hub
#import pandas as pd

#import sys
#import os

# Insert correct model path at the front of sys.path
#correct_model_path = os.path.join(os.path.dirname(__file__), 'RSOTA_venv', 'models')
#sys.path.insert(0, correct_model_path)

#from model import ATSynDataset, tensor_to_uint8



# === Config ===
#degradations_and_models = {
 #   "blur": "j5qevl745",
  #  "tilt": "jgl6ly08g",
   # "turb": "jgok8l1xp",
#}

#dataset_path = "/home/som/RSOTA_venv/models/test_static"
#dataset_path = "/home/som/RSOTA_venv/models/ATSyn_static/static_new"
# Most recent dataset path
#dataset_path = "/home/som/RSOTA_venv/models/ATSyn_static/static_new/test_static"
#PATCH_SIZE = 192
#BATCH_SIZE = 1
#NUM_WORKERS = 1

#train_mean = torch.tensor([0.4750, 0.4592, 0.4289])
#train_std = torch.tensor([0.2499, 0.2490, 0.2715])

# === Preprocessing and Post-processing ===
#normalize = transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
#inv_normalize = transforms.Normalize(
 #   mean=(-train_mean / train_std).tolist(), std=(1.0 / train_std).tolist()
#)

#transform = transforms.Compose([
 #   transforms.ToImage(),
  #  transforms.ToDtype(torch.float32, scale=True),
   # transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist()),
#])

#post_process_fn = lambda image: tensor_to_uint8(image, inv_normalize)

#for degradation_type, onnx_model_id in degradations_and_models.items():
 #   print(f"\n===== Running inference for: {degradation_type.upper()} =====")

    # === Load Test Data ===
  #  test_set = ATSynDataset(
   #     dataset_path,
    #    transform,
     #   (PATCH_SIZE, PATCH_SIZE),
      #  split="test",
       # degradation_type=degradation_type,
    #)
    #test_loader = DataLoader(
     #   test_set,
      #  batch_size=BATCH_SIZE,
       # shuffle=False,
        #num_workers=NUM_WORKERS,
        #pin_memory=True,
    #)

    # === Run Inference on QAI Hub ===
    #hub_model = hub.Model(model_id=onnx_model_id)
    #hub_model = hub.load_model(onnx_model_id)
    #hub_model = hub.Model.from_model_id(onnx_model_id)
    #hub_model = hub.models.load_model(onnx_model_id)
    #qai_test_model = hub.get_model ("jgok8l1xp")



    #psnr_scores = []
    #ssim_scores = []

    #for inputs, targets in test_loader:
     #   inputs = inputs.numpy()
      #  predictions = hub_model.run({"input": inputs})["output"]

       # predictions = torch.from_numpy(predictions)
        #targets = targets

        #for pred, target in zip(predictions, targets):
         #   pred_img = post_process_fn(pred)
          #  target_img = post_process_fn(target)

           # psnr_scores.append(psnr(pred_img, target_img, data_range=255))
            #ssim_scores.append(ssim(pred_img, target_img, data_range=255, channel_axis=-1))

    # === Plot Results ===
   # plt.figure(figsize=(10, 6))
   # plt.suptitle(f"QAI Hub Inference Results: {degradation_type}")

    #plt.subplot(1, 2, 1)
   # plt.boxplot(ssim_scores, patch_artist=True, boxprops=dict(facecolor="lightblue"))
   # plt.title("SSIM Distribution")
    #plt.ylabel("SSIM")
   # plt.grid(True, axis="y")

    #plt.subplot(1, 2, 2)
    #plt.boxplot(psnr_scores, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    #plt.title("PSNR Distribution")
    #plt.ylabel("PSNR (dB)")
   # plt.grid(True, axis="y")

   # plt.tight_layout()
   # plt.show()

    # === Print Summary ===
   # print("\n=== Evaluation Summary ===")
   # print(f"Avg PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f}")
   # print(f"Avg SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")


# === Save scores to CSV ===
#results_df = pd.DataFrame({
 #   "PSNR": psnr_scores,
  #  "SSIM": ssim_scores,
#})

#csv_save_path = f"{degradation_type}_qaihub_results.csv"
#results_df.to_csv(csv_save_path, index=False)
#print(f"\nSaved results to: {csv_save_path}")

