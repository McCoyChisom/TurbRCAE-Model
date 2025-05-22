import gc
import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# from torchmetrics.image import StructuralSimilarityIndexMeasure
# import kornia.losses as kloss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import v2 as transforms

import json 


class ATSynDataset(Dataset):
    def __init__(
        self,
        path,
        transform=None,
        image_size=(192, 192),
        split="train",
        preload_data=False,
        degradation_type="turb",
    ):
        self.base_path = path
        self.transform = transform
        self.image_size = image_size
        self.xy_paths = []
        sample_dirs = sorted(os.listdir(self.base_path))
        ground_truth_filename = "gt.jpg"
        self.preloaded_data = []
        self.preload_data = preload_data
        
        # split into train/val/test
        train_split, val_split = train_test_split(
            sample_dirs, test_size=0.15, random_state=42
        )
        samples = train_split
        if split == "validation" or split == "val":
            samples = val_split
        elif split == "test":
            samples = sample_dirs

        for sample_dir in samples:
            # build paths
            ground_truth_path = os.path.join(
                self.base_path, sample_dir, ground_truth_filename
            )
            sample_turb_dir = os.path.join(self.base_path, sample_dir, degradation_type)
            turb_variation_filenames = os.listdir(sample_turb_dir)
            
            # Only take the first 10 variations to reduce size of dataset
            NUM_VARIATIONS = 3
            turb_variation_filenames = turb_variation_filenames[
                : min(NUM_VARIATIONS, len(turb_variation_filenames))
            ]
            for filename in turb_variation_filenames:
                # Append x, y sample
                x_path = os.path.join(sample_turb_dir, filename)
                y_path = ground_truth_path
                self.xy_paths.append((x_path, y_path))
                
                if preload_data:
                    # load & preprocess immediately
                    x = cv2.imread(x_path, cv2.IMREAD_COLOR)
                    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                    y = cv2.imread(y_path, cv2.IMREAD_COLOR)
                    y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)

                    y = self.transform(y)
                    x = self.transform(x)
                    # print(f"Max {torch.max(x)}, min {torch.min(x)}")

                    i, j, h, w = transforms.RandomCrop.get_params(
                        x, output_size=self.image_size
                    )
                    x = transforms.functional.crop(x, i, j, h, w)
                    y = transforms.functional.crop(y, i, j, h, w)

                    self.preloaded_data.append((x, y))

    def __len__(self):
        return len(self.xy_paths)

    def __getitem__(self, idx):
        x_path, y_path = self.xy_paths[idx]
        if self.preload_data:
            return self.preloaded_data[idx]

        # If file issues, retry once more in hope of resolving locked files
        retry_attempts = 1
        for _ in range(1 + retry_attempts):
            try:
                x = cv2.imread(x_path, cv2.IMREAD_COLOR)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                y = cv2.imread(y_path, cv2.IMREAD_COLOR)
                y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            except:
                time.sleep(1)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

            i, j, h, w = transforms.RandomCrop.get_params(
                x, output_size=self.image_size
            )
            x = transforms.functional.crop(x, i, j, h, w)
            y = transforms.functional.crop(y, i, j, h, w)

        return x, y


class TurbRCAE(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        # — Encoder path —
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # Spatial dim / 2

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, 2 * base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * base_ch),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)  # Spatial dim / 2

        self.enc3 = nn.Sequential(
            nn.Conv2d(2 * base_ch, 4 * base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * base_ch),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)  # Spatial dim / 2

        # — Bottleneck —
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                4 * base_ch, 4 * base_ch, kernel_size=3, padding=2, dilation=2
            ),  # Dilated conv
            nn.BatchNorm2d(4 * base_ch),
            nn.ReLU(inplace=True),
        )

        # — Decoder path —
        # Upsample + Conv: 16→32, Channels: 4*base_ch → 2*base_ch
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Input channels from bottleneck = 4 * base_ch
            nn.Conv2d(4 * base_ch, 2 * base_ch, kernel_size=3, padding=1),
        )
        # Decode Conv Block 3
        # Input channels from torch.cat([up3_out, enc3_out]) = (2*base_ch + 4*base_ch) = 6*base_ch
        self.dec3 = nn.Sequential(
            nn.Conv2d(6 * base_ch, 2 * base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * base_ch),
            nn.ReLU(inplace=True),
            # Output channels = 2 * base_ch
        )

        # Upsample + Conv: 32→64, Channels: 2*base_ch → base_ch
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Input channels from dec3_out = 2 * base_ch <-- CORRECTED
            nn.Conv2d(2 * base_ch, base_ch, kernel_size=3, padding=1),
        )
        # Decode Conv Block 2
        # Input channels from torch.cat([up2_out, enc2_out]) = (base_ch + 2*base_ch) = 3*base_ch
        self.dec2 = nn.Sequential(
            nn.Conv2d(3 * base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            # Output channels = base_ch
        )

        # Upsample + Conv: 64→128, Channels: base_ch → base_ch
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Input channels from dec2_out = base_ch <-- CORRECTED
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
        )
        # Decode Conv Block 1 (Final Output)
        # Input channels from torch.cat([up1_out, enc1_out]) = (base_ch + base_ch) = 2*base_ch
        self.dec1 = nn.Sequential(
            nn.Conv2d(2 * base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            # Final projection to output channels
            nn.Conv2d(base_ch, in_ch, kernel_size=1),
            nn.Tanh(),  # Assuming output is normalized between -1 and 1
            # Output channels = in_ch
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, base,128,128]
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)  # [B,2base,64,64]
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)  # [B,4base,32,32]
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)  # [B,4base,16,16]

        # Decoder + skip3
        u3 = self.up3(b)  # [B,2base,32,32]
        d3 = torch.cat([u3, e3], dim=1)  # [B,4base,32,32]
        d3 = self.dec3(d3)  # [B,2base,32,32]

        # Decoder + skip2
        u2 = self.up2(d3)  # [B, base,64,64]
        d2 = torch.cat([u2, e2], dim=1)  # [B,2base,64,64]
        d2 = self.dec2(d2)  # [B, base,64,64]

        # Decoder + skip1 → output
        u1 = self.up1(d2)  # [B, base,128,128]
        d1 = torch.cat([u1, e1], dim=1)  # [B,2base,128,128]
        out = self.dec1(d1)  # [B, 3,128,128]

        return out


#class TurbRCAE1(nn.Module):
 #   def __init__(self):
  #      super(TurbRCAE1, self).__init__()

        # Encoder: down to 32x32
   #     self.enc1 = nn.Sequential(
    #        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
     #       nn.BatchNorm2d(32),
      #      nn.ReLU(inplace=True),
       #     nn.MaxPool2d(kernel_size=2, stride=2),  # -> 32 x 64 x 64
        #)
        #self.enc2 = nn.Sequential(
         #   nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
          #  nn.BatchNorm2d(64),
           # nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  # -> 64 x 32 x 32 (bottleneck)
        #)
        #self.enc3 = nn.Sequential(
         #   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
          #  nn.BatchNorm2d(128),
           # nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  # -> 64 x 32 x 32 (bottleneck)
        #)

        # Decoder: upsample back to 128x128 without ConvTranspose2d

        #self.dec1 = nn.Sequential(
         #   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
         #  nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
          # nn.BatchNorm2d(64),
          # nn.ReLU(inplace=True),
        #)
        #self.dec2 = nn.Sequential(
         #   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
          #  nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
           # nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
        #)
        #self.dec3 = nn.Sequential(
         #   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
          #  nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
           # nn.Tanh(),
        #)

    #def forward(self, x):
      #  e1 = self.enc1(x)  # [B,32,64,64]
      #  e2 = self.enc2(e1)  # [B,64,32,32]
       # e3 = self.enc3(e2)  # [B,128,16,16]

        #d1 = self.dec1(e3)  # [B,64,32,32]
        #d1 = torch.cat([d1, e2], dim=1)  # [B,128,32,32]

        #d2 = self.dec2(d1)  # [B,32,64,64]
        #d2 = torch.cat([d2, e1], dim=1)  # [B,64,64,64]

        #out = self.dec3(d2)  # [B,3,128,128]
        #return out


# Define the VGGPerceptualLoss class
class VGGPerceptualLoss(nn.Module):
    def __init__(self, vgg_model="vgg19", layers=None):
        super(VGGPerceptualLoss, self).__init__()
        if vgg_model == "vgg19":
            self.vgg = models.vgg19(
                weights=models.VGG19_Weights.IMAGENET1K_V1
            ).features.eval()
        else:
            raise NotImplementedError(f"Model {vgg_model} not implemented")

        if layers is None:
            self.layers = [2, 7, 12, 21, 30]
        else:
            self.layers = layers

        for param in self.vgg.parameters():
            param.requires_grad = False
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)

        loss = 0.0
        for x_f, y_f in zip(x_features, y_features):
            loss += F.mse_loss(x_f, y_f, reduction="mean")

        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.1, inv_normalize=None, device="cuda"):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta  # Beta is used to scale down perceptual loss to values closer to 0-1 like MSE
        self.mse_loss = nn.MSELoss().to(device)
        self.perceptual_loss = VGGPerceptualLoss().to(device)
        # self.ssim_loss = kloss.SSIMLoss(5)
        self.inv_normalize = inv_normalize

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        # ssim = self.ssim_loss(output, target)
        # If values alreadu have been normalized, they need to be inversed since VGG does
        # its own normalization
        if self.inv_normalize:
            output = self.inv_normalize(output)
            target = self.inv_normalize(target)
        perceptual = self.perceptual_loss(output, target)
        return self.alpha * mse + (1 - self.alpha) * (self.beta * perceptual)
        # return self.alpha * mse + (1 - self.alpha) * ssim


# Define EarlyStopping (as before)
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def plot_one_batch(
    model,
    dataloader,
    post_process_fn,
    device="cuda",
    save=False,
    show=True,
    plot_save_path="plot",
):
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(dataloader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)

    input_images = [post_process_fn(input) for input in inputs]
    prediction_images = [post_process_fn(prediction) for prediction in predictions]
    target_images = [post_process_fn(target) for target in targets]

    # Plot the images
    fig = plt.figure(figsize=(15, 6), dpi=150)
    num_imgs = min(4, len(input_images))
    for i in range(num_imgs):
        # Noisy
        plt.subplot(3, 4, i + 1)
        plt.imshow(input_images[i])
        plt.title("Noisy")
        plt.axis("off")

        # Clean
        plt.subplot(3, 4, i + 5)
        plt.imshow(target_images[i])
        plt.title("Ground Truth")
        plt.axis("off")

        # Reconstructed
        plt.subplot(3, 4, i + 9)
        plt.imshow(prediction_images[i])
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    if save:
        fig.savefig(plot_save_path, bbox_inches="tight")
        plt.close(fig)
    if show:
        fig.show()
        plt.close(fig)


def validate_improvement(model, device, dataloader, loss_fn, post_process_fn, epoch):
    val_running_psnr = 0.0
    val_running_ssim = 0.0
    val_running_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            v_output = model.forward(inputs)

            v_loss = loss_fn(v_output, targets)
            val_running_loss += (
                v_loss.item() * batch_size
            )  # multiply by batch size since the loss is already mean over the samples

            total_samples += batch_size

            for output_image, target_image in zip(v_output, targets):
                # Post process images
                output_image = post_process_fn(output_image)
                target_image = post_process_fn(target_image)
                val_running_psnr += psnr(output_image, target_image, data_range=255)
                val_running_ssim += ssim(
                    output_image, target_image, data_range=255, channel_axis=-1
                )

    avg_loss = val_running_loss / total_samples
    avg_psnr = val_running_psnr / total_samples
    avg_ssim = val_running_ssim / total_samples
    print(f"Epoch {epoch+1} validation loss: {avg_loss}")
    print(f"Epoch {epoch+1} validation ssim: {avg_ssim}")
    print(f"Epoch {epoch+1} validation psnr: {avg_psnr}")

    return avg_loss


def compute_mean_std(dataloader, device=torch.device("cpu")):
    """
    Compute the per-channel mean and standard deviation of images in a DataLoader.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader yielding image batches as tensors
            with shape (B, C, H, W) and values in [0, 1].
        device (torch.device): Device on which to perform computations.

    Returns:
        mean (torch.Tensor): Tensor of shape (C,) with per-channel means.
        std (torch.Tensor): Tensor of shape (C,) with per-channel standard deviations.
    """
    # Accumulators
    channels_sum = torch.zeros(3, device=device)
    channels_squared_sum = torch.zeros(3, device=device)
    num_batches = 0

    print("Computing mean...")
    total_batches = len(dataloader)
    for i, (images, _) in enumerate(dataloader):
        # Move to device
        images = images.to(device)

        # Sum over batch, height, and width (C stays)
        channels_sum += images.sum(dim=[0, 2, 3])
        channels_squared_sum += (images**2).sum(dim=[0, 2, 3])
        num_batches += images.size(0)
        print(f"Completed batch [{i}/{total_batches}]")

    # Number of pixels per channel
    total_pixels = num_batches * images.size(2) * images.size(3)

    # Mean and STD computation
    mean = channels_sum / total_pixels
    # Var = E[X^2] - (E[X])^2
    var = (channels_squared_sum / total_pixels) - mean**2
    std = torch.sqrt(var)

    return mean, std


def tensor_to_uint8(
    img_tensor: torch.Tensor, inv_normalize: transforms.Transform
) -> np.ndarray:
    """
    img_tensor: C×H×W FloatTensor, normalized
    Returns: H×W×C uint8 array in [0,255]
    """
    # a) undo normalization
    img_denorm = inv_normalize(img_tensor)
    # b) clamp to [0,1]
    img_denorm = img_denorm.clamp(0, 1)
    # c) convert to H×W×C numpy in [0,255]
    img_np = img_denorm.mul(255).byte()  # scale to [0,255] & to uint8
    img_np = img_np.permute(1, 2, 0).cpu().numpy()  # C×H×W → H×W×C
    return img_np


def get_dataset_mean_std(dataset_path, device):
    temp_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((128, 128)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    temp_train_set = ATSynDataset(
        dataset_path,
        temp_transform,
        split="train",
    )
    BATCH_SIZE = 16
    NUM_WORKERS = 30
    train_dataloader = DataLoader(
        temp_train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Calculate mean and std for train set once so we can normalize values
    mean, std = compute_mean_std(train_dataloader, device)

    return mean, std


def plot_train_val_loss(train_losses, val_losses, show=True):
    fig = plt.figure(
        dpi=100
    )  # Create a new figure with a resolution of 100 dots per inch (makes plot clearer)
    plt.plot(
        train_losses, label="Training Loss"
    )  # Plot the training loss values stored in the 'losses' list
    plt.plot(
        val_losses, label="Validation Loss"
    )  # Plot the training loss values stored in the 'losses' list
    plt.xlabel("Epochs")  # Label the x-axis as "Epochs"
    plt.ylabel(
        "Loss"
    )  # Label the y-axis as "Loss (MSE)" - referring to Mean Squared Error. The Lower the bettetr
    plt.title(
        "Convolutional Autoencoder Loss"
    )  # Set the title of the plot as Convolutional Autoencoder Training Loss
    plt.legend()  # Display the legend to show which line represents training loss
    plt.grid(True)  # Add a grid to the plot for better visualization
    plt.savefig("loss")
    if show:
        plt.show()  # Render & display the plot
    plt.close(fig)


def test_model(model_path, degradation_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TurbRCAE(base_ch=32).to(device)
    #model.load_state_dict(torch.load(model_path))  # Replaced since I have CPU only device
    #model.load_state_dict(
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Hard-coded CPU
    # state_dict = torch.load(model_path, mao_location=device)  # Use device variable
    
    # both hard-coded and device varible methods now gives a CPU-MAPPED state dict
    model.load_state_dict(state_dict)

    # Train set mean and std after the pixels values have been converted to float 0.0-1.0
    train_mean = torch.tensor([0.4750, 0.4592, 0.4289])
    train_std = torch.tensor([0.2499, 0.2490, 0.2715])

    # To convert images back to normal values
    inv_normalize = transforms.Normalize(
        mean=(-train_mean / train_std).tolist(), std=(1.0 / train_std).tolist()
    )
    PATCH_SIZE = 192
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist()),
        ]
    )

    test_set = ATSynDataset(
        #"../datasets/static_new/test_static/",
        "ATSyn_static/static_new/test_static",  # Relative path
        #"/home/som/RSOTA_venv/models/ATSyn_static/static_new/test_static",  # Absolute path
        transform,
        (PATCH_SIZE, PATCH_SIZE),
        split="test",
        degradation_type=degradation_type,
    )

    BATCH_SIZE = 1
    NUM_WORKERS = 1
    test_dataloader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    post_process_fn = lambda image: tensor_to_uint8(image, inv_normalize)

    criterion = CombinedLoss(
        alpha=0.7, beta=0.1, inv_normalize=inv_normalize, device=device
    )

    test_psnrs = []
    test_ssims = []
    test_losses = []
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            model.train(False)

            predictions = model(inputs)

            for input, pred, target in zip(inputs, predictions, targets):
                input = post_process_fn(input)
                pred = post_process_fn(pred)
                target = post_process_fn(target)

                test_psnrs.append(psnr(pred, target, data_range=255))
                test_ssims.append(ssim(pred, target, data_range=255, channel_axis=-1))
                # (optional) Visualize the images that give very poor or good result (change the comparison value)
                # if test_ssims[-1] > 0.9:
                #     cv2.imshow(
                #         "Input, prediction, target",
                #         np.concatenate(
                #             (np.concatenate((input, pred), axis=1), target), axis=1
                #         ),
                #     )
                #     cv2.waitKey(0)

            loss = criterion(predictions, targets)
            test_losses.append(loss)

    plot_batch_dataloader = DataLoader(test_set, batch_size=4, num_workers=0)
    plot_one_batch(
        model,
        plot_batch_dataloader,
        post_process_fn,
        save=True,
        show=True,
        plot_save_path=os.path.join(
            "results", f"{degradation_type}_best_model_comparison_on_test_data"
        ),
    )

    return {
        "psnr": test_psnrs,
        "ssim": test_ssims,
        "loss": test_losses,
    }


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    model_directory = "results/checkpoints/cae"
    os.makedirs(model_directory, exist_ok=True)
    best_model_name = "best_model"
    best_model_path = os.path.join(model_directory, best_model_name)
    use_best_model_checkpoint = True
    if not os.path.exists(best_model_path):
        use_best_model_checkpoint = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize model, optimizer, etc.
    model = TurbRCAE(base_ch=32).to(device)
    if use_best_model_checkpoint:
        model.load_state_dict(torch.load(best_model_path))

    # mean, std = get_dataset_mean_std(
    #     "/home/jdev/dev/trakka/turbulence-mitigation/datasets/ATSyn_static/train_static/",
    #     device,
    # )
    # print(f"Train dataset mean: {mean}, std: {std}")
    ## For ATSyn static it should print this:
    ## tensor([0.4750, 0.4592, 0.4289], device='cuda:0'), std: tensor([0.2499, 0.2490, 0.2715], device='cuda:0')
    # return

    # Train set mean and std after the pixels values have been converted to float 0.0-1.0
    train_mean = torch.tensor([0.4750, 0.4592, 0.4289])
    train_std = torch.tensor([0.2499, 0.2490, 0.2715])

    # To convert images back to orignal space
    inv_normalize = transforms.Normalize(
        mean=(-train_mean / train_std).tolist(), std=(1.0 / train_std).tolist()
    )
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist()),
        ]
    )

    PATCH_SIZE = 192
    train_set = ATSynDataset(
        #"../datasets/static_new/train_static/",  # Johannes's system structure
        "ATSyn_static/static_new/train_static/",  # file structure on my Ubuntu environment
        transform,
        (PATCH_SIZE, PATCH_SIZE),
        split="train",
        preload_data=True,
    )
    val_set = ATSynDataset(
        "../datasets/static_new/train_static/",  # Johannes's system structure
        "ATSyn_static/static_new/train_static/",  
        transform,
        (PATCH_SIZE, PATCH_SIZE),
        split="val",
        preload_data=True,
    )

    BATCH_SIZE = 96
    NUM_WORKERS = 15
    train_dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        prefetch_factor=1,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    post_process_fn = lambda image: tensor_to_uint8(image, inv_normalize)

    criterion = CombinedLoss(
        alpha=0.7, beta=0.1, inv_normalize=inv_normalize, device=device
    )
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Define the number of epochs
    num_epochs = 10000
    early_stopping = EarlyStopping(patience=10, delta=0.001)
    best_v_loss = 1_000_000
    train_losses = []
    val_losses = []

    # -------------
    # Training Loop
    # -------------
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        total_batches = len(train_dataloader)

        for batch_num, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            model.train(True)

            optimizer.zero_grad()
            predictions = model(inputs)

            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            i += 1

            print(f"Finished batch [{batch_num+1}/{total_batches}] in epoch {epoch+1}")

        avg_loss = running_loss / i
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validate improvement
        val_loss = validate_improvement(
            model, device, val_dataloader, criterion, post_process_fn, epoch
        )
        val_losses.append(val_loss)

        if val_loss < best_v_loss:
            torch.save(model.state_dict(), best_model_path)
            torch.save(
                model.state_dict(),
                os.path.join(model_directory, f"model_{timestamp}_{epoch+1}"),
            )

        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        if epoch % 5 == 0:
            plot_train_val_loss(train_losses, val_losses, show=False)
            plot_one_batch(
                model,
                val_dataloader,
                post_process_fn,
                save=True,
                show=False,
                plot_save_path=os.path.join(
                    "results", f"model_{timestamp}_{epoch+1}_comparison_val_data"
                ),
            )
        gc.collect()
        torch.cuda.empty_cache()

    plot_train_val_loss(train_losses, val_losses, show=True)

    plot_one_batch(
        model,
        test_dataloader,
        post_process_fn,
        save=True,
        show=True,
        plot_save_path=os.path.join("results", "best_model_comparison_on_test_data"),
    )


def compute_metrics(values):
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    variance = np.var(values)
    min = np.min(values)
    max = np.max(values)

    return mean, std, variance, min, max


def main():
    # train_model()

    # Test model
    model_paths_with_degradation_type = [
         ("/home/som/RSOTA_venv/models/blur_32_channel_vgg_mse_108_epoch/best_model.pt", "blur"),
         ("/home/som/RSOTA_venv/models/tilt_32_channel_vgg_mse_64_epoch/best_model.pt", "tilt"),
         ("/home/som/RSOTA_venv/models/turb_32_channel_vgg_mse_69_epoch/best_model.pt", "turb"),
    ]
    # ensure that test loop is active
    for model_path, degradation_type in model_paths_with_degradation_type:
        results = test_model(model_path, degradation_type)
        
        # INSERTED JSON-SAVE BLOCK HERE
        # save out the raw metrics for later inspection
        save_path = f"{degradation_type}_baseline_metrics.json"
        with open(save_path, "w") as f:
            json.dump({
                "psnr": results["psnr"],
                "ssim": results["ssim"],
                "loss": [float(l) for l in results["loss"]]
            }, f, indent=4)
        print(f"Saved baseline metrics to {save_path}")

        # now continue with your plotting...
        
        # mean, std, variance, min, max = compute_metrics(results["ssim"])
        # mean, std, variance, min, max = compute_metrics(results["psnr"])
        plt.figure(figsize=(10, 6))
        plt.title(f"Test results for {degradation_type}")
        # ...

        # mean, std, variance, min, max = compute_metrics(results["ssim"])
        # mean, std, variance, min, max = compute_metrics(results["psnr"])

        #plt.figure(figsize=(10, 6))
        #plt.title(f"Test results for {degradation_type}")

        # Box plot for SSIM
        plt.subplot(1, 2, 1)
        plt.boxplot(
            results["ssim"], patch_artist=True, boxprops=dict(facecolor="lightblue")
        )
        plt.title("SSIM Value Distribution")
        plt.ylabel("SSIM")
        plt.xticks([1], ["SSIM Scores"])  # Hides the x-axis tick label 1
        plt.grid(True, axis="y")

        # Box plot for PSNR
        plt.subplot(1, 2, 2)
        plt.boxplot(
            results["psnr"], patch_artist=True, boxprops=dict(facecolor="lightgreen")
        )
        plt.title("PSNR Value Distribution")
        plt.ylabel("PSNR (dB)")
        plt.xticks([1], ["PSNR Scores"])
        plt.grid(True, axis="y")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
