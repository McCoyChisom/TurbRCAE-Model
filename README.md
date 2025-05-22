# Please, do not follow the steps in this README file just yet, until I have created time to edit and update it with the correct information. Most of the things I have written here have been changed since the last time I managed to write down things.

To begin with, I have used Ubuntu 24.04, hence to make most of the environment setup, I had to create and use a Virtual environment:

Please Note:
To check your python code for any syntax erorr before compiling: python -m py_compile main.py

To Activate RSOTA_venv (bash/zsh):
bash
source ~/RSOTA_venv/bin/activate

# Install Conda:
See https://app.aihub.qualcomm.com/docs/hub/getting_started.ht

Miniconda is recommend to manage your python versions and environments.

Step 1: Python environment
Install miniconda on your machine.

Windows: When the installation finishes, open Anaconda Prompt from the Start menu.
macOS/Linux: When the installation finished, open a new shell window.
Set up an environment for Qualcomm® AI Hub:

conda create python=3.10 -n qai_hub
conda activate qai_hub


# If you're already inside a Conda environment ((qai_hub)), you might want to deactivate it first:
bash
conda deactivate

# If deactivation persistently fails, use the following command:
exec bash
 of close termial

# Then Activate the Virtual Environment:
bash
source ~/RSOTA_venv/bin/activate

# To deactivate RSOTA_venv:
deactivate


#  using conda (qai_hub) and then activated RSOTA_venv inside it, run the following command:
to activate everything again in a new terminal:
bash
(RSOTA_venv) som@Chisom-HP-EliteBook-Folio-1040-G3:~$ conda activate qai_hub

# This will take you to Qualcomm AI Hub Virtual environment using Conda:
(qai_hub) (RSOTA_venv) som@Chisom-HP-EliteBook-Folio-1040-G3:~$ 
Now you can perform all the other environment setup and your experiment, here.


# Move the best_model.pt files to the models folder of your Virtual environment (/home/som/RSOTA_venv/models)

mkdir -p /home/som/RSOTA_venv/models && mv /home/som/Downloads/blur_32_channel_vgg_mse_108_epoch/best_model.pt /home/som/Downloads/tilt_32_channel_vgg_mse_64_epoch/best_model.pt /home/som/Downloads/turb_32_channel_vgg_mse_69_epoch/best_model.pt /home/som/RSOTA_venv/models/


# Sumitting the ONNX file and compiling to Qualcomm AI Hub for Experimentation

 1. # Activate Your Virtual Environment, if your case is same as mine. 
 (qai_hub) (RSOTA_venv) som@Chisom-HP-EliteBook-Folio-1040-G3:~$ 
 
 2. # Import qa_hub and Submit the Compile Job. File is saved as compile_to_qaihub.py
import qai_hub as hub

def compile_model(onnx_model_path):
    compile_job = hub.submit_compile_job(
        model=onnx_model_path,
        device=hub.Device("QCS8550 (Proxy)"),
        options="--target_runtime onnx",
        input_specs=dict(input=(1, 3, 128, 128))
    )

    print(f"Submitted compile job for: {onnx_model_path}")
    compile_job.wait()

    status = compile_job.get_status()
    print(f"Compile status: {status.code}")

    if status.code == "SUCCESS":
        print(f"Compiled model is ready: {compile_job.get_target_model()}")
    else:
        print("Compilation failed.")

if __name__ == "__main__":
    compile_model("/home/som/RSOTA_venv/models/blur.onnx")
    compile_model("/home/som/RSOTA_venv/models/tilt.onnx")
    compile_model("/home/som/RSOTA_venv/models/turb.onnx")




3. Run the following bash command
(qai_hub) (RSOTA_venv) som@Chisom-HP-EliteBook-Folio-1040-G3:~$ python3 compile_to_qaihub.py
Uploading blur.onnx
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.91M/1.91M [00:00<00:00, 3.37MB/s]100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.91M/1.91M [00:01<00:00, 1.41MB/s]
Scheduled compile job (jpxnx3xl5) successfully. To see the status and results:
    https://app.aihub.qualcomm.com/jobs/jpxnx3xl5/

Submitted compile job for: /home/som/RSOTA_venv/models/blur.onnx
Waiting for compile job (jpxnx3xl5) completion. Type Ctrl+C to stop waiting at any time.
    ✅ SUCCESS                          
Compile status: SUCCESS
Compiled model is ready: Model(model_id='mnw5d5kxn', name='job_jpxnx3xl5_optimized_onnx')
Uploading tilt.onnx
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.91M/1.91M [00:01<00:00, 1.51MB/s]
Scheduled compile job (j5mq8o89p) successfully. To see the status and results:
    https://app.aihub.qualcomm.com/jobs/j5mq8o89p/

Submitted compile job for: /home/som/RSOTA_venv/models/tilt.onnx
Waiting for compile job (j5mq8o89p) completion. Type Ctrl+C to stop waiting at any time.
    ✅ SUCCESS                          
Compile status: SUCCESS
Compiled model is ready: Model(model_id='mmx0w06jm', name='job_j5mq8o89p_optimized_onnx')
Uploading turb.onnx
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.91M/1.91M [00:01<00:00, 1.11MB/s]
Scheduled compile job (jgnlkokq5) successfully. To see the status and results:
    https://app.aihub.qualcomm.com/jobs/jgnlkokq5/

Submitted compile job for: /home/som/RSOTA_venv/models/turb.onnx
Waiting for compile job (jgnlkokq5) completion. Type Ctrl+C to stop waiting at any time.
    ✅ SUCCESS                          
Compile status: SUCCESS
Compiled model is ready: Model(model_id='mnll4lown', name='job_jgnlkokq5_optimized_onnx')

#Yipee! YTour ONNX models are now successfully compiled and now you are ready for inference and evaluation on QAI_Hub. To do so, please follow this step-by-step guide, ensuring tha everything matches

Step 1: 
# Confirm Your Test Dataset and Normalization
Test dataset .../datasets/static_new/test_static/

Option 1: Because The dataset exists in my Google drive, from where I used it in Google Colab for training, you will need to download the dataset from Drive to local system. In my case, I used Ubuntu 24.o4, so I Use gdown with --remaining-ok to download the dataset from Google Drive to my local Ubuntu syste as follows:
 
(qai_hub) (RSOTA_venv) som@Chisom-HP-EliteBook-Folio-1040-G3:~$ pip install gdown==4.7.1

#Go to the dataset in Google drive and navigate to test_static. In my case: 
/content/drive/MyDrive/Dataset/static_new/test_static. 

#Then copy the share file URL which looks like: 
https://drive.google.com/drive/folders/1-1oe1FNyb6-IMElM3glouVGfO5P0AezE?usp=sharing

# Now, run the download the test-static dataset to your local system to beging testing on QAI-Hub:
gdown --folder --id 1-1oe1FNyb6-IMElM3glouVGfO5P0AezE -O /home/som/Dataset/static_new/ --remaining-ok

# Confirm the files:
ls /home/som/Dataset/static_new/test_static


Patch size: 192

Degradation types: "blur", "tilt", "turb"

Normalization: Ensure that exactly the normalization below is used before inference on QAI_Hub and for inverse normaloization (for PSNR/SSIM). 
python
train_mean = torch.tensor([0.4750, 0.4592, 0.4289])
train_std = torch.tensor([0.2499, 0.2490, 0.2715])

Step 2: 
# Modify test_model() for QAI_Hub Inference
Create a new function, say test_model_qaihub() that reuses the same processing pipeline and PSNR/SSIM logic, but replaces:
python
predictions = model(inputs) 

with :

python
predictions = qaihub_model.run({"input": inputs})["output"]


# Now Create a Test Script for your experiment in QAI_Hub
 Navigate to the folder where you have kept all your useful scripts and create the script for testing on QAI_Hub. In my case: som@Chisom-HP-EliteBook-Folio-1040-G3:~/RSOTA_venv/models$ 
 
Then, create test_model_qaihub.py

Copy and paste the test_model_qaihub.py script, but adapt to your environment.

# What the Script Does:
Loads the test_static dataset (same degradation as used locally).

Uses the exact normalization and post-processing pipeline.

Runs inference on QAI Hub using the compiled ONNX model ID.

Calculates PSNR and SSIM.

Plots boxplots and prints summary stats.

Note: 

# Make sure that you have matplotlib installed on your local system before running the script:
pip3 install matplotlib

# Also run this to install Scikit-learn in your Virtual environment:
pip install scikit-learn








# Why Test the Model on Qualcomm AI Hub?
By running test_model_qaihub.py, the thesis is aiming to compare the evaluation metrics (PSNR, SSIM, etc.) obtained on Qualcomm AI Hub with those you recorded during training locally (the baseline model). This provides a solid approach for validating consistency, performance drops, or improvements due to quantization or hardware-specific optimizations.

To make this comparison meaningful, the thesis proceeded in the following step by step sequence of actions:

# Step 1: Define and Save Baseline Metrics Local 
The thesis ensured thar evaluation metrics on the original (PyTorch) model are saved:
1. Evaluation was made on a fixed test set (the same that will be used later for Qualcomm AI Hub).

2. Metrics were saved in a .json file as baseline_metrics.json:


with open("baseline_metrics.json", "w") as f:
    json.dump(baseline_metrics, f, indent=4)
    
import json

metrics_to_save = {
    "psnr": results["psnr"],
    "ssim": results["ssim"],
    "loss": [loss.item() for loss in results["loss"]]
    "Model": "TurbRCAE",
    "Test Set": "ATSyn-static",
    "Notes": "Baseline on PyTorch T4 GPU"
}

with open(f"../results/{degradation_type}_baseline_metrics.json", "w") as f:
    json.dump(metrics_to_save, f, indent=4)





Step 4: Fetch Output from AI Hub and Evaluate Locally
If AI Hub allows batch inference, here’s a reliable way:

Upload the test images to AI Hub.

Collect the model predictions from cloud inference.

Compare them locally to compute PSNR and SSIM:



