import torch
from model import TurbRCAE  

def export(model_path, onnx_output_path):
    model = TurbRCAE(base_ch=32)  # TurbCAE2(base_ch=32)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 3, 192, 192)  # Match model's expected input size

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model to: {onnx_output_path}")


# Export ONNX for each degradation model
export(
    "/home/som/RSOTA_venv/models/blur_32_channel_vgg_mse_108_epoch/best_model.pt",
    "/home/som/RSOTA_venv/models/blur.onnx"
)

export(
    "/home/som/RSOTA_venv/models/tilt_32_channel_vgg_mse_64_epoch/best_model.pt",
    "/home/som/RSOTA_venv/models/tilt.onnx"
)

export(
    "/home/som/RSOTA_venv/models/turb_32_channel_vgg_mse_69_epoch/best_model.pt",
    "/home/som/RSOTA_venv/models/turb.onnx"
)






#import torch
#from model import TurbCAE2  # assuming model class is saved in model.py

#def export(model_path, output_path):
    # Set up model
#    model = TurbCAE2(base_ch=32)
#    model.load_state_dict(torch.load(model_path, map_location='cpu'))
#    model.eval()

    # Dummy input: 1 sample, 3 channels, 128x128 resolution
#    dummy_input = torch.randn(1, 3, 128, 128)

    # Export
#    torch.onnx.export(
#        model,
#        dummy_input,
#        output_path,
#        input_names=["input"],
#        output_names=["output"],
#        export_params=True,
#        opset_version=11,
#        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
#    )
#    print(f"Exported ONNX model to: {output_path}")

#if __name__ == "__main__":
#    export("models/blur_32_channel_vgg_mse_108_epoch/best_model", "models/blur.onnx")
#    export("models/tilt_32_channel_vgg_mse_64_epoch/best_model", "models/tilt.onnx")
 #   export("models/turb_32_channel_vgg_mse_69_epoch/best_model", "models/turb.onnx")


