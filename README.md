# 4K Video Generation Pipeline (HunyuanVideo-I2V)

This repository provides a custom tile-scale pipeline for 4K video generation, building upon [HunyuanVideo-I2V](https://github.com/Tencent/HunyuanVideo-I2V).

## Instructions

### 1. Clone the Official Repository

```bash
git clone https://github.com/Tencent/HunyuanVideo-I2V.git
cd HunyuanVideo-I2V
```

### 2. Set Up the Environment

Follow the official instructions in the HunyuanVideo-I2V repository to install the required dependencies.

### 3. Move the Provided Pipeline

Move the provided `pipeline_hunyuani2v_tilescale.py` into the pipelines directory, replacing the existing file:

```bash
cd HunyuanVideo-I2V
cp ./pipeline_hunyuani2v_tilescale.py ./hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py
```

### 4. Run the Video Generation

Use the standard scripts for video generation as described in the official repository.  
Ensure you specify the appropriate parameters for 4K output.

```bash
cd HunyuanVideo-I2V
# Here's the default command line as in HunyuanVideo-I2V's official repository.
python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick." \  # change to your prompt as needed
    --i2v-mode \
    --i2v-image-path ./assets/demo/i2v/imgs/0.jpg \  # change to your desired path
    --i2v-resolution 720p \
    --i2v-stability \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --use-cpu-offload \
    --save-path ./results
```
```