# Image Colorization with CNN & GANs

A PyTorch conditional GAN that learns to colorize grayscale images. The generator predicts the two Lab chroma channels from luminance; a discriminator distinguishes real from generated color conditioned on luminance. Training pairs are created automatically from RGB images, so no manually paired grayscale dataset is needed.

## Data layout
Place RGB training images in class subfolders (the class names do not affect training):
```
data/train/cats/*.jpg
 data/train/scenes/*.png
```
The script recursively reads common image formats. Start with a small dataset for a quick experiment.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python src/train.py --data data/train --epochs 50 --image-size 128
```
Checkpoints and preview grids are saved under `artifacts/`. Preview columns show grayscale input, generated color, and ground truth. Training can benefit from a CUDA GPU; CPU runs are supported but slower.

## Approach
Images are converted to CIE Lab. The generator maps L (lightness) to ab (color), then combines predicted ab with the original L to reconstruct RGB. The objective combines adversarial loss with L1 reconstruction loss to discourage plausible but incorrect colors. This is a learning experiment: color is ambiguous and generated hues may not match the original scene.

## Colorize one image
After training, run:
```bash
python src/colorize.py --input path/to/grayscale-or-color-image.jpg --checkpoint artifacts/generator.pt --output artifacts/colorized.png
```
The model uses the input image's luminance and predicts color. Keep sample and licensed data out of Git if the files are large; include only images you own or are permitted to redistribute.
## Sample images
The sample training images will be placed in `data/train/` when image generation is available. Until then, use any RGB images you have permission to train on. The training script converts RGB images into paired Lab luminance/chroma examples automatically.

## Included demo dataset (no API required)
This repository includes 24 small, procedurally drawn 128×128 BMP landscapes in `data/train/generated/`. They are created locally with the Python standard library—no account, API key, or network call is used. They let you run the full training flow immediately, but are synthetic illustrations, so use a larger licensed photo dataset for realistic colorization.

Create another local set or regenerate the included examples with:
```bash
python src/create_sample_data.py --output data/train/generated --count 24 --size 128
```
Then train as usual:
```bash
python src/train.py --data data/train --epochs 50 --image-size 128
```
