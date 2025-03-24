# Human Face Colorization

## Overview
This project focuses on human face colorization using deep learning techniques. The goal is to develop a model capable of converting grayscale facial images into realistic color images using convolutional neural networks (CNNs) and generative adversarial networks (GANs).

## Features
- Converts black-and-white facial images into colored versions
- Uses deep learning architectures like CNNs and GANs
- Pretrained models for improved colorization accuracy
- User-friendly interface for easy usage

## Dataset
We use publicly available datasets such as:
- CelebA (Large-scale celebrity face dataset)
- FFHQ (Flickr-Faces-HQ for high-quality face images)
- Custom grayscale images for testing

## Installation
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/face-colorization.git
cd face-colorization

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the following command to colorize an image:

```bash
python colorize.py --input path/to/grayscale_image.jpg --output path/to/output.jpg
```

## Model Architecture
- **Feature Extractor:** A deep CNN extracts features from grayscale images.
- **Colorization Network:** Uses GANs or CNN-based U-Net to predict color channels.
- **Loss Functions:** Combination of pixel-wise loss and perceptual loss for better color accuracy.

## Results
Include some before-and-after images showcasing the model's performance.

## Future Improvements
- Improve model accuracy using transformer-based approaches
- Enhance real-time performance
- Integrate with mobile applications

## Contributors
- [Elias Huber](https://github.com/yourusername)
- [Sam Rouppe Van der Voort](https://github.com/teammateusername)
- [Oliver Sange](https://github.com/teammateusername)
## License
This project is licensed under the MIT License.

