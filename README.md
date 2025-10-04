# Shadowfox Image Classifier

A PyTorch-based image classification project that uses a ResNet18 model to classify images from the CIFAR-10 dataset. This project demonstrates training, prediction, evaluation, and visualization of an image classifier.

## Features

- **Training**: Fine-tune a pretrained ResNet18 model on the CIFAR-10 dataset.
- **Prediction**: Classify individual images and output the predicted class.
- **Evaluation**: Assess model performance on the test set with accuracy, confusion matrix, and classification report.
- **Visualization**: Save predictions on random test images with overlaid labels.

## Installation

1. Ensure you have Python 3.7+ installed.
2. Clone this repository:
   ```
   git clone <repository-url>
   cd Shadowfox-imageclassifier
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the ResNet18 model on CIFAR-10:

```
python src/train.py --epochs 10 --batch_size 64 --data_dir ./data
```

- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 64)
- `--data_dir`: Directory to store the dataset (default: ./data)

The trained model will be saved to `checkpoints/resnet18_cifar10.pth`.

### Making Predictions

To predict the class of a single image:

```
python src/predict.py path/to/your/image.jpg
```

Example:
```
python src/predict.py test_dog1.jpg
```

This will output the predicted class (e.g., "Predicted Class: dog").

### Evaluating the Model

To evaluate the trained model on the CIFAR-10 test set:

```
python src/evaluate.py --data_dir ./data --batch_size 64 --checkpoint_path checkpoints/resnet18_cifar10.pth
```

This will print the accuracy, confusion matrix, and classification report.

### Saving Predictions

To save predictions on 10 random test images:

```
python src/save_predictions.py
```

The images with predicted labels will be saved in the `outputs/` directory.

## Dataset

This project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 6,000 images per class. The dataset is automatically downloaded when running the training or evaluation scripts.

## Model Architecture

- **Base Model**: ResNet18 pretrained on ImageNet
- **Modifications**: The final fully connected layer is replaced with a 10-class classifier
- **Input Size**: Images are resized to 224x224 pixels during preprocessing

## Project Structure

```
Shadowfox-imageclassifier/
├── src/
│   ├── data_loader.py      # Data loading and preprocessing utilities
│   ├── model.py            # Model building functions
│   ├── utils.py            # Utility functions (checkpoint saving, visualization)
│   ├── train.py            # Training script
│   ├── predict.py          # Prediction script
│   ├── evaluate.py         # Evaluation script
│   └── save_predictions.py # Script to save predictions on test images
├── data/                   # CIFAR-10 dataset (downloaded automatically)
├── checkpoints/            # Saved model checkpoints
├── outputs/                # Prediction visualizations
├── requirements.txt        # Python dependencies
├── test_dog1.jpg           # Sample test image
└── README.md               # This file
```

## Requirements

- torch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- tqdm
- opencv-python
- Pillow

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
