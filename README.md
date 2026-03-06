# E-Commerce Clothing Classification Project

## Overview
This project implements deep learning models for classifying men's clothing items from e-commerce images. The notebook compares two approaches:
1. **Custom CNN**: A custom-built convolutional neural network trained from scratch
2. **ResNet50 Transfer Learning**: Pre-trained ResNet50 model fine-tuned for clothing classification

**Project Type**: Kaggle Dataset Project

## Dataset
- **Source**: [Kaggle - E-Commerce Men's Clothing Dataset](https://www.kaggle.com/datasets/prashantsharma526/e-commerce-mens-clothing-dataset)
- **Classes**: 8 clothing categories (target categories from the dataset)
- **Image Size**: 224x224 pixels (resized)
- **Train/Val/Test Split**: 80% / 10% / 10%
- **Note**: Dataset not included in this repository. Download from the Kaggle link above.

## Project Structure

### Sections in Notebook

1. **Libraries & Setup**
   - Import all required packages
   - Check library versions
   - Setup GPU/CPU device

2. **Data Loading & Preprocessing**
   - Load images from ImageFolder structure
   - Custom RGB conversion for grayscale images
   - Compute mean and standard deviation for normalization
   - Apply transformations (resize, normalize, convert to tensor)

3. **Exploratory Data Analysis**
   - Visualize sample images from each class
   - Analyze class distribution in training/validation sets
   - Create balanced train/val/test splits

4. **Custom CNN Model**
   - 3 convolutional blocks (16 → 32 → 64 filters)
   - 2 fully connected layers with dropout (0.5)
   - Train for 10 epochs with validation monitoring
   - Evaluate on test set

5. **ResNet50 Transfer Learning**
   - Load pre-trained ResNet50 from ImageNet
   - Freeze convolutional layers (feature extraction)
   - Replace final layer with 8-class classifier
   - Fine-tune final layer for 10 epochs
   - Generate confusion matrix and training curves

6. **Model Comparison**
   - Compare test accuracy between CNN and ResNet50
   - Display performance improvement percentage

## Model Details

### Custom CNN Architecture
```
Conv2d(3→16, 3×3) → ReLU → MaxPool(2)
Conv2d(16→32, 3×3) → ReLU → MaxPool(2)
Conv2d(32→64, 3×3) → ReLU → MaxPool(2)
Flatten
Linear(64×28×28→128) + Dropout(0.5) → ReLU
Linear(128→8)
```

### ResNet50 Transfer Learning
- Pre-trained encoder: ImageNet weights (frozen)
- Custom classifier: Linear(2048→8)
- Only final layer trainable
- Learns to adapt features to clothing classification task

## Hyperparameters
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 10
- **Dropout**: 0.5 (for custom CNN)

## Key Features
- ✅ Automatic GPU/CPU device detection and switching
- ✅ Data normalization using computed mean/std
- ✅ Validation monitoring during training
- ✅ Comprehensive metrics (loss, accuracy)
- ✅ Confusion matrix visualization
- ✅ Training curves (loss and accuracy plots)
- ✅ Model comparison summary

## Results
The notebook outputs:
- Training and validation metrics for both models
- Test set accuracy and loss
- Confusion matrices for error analysis
- Performance comparison and improvement percentage
- Training curves showing convergence behavior

## Dependencies
See `requirements.txt` for exact package versions.

Main packages:
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- numpy >= 1.19.0
- pandas >= 1.1.0
- PIL >= 8.0.0
- tqdm >= 4.50.0
- torchinfo >= 1.5.0

## Usage

### Running the Notebook
1. Ensure you have the clothing dataset in the correct location (/kaggle/input/e-commerce-mens-clothing-dataset/dataset_clean/)
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook in Jupyter: `jupyter notebook notebook-dress-classification.ipynb`
4. Run cells sequentially from top to bottom

### Expected Output
- Library version information
- Class list and distribution charts
- Sample images from dataset
- Training progress for both models (loss/accuracy per epoch)
- Test set results
- Confusion matrices
- Performance comparison

## Notes
- GPU recommended for faster training (automatic detection included)
- First run will compute dataset mean/std for normalization
- ResNet50 download (~100MB) happens on first load
- Training takes ~5-15 minutes depending on hardware (CPU/GPU)

## Author
Created as part of e-commerce image classification project

## License
Dataset source: Kaggle E-commerce Men's Clothing Dataset
