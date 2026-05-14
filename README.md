# Who's That Pokémon? A Multi-Model Image Classifier
---

## Author: Danush Sinnan

## 📌 Project Overview

This project applies the machine learning skills covered in the **NYU Data Bootcamp** course to build a **Pokémon image classifier** inspired by the classic *Who's That Pokémon?* segment from the TV show. The goal is to correctly identify a Generation 1 Pokémon from an image across 150 possible classes.

The project builds progressively from a trivial baseline through classical ML with dimensionality reduction to a pretrained convolutional neural network, with each step motivated by the limitations of the previous one.

## 🗂️ Repository Structure

```
├── Danush_Sinnan_Data_Bootcamp_Pokemon_Final_Project.ipynb        # Main Jupyter Notebook (full project code)
├── Danush_Sinnan_Data_Bootcamp_Pokemon_Final_Project_Writeup.docx # Full write-up
├── README.md
├── requirements.txt
```

---

## 📁 Dataset

This project uses the [7,000 Labeled Pokémon](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) dataset from Kaggle.

| Detail | Value |
|---|---|
| Source | Kaggle (lantian773030/pokemonclassification) |
| Images | ~6,820 JPEG/PNG images |
| Classes | 150 Generation 1 Pokémon |
| Images per class | 26–66 (mean: 45.5) |

The dataset is downloaded automatically via the Kaggle API in the notebook. See **Getting Started** below for setup instructions.

---

## 🤖 Models Compared

| Model | Approach | Test Accuracy |
|---|---|---|
| Dummy Classifier | Baseline | 0.95% |
| KNN on Raw Pixels | Naive pixel-distance | 16.50% |
| KNN + PCA | Classical ML + dimensionality reduction | 17.67% |
| Logistic Regression + PCA | Classical ML + dimensionality reduction | 33.43% |
| SVC + PCA | Classical ML + dimensionality reduction | 39.88% |
| ResNet18 (pretrained CNN) | Transfer learning | 36.36% |

**Key finding:** SVC with an RBF kernel on PCA-compressed features outperformed the pretrained CNN (39.88% vs. 36.36%), explained by the small dataset size, domain mismatch between ImageNet photographs and Pokémon cartoon artwork, and limited CNN training time.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/DSin145/DATA_BOOTCAMP_FINAL_PROJECT.git
cd DATA_BOOTCAMP_FINAL_PROJECT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle API access

The notebook downloads the dataset automatically using the Kaggle API. To enable this:

1. Go to [kaggle.com](https://www.kaggle.com) → Settings → API → **Create New Token**
2. In the notebook's Step 2 cell, replace `'YOUR_KAGGLE_TOKEN_HERE'` with your token:
   ```python
   os.environ['KAGGLE_API_TOKEN'] = 'YOUR_KAGGLE_TOKEN_HERE'
   ```

### 4. Enable GPU (recommended)

The CNN section (Step 8) runs significantly faster on a GPU. In Google Colab, go to:
**Runtime → Change Runtime Type → T4 GPU**

### 5. Run the notebook

Open in **Google Colab** by uploading to your Drive, or run locally:

```bash
jupyter notebook Danush_Sinnan_Data_Bootcamp_Pokemon_Final_Project.ipynb
```

---

## 📈 Executive Summary

### Overview

This project builds a multi-model image classifier to identify Generation 1 Pokémon from images, framed around the classic *Who's That Pokémon?* TV show segment. Six models were implemented and evaluated, ranging from a simple dummy baseline to a pretrained convolutional neural network, with the goal of understanding when classical machine learning outperforms deep learning and why.

### Data

The image dataset contains approximately 6,820 labeled JPEG and PNG images sourced from Kaggle, organized into 150 subfolders corresponding to each Generation 1 Pokémon. The dataset is mildly imbalanced, with class sizes ranging from 26 to 66 images and a mean of 45.5. All images were standardized to 64x64 pixels, converted to RGB, and normalized to [0, 1] before modeling. An 80/20 stratified train/test split was applied, yielding 5,456 training images and 1,364 test images.

### Methods

All classical models operated on PCA-compressed pixel features, following a suggestion from Professor Koehler. Raw pixel arrays (12,288 features per image) were scaled with StandardScaler, then compressed to 710 principal components retaining 95% of explained variance, a 17.3x reduction. Three classical classifiers were then trained on these compressed features: KNN, Logistic Regression, and SVC with an RBF kernel. The deep learning model used ResNet18 pretrained on ImageNet, with all layers frozen except a new 150-class output head trained for 10 epochs using CrossEntropyLoss and SGD.

### Results

| Model | Test Accuracy |
|---|---|
| SVC + PCA | 39.88% |
| ResNet18 (CNN) | 36.36% |
| Logistic Regression + PCA | 33.43% |
| KNN + PCA | 17.67% |
| KNN (raw pixels) | 16.50% |
| Dummy Classifier | 0.95% |

SVC with an RBF kernel was the strongest model overall, outperforming ResNet18 despite the neural network using pretrained ImageNet weights. The CNN underperformed due to a combination of factors: only the final linear layer was retrained, 10 epochs of unscheduled SGD may not have been sufficient for convergence, ImageNet filters trained on real photographs do not transfer cleanly to cartoon artwork, and the 64x64 input resolution is well below ResNet's designed 224x224. SVC's maximum-margin formulation is specifically well-suited to high-dimensional, small-sample problems, which describes this dataset well.

### Conclusion and Next Steps

Classical machine learning with thoughtful feature engineering can outperform transfer learning under the right conditions. The most impactful extensions to this project would be upsampling images to 224x224, running more epochs with a learning rate scheduler, adding data augmentation to reduce overfitting, unfreezing deeper ResNet layers for full fine-tuning, and deploying the best model as a Streamlit app. These changes would give a fairer comparison between SVC and deep learning and would likely reverse the accuracy ranking in favor of the CNN.

For the full write-up including figures, confusion matrices, and detailed per-model analysis, see `Danush_Sinnan_Data_Bootcamp_Pokemon_Final_Project_Writeup.docx`.
