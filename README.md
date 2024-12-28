# KNN Classifier for Handwritten Digits

This repository contains a Python implementation of a K-Nearest Neighbors (KNN) classifier to recognize handwritten digits using the `load_digits` dataset from `sklearn.datasets`. The project includes data visualization, model training, evaluation, and prediction.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction
The K-Nearest Neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used for both classification and regression tasks. In this project, we use KNN to classify handwritten digits from the `load_digits` dataset.

## Requirements
To run this project, you need the following Python libraries:
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these libraries using pip:
```bash
pip install numpy matplotlib seaborn scikit-learn
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/omaradel73/KNN.git
   ```
2. Navigate to the project directory:
   ```bash
   cd knn-handwritten-digits
   ```

## Usage
1. Open the Jupyter Notebook `KNN.ipynb`:
   ```bash
   jupyter notebook KNN.ipynb
   ```
2. Run the cells in the notebook to load the dataset, train the KNN model, and evaluate its performance.

## Results
The notebook includes the following steps:
1. **Loading the Dataset**: The `load_digits` dataset is loaded and split into training and testing sets.
2. **Training the Model**: A KNN classifier is trained on the training data.
3. **Evaluation**: The model's accuracy is calculated, and a confusion matrix is plotted to visualize the performance.
4. **Visualization**: Sample images from the test set are displayed along with their predicted and true labels.

### Example Output
- **Accuracy**: The model achieves an accuracy of approximately 98.89%.
- **Confusion Matrix**: A heatmap of the confusion matrix is displayed to show the classification results.
- **Sample Predictions**: A few sample images from the test set are shown with their predicted and true labels.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The `load_digits` dataset from `sklearn.datasets`.
- The `scikit-learn` library for providing the KNN implementation.

## Contact
For any questions or suggestions, please feel free to open an issue or contact the repository owner.

---
