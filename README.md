# Task 1: Age Detection using VGGFace

## Description
This project fine-tunes a pre-trained VGGFace model on the IMDB-WIKI dataset to detect a person’s age.

## Structure
- `model_training.ipynb`: Training notebook
- `utils/data_loader.py`: Preprocessing script
- `saved_models/`: Contains model architecture and weights
- `results/`: Contains confusion matrix and classification report

## How to Run
1. Download the IMDB-WIKI dataset and extract it.
2. Update the paths in `model_training.ipynb`.
3. Run all cells in order.
4. Evaluate results from `results/`.

## Google Drive Link for Model Weights
IMDB-WIKI dataset
**[https://www.kaggle.com/datasets/abhikjha/imdb-wiki-faces-dataset]**

## Accuracy
Achieved MAE < 5 on test set. Accuracy is > 70% within ±5 years.

## Author
Ritwika Saha
