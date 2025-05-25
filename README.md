# Detecting Nutritional Deficiency in Plants Using CNN

This project implements a Convolutional Neural Network (CNN) based system integrated with fuzzy logic to automatically detect nutritional deficiencies in plants from images. The solution is tailored for precision agriculture, enabling early diagnosis and effective nutrient management.

---

## Abstract

The health of plants is crucial for sustainable agriculture. Nutritional deficiencies can severely reduce crop yield and quality. Manual observation is slow and error-prone. This project uses CNNs to automate detection of deficiencies through image analysis and integrates fuzzy logic for interpretable predictions. The system is capable of detecting and classifying the severity of deficiencies as "low," "medium," or "high," aiding farmers in timely intervention.

---

## Objectives

* **Image Acquisition**: Gather images from publicly available datasets (e.g., Kaggle).
* **Preprocessing**: Resize, crop, and normalize images.
* **Segmentation**: Focus on relevant parts (e.g., leaves).
* **Feature Extraction**: Use CNN (VGG16) for automatic extraction.
* **Classification**: Label images as healthy or deficient.
* **Fuzzy Logic Integration**: Add interpretability and fine-tune CNN output.

---

## Dataset

[Plant Disease dataset](https://www.kaggle.com/datasets/saroz014/plant-disease)

---

## Methodology

1. **Data Acquisition**: Leaf images representing various deficiencies.
2. **Preprocessing**: Standardization of images in RGB space.
3. **Feature Extraction**: Fine-tuning pre-trained CNN (e.g., VGG16).
4. **Fuzzy Logic Module**:

   * Membership functions: low, medium, high
   * Rules: Confidence → Nutrient deficiency severity
5. **Integration**: Combine CNN prediction confidence with fuzzy logic to classify severity.
6. **Evaluation**: Confusion matrix, precision, recall, F1-score.

---

## Technologies Used

* **Languages**: Python
* **Libraries**: TensorFlow/Keras, NumPy, Matplotlib, scikit-fuzzy
* **Model**: Pre-trained VGG16
* **Fuzzy Inference**: Mamdani-style rule system

---

## Performance

| Class                   | Precision | Recall | F1-Score | Support |
| ----------------------- | --------- | ------ | -------- | ------- |
| cherry (including sour) | 1.00      | 1.00   | 1.00     | 190     |

* **Accuracy**: 100%
* CNN model + fuzzy logic achieved perfect results on the test dataset.

---

## Integration of CNN and Fuzzy Logic

* CNN outputs confidence levels for classes.
* Fuzzy logic interprets these confidences:

  * Confidence < 0.5 → Low nutrient deficiency
  * 0.5 ≤ Confidence ≤ 0.75 → Medium
  * > 0.75 → High deficiency

---

## Test Strategy

* Tested on individual images and the entire dataset.
* Visual comparison between predicted and actual class labels.
* Graphs show refined predictions vs. true labels.

---

## Confusion Matrix & Evaluation

* Plotted using Seaborn heatmaps.
* True Positives, False Positives, False Negatives, and True Negatives calculated.

---

## Results and Discussion

* The CNN demonstrated strong learning and generalization.
* Fuzzy logic added nuanced classification.
* Results were consistently high in all metrics.
* Graphs and accuracy plots validate the model.

---
