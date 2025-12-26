📌 Milestone 4 – Deep Learning Model & Explainability
📖 Overview

Milestone 4 focuses on implementing a Deep Learning–based scanner identification system using a Convolutional Neural Network (CNN) and integrating explainability techniques to understand model behavior.

This milestone validates the deep learning pipeline, data augmentation strategy, evaluation metrics, and explainability workflow using Grad-CAM style visualizations.

🎯 Objectives

Build and validate a CNN model for scanner identification

Apply image augmentation to improve generalization

Track training behavior using accuracy curves

Evaluate model performance using confusion matrix and metrics

Implement explainability to visualize scanner-specific patterns

🗂️ Folder Structure
Milestone 4/
│
├─ images/                         # Raw scanned images
│
├─ cnn/
│   └─ train_cnn.py                # CNN training pipeline
│
├─ explainability/
│   └─ gradcam.py                  # Grad-CAM explainability (safe mode)
│
├─ models/
│   └─ cnn_model.h5                # Trained CNN model
│
├─ results/
│   ├─ training_curves.png         # CNN accuracy curve
│   ├─ confusion_matrix.png        # CNN evaluation
│   └─ gradcam_outputs/            # Explainability heatmaps
│
├─ labels.csv                      # Image paths and scanner labels
└─ README.md

🧠 CNN Model Description

The CNN model is designed to automatically learn scanner-specific patterns directly from raw scanned images.

Key Components:

Convolutional layers for feature extraction

Max-Pooling layers for spatial reduction

Fully connected layers for classification

Dropout for regularization

Data Augmentation Used:

Rotation

Brightness adjustment

Horizontal flipping

These augmentations help improve robustness and generalization.

📊 Model Evaluation

The CNN model evaluation includes:

Training accuracy visualization

Confusion matrix for class-level analysis

Classification readiness assessment

Due to limited dataset size, the pipeline focuses on architecture validation and scalability rather than absolute accuracy.

🔍 Explainability (Grad-CAM)

Grad-CAM-based explainability is implemented to visualize image regions contributing to scanner identification.

Explainability Highlights:

Heatmap overlays on original images

Visualization of scanner-specific activation regions

Safe explainability mode used due to limited training samples

The explainability framework is fully scalable and supports gradient-based visualization when trained on larger datasets.

▶️ How to Run (Windows)
1️⃣ Navigate to Milestone 4
cd "D:\Infosys Intern\Tracefinder\Milestone 4"

2️⃣ Train CNN model
python cnn\train_cnn.py


✔ Output:

models/cnn_model.h5
results/training_curves.png
results/confusion_matrix.png

3️⃣ Run Explainability
python explainability\gradcam.py


✔ Output:

results/gradcam_outputs/

📁 Outputs Generated

cnn_model.h5 – Trained CNN model

training_curves.png – Accuracy trends

confusion_matrix.png – Model evaluation

gradcam_outputs/ – Explainability heatmaps

📝 Milestone 4 Conclusion

In this milestone, a convolutional neural network was implemented to learn scanner-specific features directly from raw scanned images.
Data augmentation techniques were applied to improve generalization.
Explainability using Grad-CAM was integrated to visualize important regions influencing scanner identification.
The deep learning pipeline was successfully validated and is scalable for future datasets.

🎓 Viva-Ready Notes

CNN automatically learns scanner artifacts without handcrafted features

Data augmentation improves robustness

Grad-CAM provides interpretability and transparency

Limited dataset handled using safe validation strategy

✅ Status

✔ CNN architecture implemented
✔ Training pipeline validated
✔ Explainability workflow completed
✔ Milestone 4 successfully completed

🚀 Next Steps

Expand dataset with multiple scanners

Train CNN fully and optimize hyperparameters

Apply full gradient-based Grad-CAM analysis

Integrate end-to-end system with user interface