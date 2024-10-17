# facial-recognition-systems
This project explores facial recognition using PCA, LDA, LBP, and deep learning techniques, focusing on feature extraction, detection, and face verification systems.

![image](https://github.com/user-attachments/assets/265c2571-60f4-4460-9775-0fa5bd77ff1f)  
*Figure 1: Visualization of some data*

## Feature Extraction

### Eigenfaces for Face Recognition
The Eigenfaces method uses Principal Component Analysis (PCA) to reduce the dimensionality of facial images. This approach projects the images into a lower-dimensional space, enabling efficient face comparison. Eigenfaces are effective in scenarios where variations in lighting and facial expression need to be minimized, and they offer high recognition accuracy in controlled environments.

### Fisherfaces for Face Recognition
Fisherfaces use Linear Discriminant Analysis (LDA), a technique that seeks to maximize the separation between different facial identities while minimizing variation within a single identity. Fisherfaces offer better performance compared to Eigenfaces when dealing with varying lighting conditions and expressions, making them suitable for more realistic recognition scenarios.

### LBP for Face Recognition
Local Binary Patterns (LBP) is a texture-based feature descriptor that captures the local structure of an image by comparing each pixel with its surrounding neighborhood. It is particularly effective in face recognition because of its robustness to lighting variations and facial alignment. The LBP histograms generated for each face are compared using the chi-squared distance to determine similarity.

### Deep Metric Learning
Deep metric learning involves training neural networks to learn an embedding space where facial images of the same person are mapped closely together, and images of different people are far apart. This is often done using loss functions like contrastive or triplet loss. Deep learning methods typically outperform traditional approaches, especially in large datasets or unconstrained environments.

## Evaluation

### Validation as Verification System

#### Verification Setup
The verification setup in biometric systems uses two datasets: a reference set with 122 images and a verification set with 318 images. The reference set contains accurate baseline biometric samples for each registered individual. In face recognition tasks, models like PCA, LDA, and LBP are trained on the reference data and applied to the verification data. The similarity between reference and verification images is captured in a distance matrix (e.g., a 120 by 320 matrix), enabling accurate identity verification.

#### Distance Matrix
The verification process in this project uses a **120 by 320 distance matrix**, where 120 verification images are compared against 320 reference images. The reference set was used to train models like PCA (Eigenfaces), and the distance matrix captures the similarity between the verification and reference images.

<img src="https://github.com/user-attachments/assets/e66b8755-f2ae-4fe4-be7f-e59b1d41f962" width="50%" height="50%">  
*Figure 2: Distance Matrix Setup*

- **120 verification set**  
- **320 reference set** → used for training PCA (eigenvectors)
- Matrix normalized for better performance
- Computation of **True Positives (TP)**, **False Positives (FP)**, and **False Negatives (FN)**

#### Histogram of Genuine and Impostor Scores
This project evaluated four feature extraction methods: PCA, LDA, LBP, and deep learning. For each method, a histogram of genuine and impostor scores was created to analyze the overlap between the two groups.

- **LDA**: Most effective method with minimal overlap between genuine and impostor scores, achieving distinct peak separation.
- **DL (Deep Learning)**: Performs well, though with slightly more overlap than LDA.
- **LBP and PCA**: Show the most overlap, making them less effective at separating genuine and impostor scores.

![image](https://github.com/user-attachments/assets/5e5f5618-9d0a-4f46-9a5d-19c1cc295f47)  
*Figure 3: Histograms of Genuine and Impostor Scores*

#### F1 Across Thresholds and Accuracy Across Thresholds
The F1 score and accuracy were computed across varying thresholds for each system:

- **PCA**: Achieved the highest F1 score peak and highest accuracy across most thresholds.
- **LDA**: Second highest F1 score and accuracy.
- **DL and LBP**: Lower F1 and accuracy scores compared to PCA and LDA.

Optimal F1 thresholds for each system:  
- PCA = 0.202  
- LDA = 0.182  
- DL = 0.111  
- LBP = 0.111  

<img src="https://github.com/user-attachments/assets/c70a0892-0c3b-4f88-9f01-20a5b541e3d1" width="70%" height="70%">  
*Figure 4: F1 and Accuracy across Thresholds*

#### Precision-Recall for Multiple Systems
Precision-recall curves compare the trade-off between precision and recall across systems:

- **DL**: Best performance, maintains high precision across varying recall levels.
- **LDA**: Slightly lower performance than DL, but better than PCA and LBP.
- **PCA and LBP**: Similar lower performance, with precision dropping as recall increases.

<img src="https://github.com/user-attachments/assets/320cb634-e1e8-490f-9269-dc245e0a587e" width="70%" height="70%">  
*Figure 5: Precision-Recall Curves for Multiple Systems*

### Validation as Identification System

#### Identification Setup
The identification process aims to determine the identity of an unknown individual by comparing their biometric data against a database of known entities in a 1-to-N matching scenario. A train-test split is used, where the training data help to build a database of known identities, and test data consist of new samples to evaluate the system’s ability to identify unknown individuals. For example, in airport security systems, a traveler’s face can be scanned and matched against a database of known identities.

#### Cumulative Matching Characteristic (CMC) Curve
The CMC curve shows how well the system identifies an individual as you increase the number of candidates considered in the rank list. The steepest CMC curves indicate superior performance.

- **LDA and DL**: Steepest curves, indicating near-perfect identification rates early in the rank list.
- **PCA and LBP**: Gradual rise in CMC curves, requiring more candidates to achieve similar identification rates as LDA and DL.

<img src="https://github.com/user-attachments/assets/f1f06c2a-27d3-47c8-a434-a35fda7a893f" width="70%" height="70%">  
*Figure 6: Cumulative Matching Characteristic (CMC) Curves*

- **Rank-1 Performance:**
  - PCA = 0.8583
  - LDA = 0.9417
  - DL = 0.8917
  - LBP = 0.8583
