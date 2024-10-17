# facial-recognition-systems
This project explores facial recognition using PCA, LDA, LBP, and deep learning techniques, focusing on feature extraction, detection, and face verification systems.

![image](https://github.com/user-attachments/assets/265c2571-60f4-4460-9775-0fa5bd77ff1f)  

*Figure 1: Visualization of some data*

## Feature Extraction

### Feature Extraction

#### Eigenfaces for Face Recognition

The Eigenfaces method uses Principal Component Analysis (PCA) to reduce the dimensionality of facial images. Introduced by Kirby and Sirovich in 1987, it represents faces as linear combinations of eigenvectors derived from the face dataset, known as eigenfaces. This method involves collecting a dataset of aligned face images, applying an eigenvalue decomposition, and keeping the eigenvectors with the largest corresponding eigenvalues. Recognition is performed by projecting new face images into this eigenface space and comparing them using distance metrics like Euclidean distance, often treating face identification as a k-Nearest Neighbor classification problem. Although initially designed for face recognition, the Eigenfaces algorithm can be applied to any object classification task involving similar datasets, such as identifying bicycles, cans of soup, or ancient coins.

#### Fisherfaces for Face Recognition

Fisherfaces employ Linear Discriminant Analysis (LDA) to find a subspace that maximizes the separation between different classes while minimizing variation within the same class. Based on R.A. Fisher's work from 1936, this method generates basis vectors called Fisherfaces. Unlike PCA, which focuses on representation, LDA aims at classification by finding a subspace that maps sample vectors of the same class to a single spot in the feature representation while separating different classes as much as possible. Fisherfaces are particularly effective for classification tasks like face recognition under varying lighting conditions and expressions, as they enhance class separability. This approach often outperforms Eigenfaces in more realistic recognition scenarios where lighting and facial expressions vary.

#### LBP for Face Recognition

Local Binary Patterns (LBP) are texture descriptors that capture local image structure by comparing each pixel with its neighboring pixels to form a binary code. Introduced by Ojala et al. in their 2002 paper, LBPs provide a local representation of texture that is robust to lighting variations and facial misalignments. Unlike global texture descriptors like Haralick texture features, LBPs compute a local representation by comparing each pixel with its surrounding neighborhood. In face recognition, images are represented as histograms of LBP codes, which are then compared using distance metrics like chi-squared distance to determine similarity. The LBP method is particularly effective due to its ability to capture fine details and its relative invariance to monotonic gray-level changes caused by illumination variations.

#### Deep Metric Learning

Deep Metric Learning trains deep neural networks to map facial images into a feature space where images of the same person are close together, and those of different people are far apart. Key approaches include:

1. DeepFace (2014): Developed by Facebook, it uses a Convolutional Neural Network (CNN) combined with a 3D alignment process. Trained on a large dataset of 4.4 million images of 4,000 individuals, it learns embedded representations that generalize well to other datasets. DeepFace achieved over 97% accuracy on the Labeled Faces in the Wild (LFW) dataset, approaching human-level performance.

2. FaceNet/OpenFace: Developed by Google, it utilizes a triplet loss function during training. The triplet loss ensures that an anchor image is closer to a positive image (same identity) than to a negative image (different identity) by at least a specified margin. This method achieves high accuracy on benchmarks like the LFW dataset. FaceNet was trained on an enormous dataset of 200 million face images of about 8 million different identities.

3. Siamese Networks: These networks consist of two identical subnetworks with shared weights that process pairs of images. Trained using contrastive loss, they minimize the distance between embeddings of the same class and maximize the distance between different classes. Siamese networks are particularly effective for one-shot learning scenarios with few samples per class or dynamically changing numbers of subjects.

Implementations like Dlib's face_recognition module use these deep metric learning techniques to map faces into a 128-dimensional embedding space. When using a distance threshold of 0.6, the Dlib model achieves 99.38% accuracy on the LFW benchmark, comparable to other state-of-the-art methods as of February 2017. This approach allows for highly accurate face recognition and verification tasks.


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


### Equal Error Rate (EER) Results

The table below shows the Equal Error Rate (EER) values and corresponding thresholds for the four methods (PCA, LDA, DL, LBP) used in the analysis. The EER is the point where the False Acceptance Rate (FAR) and False Rejection Rate (FRR) are equal, which is a critical metric for evaluating the performance of biometric or verification systems. A lower EER indicates better performance as it represents a lower overall error.

| Method | EER  | Threshold |
|--------|------|-----------|
| PCA    | 0.2538 | 0.3838    |
| LDA    | 0.0466 | 0.3131    |
| DL     | 0.0433 | 0.2323    |
| LBP    | 0.2358 | 0.2424    |

From the table, it is evident that the **DL (Deep Learning)** method achieves the lowest EER of **0.0433**, indicating the best performance among the four methods. The second-best result is achieved by **LDA (Linear Discriminant Analysis)** with an EER of **0.0466**. Both PCA and LBP have significantly higher EERs, with PCA having an EER of **0.2538** and LBP an EER of **0.2358**, suggesting these methods are less effective in minimizing error.

In summary, **DL** demonstrates the highest accuracy for this biometric verification task, followed closely by **LDA**. These results show that more advanced techniques, such as deep learning, are better suited for reducing errors in biometric systems compared to traditional methods like PCA and LBP.



### Validation as Identification System

#### Identification Setup
The identification process aims to determine the identity of an unknown individual by comparing their biometric data against a database of known entities in a 1-to-N matching scenario. A train-test split is used, where the training data help to build a database of known identities, and test data consist of new samples to evaluate the system’s ability to identify unknown individuals. For example, in airport security systems, a traveler’s face can be scanned and matched against a database of known identities.

#### Cumulative Matching Characteristic (CMC) Curve
The CMC curve shows how well the system identifies an individual as you increase the number of candidates considered in the rank list. The steepest CMC curves indicate superior performance.

- **LDA and DL**: Steepest curves, indicating near-perfect identification rates early in the rank list.
- **PCA and LBP**: Gradual rise in CMC curves, requiring more candidates to achieve similar identification rates as LDA and DL.

<img src="https://github.com/user-attachments/assets/c2df5051-7f52-40a8-aa3b-c3ddf67b1fbf" alt="image" width="400"/>

*Figure 6: Cumulative Matching Characteristic (CMC) Curves*

- **Rank-1 Performance:**
  - PCA = 0.8583
  - LDA = 0.9417
  - DL = 0.8917
  - LBP = 0.8583
