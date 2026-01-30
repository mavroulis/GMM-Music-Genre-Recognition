# GMM-Based Music Genre Recognition System🎶

**Course:** ECE443 Speech and Audio Processing, university of thessaly 
**Author:** Nikos Mavros

![Matlab](https://img.shields.io/badge/matlab%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)


A Pattern Recognition system capable of distinguishing between **Blues, Reggae, and Classical** music genres with 100% accuracy. This project implements **Gaussian Mixture Models (GMM)** trained via a custom **Expectation-Maximization (EM)** algorithm, utilizing **Mel-Frequency Cepstral Coefficients (MFCC)** as feature vectors.

![Feature Space Visualization](assets/feature_scatter.png)
*Figure: 2D Projection of the Feature Space showing clear clustering of genres.*

## 🚀 Key Features
* **Feature Extraction:** 20ms windowing with MFCC extraction.
* **Preprocessing:** Implements **Cepstral Mean Subtraction (CMS)** and Energy coefficient removal for channel robustness.
* **Custom Machine Learning:**
    * **Not a toolbox wrapper:** The Expectation-Maximization (EM) algorithm is manually implemented for GMM training.
    * **Initialization:** Uses K-Means clustering to robustly initialize GMM parameters.
    * **Classification:** Uses Maximum Likelihood / Maximum A Posteriori (MAP) estimation.
* **Visualization:** Includes tools to generate spectral heatmaps and feature scatter plots.

## 📂 Project Structure
* `src/main.m`: Core pipeline. Extracts features, trains models, and runs the classification test.
* `src/create_graphs.m`: Generates visualizations of the feature space.
* `docs/`: Contains the detailed [Final Project Report](docs/report.pdf).
* `assets/`: Contains all the plots was generated in the `src/create_graphs.m`.
* `model/`: Contains all the matrixies that were created in the `src/main.m`.

## 📊 Performance
The system was evaluated on a test set (unseen during training) with two model orders ($M$ = number of Gaussian components).

| Model Order (M) | Accuracy | Notes |
| :--- | :--- | :--- |
| **M = 8** | **100%** | Computationally efficient |
| **M = 16** | **100%** | Higher log-likelihood confidence |

### Visualization
As seen below, the spectral textures of the genres are distinct. Classical music (right) shows continuity, while Blues and Reggae exhibit rhythmic transient patterns.

![Heatmap Comparison](assets/mfcc_comparison.png)

## 🛠️ Installation & Usage

### Prerequisites
* MATLAB (Signal Processing Toolbox recommended for `mfcc` function)

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/GMM-Music-Genre-Recognition.git](https://github.com/yourusername/GMM-Music-Genre-Recognition.git)
    ```
2.  Add your audio dataset to a `Data/` folder in the root directory:
    * `Data/Train/blues/`, `Data/Train/reggae/`, etc.
    * `Data/Test/blues/`, `Data/Test/reggae/`, etc.
3.  Run the main script in MATLAB:
    ```matlab
    >> main
    ```
