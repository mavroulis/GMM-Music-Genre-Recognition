# GMM-Based Music Genre Recognition System

A Pattern Recognition system capable of distinguishing between **Blues, Reggae, and Classical** music genres with 100% accuracy. This project implements **Gaussian Mixture Models (GMM)** trained via a custom **Expectation-Maximization (EM)** algorithm, utilizing **Mel-Frequency Cepstral Coefficients (MFCC)** as feature vectors.

![Feature Space Visualization](assets/scatter_plot.png)
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
* `docs/`: Contains the detailed [Final Project Report](docs/Final_Report.pdf).

## 📊 Performance
The system was evaluated on a test set (unseen during training) with two model orders ($M$ = number of Gaussian components).

| Model Order (M) | Accuracy | Notes |
| :--- | :--- | :--- |
| **M = 8** | **100%** | Computationally efficient |
| **M = 16** | **100%** | Higher log-likelihood confidence |

### Visualization
As seen below, the spectral textures of the genres are distinct. Classical music (right) shows continuity, while Blues and Reggae exhibit rhythmic transient patterns.

![Heatmap Comparison](assets/heatmap.png)

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

## 📝 Theory
The probability density of a feature vector $x$ is modeled as:
$$p(x|\lambda) = \sum_{i=1}^{M} w_i g(x|\mu_i, \Sigma_i)$$

Where parameters $\lambda = \{w_i, \mu_i, \Sigma_i\}$ are estimated using the iterative EM algorithm:
1.  **E-Step:** Compute posterior probabilities.
2.  **M-Step:** Update weights, means, and variances.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
**Nikos Mavros** - *University of Thessaly*
