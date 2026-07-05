# 🚀 Dynamic Boundary K-Means

An enhanced implementation of the **K-Means clustering algorithm** that improves cluster quality by dynamically detecting and reassigning ambiguous boundary points while automatically selecting the optimal number of clusters.

Unlike the classical K-Means algorithm, this approach performs a local optimization step that evaluates whether moving ambiguous points to their second-closest cluster reduces the local clustering error.

---

# 📌 Overview

Traditional K-Means assigns every sample directly to its nearest centroid. However, samples located near the boundary between two clusters are often difficult to classify correctly.

This project introduces a **Dynamic Boundary Point Strategy** that:

* Detects ambiguous samples
* Evaluates possible reassignment
* Accepts only beneficial moves
* Improves cluster compactness
* Preserves the simplicity of K-Means

The algorithm was evaluated using the **Mall Customers** dataset.

---

# ✨ Features

* ✅ Classical K-Means implementation
* ✅ Dynamic boundary point detection
* ✅ Local reassignment optimization
* ✅ Automatic K selection using Silhouette Score
* ✅ Multiple random runs
* ✅ Cluster quality evaluation
* ✅ Performance visualization

---

# 📊 Dataset

* **Dataset:** Mall Customers
* **Samples:** 200
* **Features:** 5
* **Preprocessing**

  * Remove CustomerID
  * One-hot encoding
  * Feature scaling

---

# ⚙️ Algorithm

The improved algorithm follows these steps:

1. Initialize K-Means.
2. Assign each sample to its nearest centroid.
3. Compute the distance to the two closest centroids.
4. Detect ambiguous boundary points.
5. Apply a dynamic ambiguity threshold.
6. Test local reassignment.
7. Accept the reassignment only if the local SSE decreases.
8. Recompute centroids.
9. Repeat until convergence.

The optimal number of clusters is selected automatically by maximizing the Silhouette Score.

---

# 📈 Performance Comparison

| Metric                 | Classical K-Means | Dynamic Boundary K-Means |
| ---------------------- | ----------------: | -----------------------: |
| K                      |                 2 |                       10 |
| Inertia                |            597.95 |                   152.59 |
| Silhouette             |            0.3964 |                   0.4383 |
| Davies-Bouldin         |            1.1425 |                   0.8160 |
| Calinski-Harabasz      |            133.13 |                   117.24 |
| Intra-cluster Distance |            1.6185 |                   0.7990 |
| Inter-cluster Distance |            2.8563 |                   3.0250 |
| Inter/Intra Ratio      |            1.7648 |                   3.7858 |
| Execution Time         |           4.018 s |                  0.952 s |

---

# 🏆 Results

Compared with the classical K-Means algorithm, the proposed approach achieved:

* 🚀 **74.48% lower inertia**
* 🚀 **10.55% higher Silhouette Score**
* 🚀 **28.58% lower Davies-Bouldin Index**
* 🚀 **50.63% lower intra-cluster distance**
* 🚀 **114.52% higher inter/intra distance ratio**
* 🚀 **76.31% faster execution time**

During optimization, the algorithm:

* Processed **292** boundary points
* Accepted **64** local reassignments

These improvements produced more compact and better-separated clusters while maintaining the original simplicity of K-Means.

---

# 📊 Evaluation Metrics

The implementation evaluates clustering quality using:

* Inertia
* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Index
* Intra-cluster Distance
* Inter-cluster Distance
* Cluster Balance

---

# 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

# 🚀 Future Improvements

* K-Means++ initialization
* GPU acceleration
* Parallel boundary optimization
* Support for high-dimensional datasets
* Interactive visualization dashboard
* Benchmark against DBSCAN and Gaussian Mixture Models
