# Principal Component Analysis (PCA)

- PCA is a technique used to simplify complex datasets by reducing the number of variables while preserving the most important information.
- PCA is an unsupervised technique.
- PCA Works for Dimensionality Reduction.
- PCAs are nothing but eigenvectors.
- Step-by-step:
    - Ensures all features are on the same scale, preventing features with larger values from dominating the analysis. Subtract the mean and divide by the standard deviation for each feature.
    - Measures the relationship between different features. Calculate the covariance between each pair of features in the standardized dataset.
    - Eigenvectors and eigenvalues to compute from the covariance matrix in order to determine the principal components of the data.

- Eigenvalues represent the total variance that can be explained by a given principal component. Larger eigenvalues indicate that the corresponding principal component captures more of the data's variability.
- PCAs are eigenvectors of the data (covariance matrix) which represent the directions of maximum variance.
- By focusing on the principal components with the largest eigenvalues, PCA can reduce the number of features while minimizing information loss. This is because the principal components capture the most significant patterns in the data.
- Eigenvalues are used in PCA to quantify the amount of variance explained by each principal component.
- The corresponding eigenvalues indicate the magnitude of this variance.
- By selecting the eigenvectors associated with the largest eigenvalues, PCA effectively reduces dimensionality while retaining the most important information.
- Example: https://github.com/amitmse/in_Python_/blob/master/PCA/Example.xlsx

        https://medium.com/analytics-vidhya/understanding-principle-component-analysis-pca-step-by-step-e7a4bb4031d9
