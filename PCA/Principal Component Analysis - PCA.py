#################################################################################################################################################
## Principal Component Analysis in 3 Simple Steps
## http://www.gladwinanalytics.com/blog/principal-component-analysis-in-3-simple-steps
## https://www.linkedin.com/pulse/understanding-pca-example-subhasree-chatterjee?trk=hp-feed-article-title-share
## http://ucanalytics.com/blogs/step-step-regression-models-pricing-case-study-example-part-5/
#################################################################################################################################################

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

##Loading the Dataset
df 		= pd.read_csv(filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',  header=None,  sep=',')
df.columns	=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

df.dropna(how="all", inplace=True) # drops the empty line at file-end
df.tail()

# split data table into data X and class labels y
X 		= df.ix[:,0:4].values
y 		= df.ix[:,4].values

## Exploratory Visualization
label_dict 	= {1: 'Iris-Setosa', 2: 'Iris-Versicolor', 3: 'Iris-Virgnica'}
feature_dict 	= {0: 'sepal length [cm]', 1: 'sepal width [cm]', 2: 'petal length [cm]', 3: 'petal width [cm]'}

## Not working
with plt.style.context('seaborn-whitegrid'):
		plt.figure(figsize=(8, 6))
		for cnt in range(4):
				plt.subplot(2, 2, cnt+1)
				for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
						plt.hist(X[y==lab, cnt], label=lab, bins=10, alpha=0.3,)
				plt.xlabel(feature_dict[cnt])
		plt.legend(loc='upper right', fancybox=True, fontsize=8)
		plt.tight_layout()
		plt.show()

###########################################################		
## Standardizing

	## Whether to standardize the data prior to a PCA on the covariance matrix depends on the measurement scales of the original features. 
	## Since PCA yields a feature subspace that maximizes the variance along the axes, it makes sense to standardize the data, especially, 
	## if it was measured on different scales. Although, all features in the Iris dataset were measured in centimeters, 
	## let us continue with the transformation of the data onto unit scale (mean=0 and variance=1), 
	## which is a requirement for the optimal performance of many machine learning algorithms.

X_std = StandardScaler().fit_transform(X)

###Eigenvectors and Eigenvalues: 
	##  The eigenvectors and eigenvalues of a covariance (or correlation) matrix represent the “core” of a PCA: 
	##	The eigenvectors (principal components) determine the directions of the new feature space, 
	##	and the eigenvalues determine their magnitude. In other words, the eigenvalues explain the variance of the data along the new feature axes.

##Method-1:	Covariance Matrix 
mean_vec 	= np.mean(X_std, axis=0)
cov_mat 	= (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

## Same as cov_mat
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

#eigendecomposition
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

##Method-2:	Correlation Matrix
cor_mat1 = np.corrcoef(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

##Method-3:	Eigendecomposition of the raw data based on the correlation matrix:
cor_mat2 = np.corrcoef(X.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat2)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

## We can clearly see that all three approaches yield the same eigenvectors and eigenvalue pairs:
	# - Eigendecomposition of the covariance matrix after standardizing the data.
	# - Eigendecomposition of the correlation matrix.
	# - Eigendecomposition of the correlation matrix after standardizing the data.

##Method-4:	Singular Vector Decomposition
	## 	While the eigendecomposition of the covariance or correlation matrix may be more intuitiuve, 
	## 	most PCA implementations perform a Singular Vector Decomposition (SVD) to improve the computational efficiency. 
	##	So, let us perform ##an SVD to confirm that the result are indeed the same:

u,s,v = np.linalg.svd(X_std.T)

## Selecting Principal Components

	for ev in eig_vecs:
		np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
	print('Everything ok!')

	## In order to decide which eigenvector(s) can dropped without losing too much information for the construction of lower-dimensional subspace, 
	## we need to inspect the corresponding eigenvalues: The eigenvectors with the lowest eigenvalues bear the least information about 
	## the distribution of the data; those are the ones can be dropped.
	## In order to do so, the common approach is to rank the eigenvalues from highest to lowest in order choose the top k eigenvectors.

	
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

## Explained Variance
	## After sorting the eigenpairs, the next question is “how many principal components are we going to choose for our new feature subspace?” 
	## A useful measure is the so-called “explained variance,” which can be calculated from the eigenvalues. 
	## The explained variance tells us how much information (variance) can be attributed to each of the principal components.

tot 		= sum(eig_vals)
var_exp 	= [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp 	= np.cumsum(var_exp)

## Not working
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.bar(range(4), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(4), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

############################################################################################################	
## Projection Matrix
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrix W:\n', matrix_w)

## Final data with two Principal Component
Y = X_std.dot(matrix_w)
############################################################################################################

## Not working
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),('blue', 'red', 'green')):
        plt.scatter(Y[y==lab, 0],Y[y==lab, 1],label=lab,c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()

############################################################	
## Shortcut - PCA in scikit-learn	
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca 	= sklearnPCA(n_components=2)

## Final data with two Principal Component
Y_sklearn 	= sklearn_pca.fit_transform(X_std)

## Not working
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),('blue', 'red', 'green')):
        plt.scatter(Y_sklearn[y==lab, 0],Y_sklearn[y==lab, 1],label=lab,c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pandas as pd
#from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

##Loading the Dataset
df = pd.read_csv(filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',  header=None,  sep=',')
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end
df.tail()

# split data table into data X and class labels y
X = df.ix[:,0:4].values
y = df.ix[:,4].values

## Exploratory Visualization
label_dict 		= {1: 'Iris-Setosa', 2: 'Iris-Versicolor', 3: 'Iris-Virgnica'}
feature_dict 	= {0: 'sepal length [cm]', 1: 'sepal width [cm]', 2: 'petal length [cm]', 3: 'petal width [cm]'}

## Standardizing
	## Whether to standardize the data prior to a PCA on the covariance matrix depends on the measurement scales of the original features. 
	## Since PCA yields a feature subspace that maximizes the variance along the axes, it makes sense to standardize the data, especially, 
	## if it was measured on different scales. Although, all features in the Iris dataset were measured in centimeters, 
	## let us continue with the transformation of the data onto unit scale (mean=0 and variance=1), 
	## which is a requirement for the optimal performance of many machine learning algorithms.

X_std = StandardScaler().fit_transform(X)

###Eigenvectors and Eigenvalues: 
	##  The eigenvectors and eigenvalues of a covariance (or correlation) matrix represent the “core” of a PCA: 
	##	The eigenvectors (principal components) determine the directions of the new feature space, 
	##	and the eigenvalues determine their magnitude. In other words, the eigenvalues explain the variance of the data along the new feature axes.

##Method-1:	Covariance Matrix 
mean_vec 	= np.mean(X_std, axis=0)
cov_mat 	= (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

## Same as cov_mat
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

#eigendecomposition
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

##Method-4:	Singular Vector Decomposition
	## 	While the eigendecomposition of the covariance or correlation matrix may be more intuitiuve, 
	## 	most PCA implementations perform a Singular Vector Decomposition (SVD) to improve the computational efficiency. 
	##	So, let us perform ##an SVD to confirm that the result are indeed the same:

## Selecting Principal Components
	## In order to decide which eigenvector(s) can dropped without losing too much information for the construction of lower-dimensional subspace, 
	## we need to inspect the corresponding eigenvalues: The eigenvectors with the lowest eigenvalues bear the least information about 
	## the distribution of the data; those are the ones can be dropped.
	## In order to do so, the common approach is to rank the eigenvalues from highest to lowest in order choose the top k eigenvectors.
	
## Explained Variance

	## After sorting the eigenpairs, the next question is “how many principal components are we going to choose for our new feature subspace?” 
	## A useful measure is the so-called “explained variance,” which can be calculated from the eigenvalues. 
	## The explained variance tells us how much information (variance) can be attributed to each of the principal components.

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()
	
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
	
## Projection Matrix
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrix W:\n', matrix_w)

## Final data with two Principal Component
Y = X_std.dot(matrix_w)

############################################################	
## Shortcut - PCA in scikit-learn	
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
## Final data with two Principal Component
Y_sklearn = sklearn_pca.fit_transform(X_std)

############################################################################################################