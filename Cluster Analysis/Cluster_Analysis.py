###############################################################################
################ K-Means Cluster Analysis #####################################
###############################################################################


###############################################################################
#FeatureAgglomeration : It uses agglomerative clustering to group together features that look very similar, thus decreasing the number of features. 
						# It is a dimensionality reduction tool, see Unsupervised dimensionality reduction.
	#http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
	#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration
	#http://scikit-learn.org/stable/auto_examples/cluster/plot_digits_agglomeration.html#sphx-glr-auto-examples-cluster-plot-digits-agglomeration-py
	#http://scikit-learn.org/stable/auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html#sphx-glr-auto-examples-cluster-plot-feature-agglomeration-vs-univariate-selection-py
###############################################################################

#	https://jasdumas.github.io/2016-05-25-kmeans-analysis-in-python/
#	https://www.coursera.org/learn/machine-learning-data-analysis/home/week/4
#	http://www.sthda.com/english/wiki/determining-the-optimal-number-of-clusters-3-must-known-methods-unsupervised-machine-learning
#	http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#######################################################################################################################################
### import module
from __future__ import print_function
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering, FeatureAgglomeration
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy as sp
import seaborn as sns

from sklearn.datasets import load_iris
import sklearn.datasets as datasets
import random

print(__doc__)

#######################################################################################################################################
### Define functions
###
def check_constant_variable(input_data, column_name_data):
	#Check for constant variable
	for var in range(input_data.shape[1]):
		unique_count=np.unique(input_data[:,var]).size
		if unique_count < 2:
			print ("REMOVE constant variable", '\n', unique_count, "no of levels in variable:", list(column_name_data.columns)[var])
		#else:
			#pass
###
def KMeans_and_Plot_Elbow_Method(clusters, 	input_data):
		meandist	= []
		##################################
		for k in clusters:
				model		= KMeans(n_clusters=k)
				model.fit(input_data)
				clusassign	= model.predict(input_data)
				meandist.append(sum(np.min(cdist(input_data, model.cluster_centers_, 'euclidean'), axis=1)) / input_data.shape[0])
		#Plot average distance from observations from the cluster centroid to use the Elbow Method to identify number of clusters to choose
		plt.plot(clusters, meandist)
		plt.xlabel('Number of clusters')
		plt.ylabel('Average distance')
		plt.title('Selecting k with the Elbow Method')
		return model, meandist

###
def compute_BIC(kmeans,input_data):
		#Computes the BIC metric for a given clusters. Parameters:
			#kmeans		:  	List of clustering object from scikit learn
			#input_data :  	multidimension np array of data points
			#Returns	: 	BIC value
		#####################
		# assign centers and labels
		centers 	= [kmeans.cluster_centers_]
		labels  	= kmeans.labels_
		#number of clusters
		m 			= kmeans.n_clusters
		# size of the clusters
		n 			= np.bincount(labels)
		#size of data set
		N, d 		= input_data.shape
		#compute variance for all clusters beforehand
		cl_var 		= (1.0 / (N - m) / d) * sum([sum(cdist(input_data[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])
		const_term 	= 0.5 * m * np.log(N) * (d+1)
		BIC 		= np.sum([n[i]*np.log(n[i]) - n[i]*np.log(N) - ((n[i] * d)/2)*np.log(2*np.pi*cl_var) - ((n[i] - 1)*d/2) for i in range(m)]) - const_term				
		return(BIC)

###
def plot_BIC(clusters, BIC):		
		plt.plot(clusters,BIC,'r-o')
		plt.title("cluster vs BIC")
		plt.xlabel("clusters")
		plt.ylabel("BIC")

###
def silhouette_plot(input_data, range_n_clusters):
		from sklearn.cluster import KMeans
		print ('choose k where silhouette_score is maximum')
		### silhouette close to 1 implies the datum is in an appropriate cluster, while a silhouette close to -1 implies the datum is in the wrong cluster.
		#http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html		
		silhouette_scr 		= pd.DataFrame(columns=['cluster_number', 'average_silhouette_score'])
		### silhouette close to 1 implies the datum is in an appropriate cluster, while a silhouette close to -1 implies the datum is in the wrong cluster.	
		for n_clusters in range_n_clusters:
				# Create a subplot with 1 row and 2 columns
				fig, (ax1, ax2) = plt.subplots(1, 2)
				fig.set_size_inches(18, 7)
				# The 1st subplot is the silhouette plot. The silhouette coefficient can range from [-1, 1]
				ax1.set_xlim([-1, 1])
				# The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
				ax1.set_ylim([0, len(input_data) + (n_clusters + 1) * 10])
				# Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
				clusterer = KMeans(n_clusters=n_clusters)
				cluster_labels = clusterer.fit_predict(input_data)
				# The silhouette_score gives the average value for all the samples. This gives a perspective into the density and separation of the formed clusters
				silhouette_avg = silhouette_score(input_data, cluster_labels)
				silhouette_scr.loc[len(silhouette_scr.index)] = [n_clusters, silhouette_avg]								
				print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)				
				# Compute the silhouette scores for each sample
				sample_silhouette_values = silhouette_samples(input_data, cluster_labels)
				y_lower = 10
				for i in range(n_clusters):
						# Aggregate the silhouette scores for samples belonging to cluster i, and sort them
						ith_cluster_silhouette_values 	= sample_silhouette_values[cluster_labels == i]
						ith_cluster_silhouette_values.sort()
						size_cluster_i 					= ith_cluster_silhouette_values.shape[0]
						y_upper 						= y_lower + size_cluster_i
						color 							= cm.spectral(float(i) / n_clusters)
						ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
						# Label the silhouette plots with their cluster numbers at the middle
						ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
						# Compute the new y_lower for next plot
						y_lower 						= y_upper + 10  # 10 for the 0 samples					
				ax1.set_title("The silhouette plot for the various clusters.")
				ax1.set_xlabel("The silhouette coefficient values")
				ax1.set_ylabel("Cluster label")
				# The vertical line for average silhouette score of all the values
				ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
				ax1.set_yticks([])  # Clear the yaxis labels / ticks
				ax1.set_xticks([])
				# 2nd Plot showing the actual clusters formed
				colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
				ax2.scatter(input_data[:, 0], input_data[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)
				# Labeling the clusters
				centers = clusterer.cluster_centers_
				# Draw white circles at cluster centers
				ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200)
				for i, c in enumerate(centers):
						ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
				##
				ax2.set_title("The visualization of the clustered data.")
				ax2.set_xlabel("Feature space for the 1st feature")
				ax2.set_ylabel("Feature space for the 2nd feature")
				plt.suptitle(("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
				plt.savefig('Silhouette score for cluster - '+ str(n_clusters) +'.png')
				#plt.show()				
		return silhouette_scr

###
def compute_ssq(data, k, kmeans):
		#http://saravanan-thirumuruganathan.github.io/cse5334Spring2015/assignments/PA3/PA3_Clustering_DimensionalityReduction.html
		dist = np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)
		tot_withinss = sum(dist**2) # Total within-cluster sum of squares
		totss = sum(pdist(data)**2) / data.shape[0] # The total sum of squares
		betweenss = totss - tot_withinss # The between-cluster sum of squares
		return betweenss/totss*100

###    
def ssq_statistics(data, ks, ssq_norm=True):
		ssqs = sp.zeros((len(ks),)) # array for SSQs (lenth ks)    
		for (i,k) in enumerate(ks): # iterate over the range of k values
			kmeans = KMeans(n_clusters=k, random_state=1234).fit(data)        
			if ssq_norm:
				ssqs[i] = compute_ssq(data, k, kmeans)
			else:
				# The sum of squared error (SSQ) for k
				ssqs[i] = kmeans.inertia_
		return ssqs

###    		
def gap_statistics(data, refs=None, nrefs=20, ks=range(1,11)):
		########################################################
		#This function computes the Gap statistic of the data (given as a nxm matrix)
		# Returns:
		#    gaps: an array of gap statistics computed for each k.
		#    errs: an array of standard errors (se), with one corresponding to each gap computation.
		#    difs: an array of differences between each gap_k and the sum of gap_k+1 minus err_k+1.
		# The gap statistic measures the difference between within-cluster dispersion on an input dataset and that expected under an appropriate reference null distribution.
		# If you did not fully understand the definition, no worry - it is quite complex anyway. However, you should know how to USE the gap statistic if not how it is computed
		#########################################################    
		sp.random.seed(1234)
		shape = data.shape
		dst = sp.spatial.distance.euclidean    
		if refs is None:
				tops = data.max(axis=0) # maxima along the first axis (rows)
				bots = data.min(axis=0) # minima along the first axis (rows)
				dists = sp.matrix(sp.diag(tops-bots)) # the bounding box of the input dataset			
				# Generate nrefs uniform distributions each in the half-open interval [0.0, 1.0)
				rands = sp.random.random_sample(size=(shape[0],shape[1], nrefs))			
				# Adjust each of the uniform distributions to the bounding box of the input dataset
				for i in range(nrefs):
					rands[:,:,i] = rands[:,:,i]*dists+bots
		else:
				rands = refs        
		gaps = sp.zeros((len(ks),))   # array for gap statistics (lenth ks)
		errs = sp.zeros((len(ks),))   # array for model standard errors (length ks)
		difs = sp.zeros((len(ks)-1,)) # array for differences between gaps (length ks-1)
		for (i,k) in enumerate(ks): # iterate over the range of k values
				# Cluster the input dataset via k-means clustering using the current value of k
				try:
						(kmc,kml) = sp.cluster.vq.kmeans2(data, k)
				except LinAlgError:
						kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(data)
						(kmc, kml) = kmeans.cluster_centers_, kmeans.labels_

				# Generate within-dispersion measure for the clustering of the input dataset
				disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])

				# Generate within-dispersion measures for the clusterings of the reference datasets
				refdisps = sp.zeros((rands.shape[2],))
				for j in range(rands.shape[2]):
						# Cluster the reference dataset via k-means clustering using the current value of k
						try:
								(kmc,kml) = sp.cluster.vq.kmeans2(rands[:,:,j], k)
						except LinAlgError:
								kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(rands[:,:,j])
								(kmc, kml) = kmeans.cluster_centers_, kmeans.labels_
						refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
				# Compute the (estimated) gap statistic for k
				gaps[i] = sp.mean(sp.log(refdisps) - sp.log(disp))
				# Compute the expected error for k
				errs[i] = sp.sqrt(sum(((sp.log(refdisp)-sp.mean(sp.log(refdisps)))**2) for refdisp in refdisps)/float(nrefs)) * sp.sqrt(1+1/nrefs)
		# Compute the difference between gap_k and the sum of gap_k+1 minus err_k+1
		difs = sp.array([gaps[k] - (gaps[k+1]-errs[k+1]) for k in range(len(gaps)-1)])
		return gaps, errs, difs
		
###    
def plot_Elbow_Gap(data, k_min, k_max):
		#############################################################
		# Implement the following function plot_clustering_statistics. It accepts three arguments: data as nxm matrix minimum and maximum value
		# within which you think the best k lies. Of course, in the worst case this is between 1 and n (where n=number of data points)
		# You will compute the necessary statisitcs using the above function and use it to find a good $k$
		# Finding a good k, even with the statistics is a bit tricky. So we will plot the values and find a good $k$ by visually inspecting the plot
		# Interpreting the charts:
		#  	Elbow method: http://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#The_Elbow_Method
		#  	Gap Statistics: $k$ where the first drop in trend happens. 
		#	Gap Statistics differences: $k$ where you get the first positive values
		##############################################################
		#plt.figure()
		fig,axes = plt.subplots(1, 3, figsize=(16, 4))
		#range(a,b) returns a .. b-1
		ks = range(k_min, k_max+1)
		#Change below: plot the data distribution as a scatter plot on axes[0] variable
		#For now ignore the color field. We will use data where #clusters is easy to see
		#axes[0].scatter()
		#axes[0].set_title("Original Data")
		ssqs = ssq_statistics(data, ks=ks)
		#Change below: create a line chart with x axis as different k values and y-axis as ssqs on axes[1] variable. [axes[1].plot()]
		axes[0].plot(ks, ssqs)
		axes[0].set_title("Elbow Method and sum of squared distances(SSQ)")
		axes[0].set_xlabel("Number of clusters k\n(choose k where SSQ decreases abruptly)")
		axes[0].set_ylabel("sum of squared error of SSQ")
		#Do not change anything below for the rest of the function Code courtesy: Reid Johnson from U. of Notre Dame
		gaps, errs, difs = gap_statistics(data, nrefs=25, ks=ks)
		max_gap = None
		if len(np.where(difs > 0)[0]) > 0:
			max_gap = np.where(difs > 0)[0][0] + 1 # the k with the first positive dif
		if max_gap:
			print ("By gap statistics, optimal k seems to be ", max_gap)
		else:
			print ("Please use some other metrics for finding k")
		#Create an errorbar plot
		rects = axes[1].errorbar(ks, gaps, yerr=errs, xerr=None, linewidth=1.0)
		#Add figure labels and ticks. Gap Statistics: $k$ where the first drop in trend happens. 
		axes[1].set_title('Clustering Gap Statistics')
		axes[1].set_xlabel('Number of clusters k\n(choose k where the first drop in trend happens)')
		axes[1].set_ylabel('Gap Statistic')
		axes[1].set_xticks(ks)
		# Add figure bounds
		axes[1].set_ylim(0, max(gaps+errs)*1.1)
		axes[1].set_xlim(0, len(gaps)+1.0)

		ind = range(1,len(difs)+1) # the x values for the difs

		max_gap = None
		if len(np.where(difs > 0)[0]) > 0:
			max_gap = np.where(difs > 0)[0][0] + 1 # the k with the first positive dif

		#Create a bar plot
		axes[2].bar(ind, difs, alpha=0.5, color='g', align='center')

		# Add figure labels and ticks. Gap Statistics differences: $k$ where you get the first positive values
		if max_gap:
			axes[2].set_title('Clustering Gap Differences\n(k=%d Estimated as Optimal)' % (max_gap))
		else:
			axes[2].set_title('Clustering Gap Differences\n')
		axes[2].set_xlabel('Number of clusters k\n (choose k where you get the first positive values)')
		axes[2].set_ylabel('Gap Difference')
		axes[2].xaxis.set_ticks(range(1,len(difs)+1))

		#Add figure bounds
		axes[2].set_ylim(min(difs)*1.2, max(difs)*1.2)
		axes[2].set_xlim(0, len(difs)+1.0)
		plt.savefig('Gap.png')
	
###
#http://scikit-learn.org/stable/auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html#sphx-glr-auto-examples-cluster-plot-feature-agglomeration-vs-univariate-selection-py

###		
#######################################################################################################################################

os.chdir('C:\\Users\\amit.kumar\\Google Drive\\Study\\Other\\03.Cluster')
data 					= pd.read_csv("Hilton_Model_Dev_Val_Data.csv")
data.columns 			= map(str.upper, data.columns)	#upper-case all DataFrame column names
data_clean 				= data.dropna()
cluster 				= data_clean[:]				#cluster=data_clean[['ACTIVE_SESSION','SESSION_START_SERVER']]
cluster.describe() 									#(len(list(cluster.columns)))
clustervar				= cluster.copy()			# standardize clustering variables to have mean=0 and sd=1
for var in list(clustervar.columns):
		clustervar[var]	= preprocessing.scale(clustervar[var].astype('float64'))
clustervar				= clustervar.as_matrix()
clus_train, clus_test 	= train_test_split(clustervar, test_size=.3, random_state=123)
check_constant_variable(clus_train, data)
clusters				= range(1,10)
###########################################################################
### Elbow
model, meandist 		= KMeans_and_Plot_Elbow_Method(clusters, clus_train)
plt.show()
### BIC computation
KMeans 					= [KMeans(n_clusters = i, init="k-means++").fit(clus_train) for i in clusters]
BIC 					= [compute_BIC(kmeansi,clus_train) for kmeansi in KMeans]
plot_BIC(clusters, BIC)
plt.show()
### Silhouette plot
range_n_clusters		= range(2,10)
silhouette_scr 			= silhouette_plot(clus_train, range_n_clusters)
print ('maximum silhouette_score: ')
print (silhouette_scr.loc[silhouette_scr['average_silhouette_score'].idxmax()])
###### Gap	
from sklearn.cluster import KMeans, AgglomerativeClustering
plot_Elbow_Gap(clus_train, 1, 10)
plt.show()
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

# Interpret 3 cluster solution
model3			= cluster.KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign		= model3.predict(clus_train)
# plot clusters		from sklearn.decomposition import PCA
pca_2 			= PCA(2)
plot_columns 	= pca_2.fit_transform(clus_train)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

#BEGIN multiple steps to merge cluster assignment with clustering variables to examine cluster variable means by cluster
#create a unique identifier variable from the index for the cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist			= list(clus_train['index'])
# create a list of cluster assignments
labels				= list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist				= dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus				= DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns 	= ['cluster']
# now do the same for the cluster assignment variable create a unique identifier variable from the index for the cluster assignment dataframe to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe by the index variable
merged_train		= pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()
#END multiple steps to merge cluster assignment with clustering variables to examine cluster variable means by cluster
# FINALLY calculate clustering variable means by cluster
clustergrp 			= merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)
# validate clusters in training data by examining cluster differences in GPA using ANOVA first have to merge GPA with clustering variables and cluster assignment data 
gpa_data			= data_clean['ACTIVE_SESSION']
# split GPA data into train and test sets
gpa_train, gpa_test = 	train_test_split(gpa_data, test_size=.3, random_state=123)
gpa_train1			=	pd.DataFrame(gpa_train)
gpa_train1.reset_index(level=0, inplace=True)
merged_train_all	=	pd.merge(gpa_train1, merged_train, on='index')
sub1 				= 	merged_train_all[['ACTIVE_SESSION_x', 'cluster']].dropna()
gpamod 				= 	smf.ols(formula='ACTIVE_SESSION_x ~ C(cluster)', data=sub1).fit()
print (gpamod.summary())
print ('means for GPA by cluster')
m1					= 	sub1.groupby('cluster').mean()
print (m1)
print ('standard deviations for GPA by cluster')
m2					= 	sub1.groupby('cluster').std()
print (m2)
mc1 				= 	multi.MultiComparison(sub1['ACTIVE_SESSION_x'], sub1['cluster'])
res1 				= 	mc1.tukeyhsd()
print(res1.summary())

##############################################################################################################
## variable clustering
agglo = FeatureAgglomeration(n_clusters=2)
agglo.fit(clus_train)
agglo.labels_
##############################################################




#	https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/

import numpy as np
 
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
				clusters[bestmukey].append(x)
        except KeyError:
				clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu 	= []
    keys 	= sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu 	= random.sample(X, K)
    mu 		= random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu 		= mu
        # Assign all points in X to clusters
        clusters 	= cluster_points(X, mu)
        # Reevaluate centers
        mu 			= reevaluate_centers(oldmu, clusters)
    return(mu, clusters)
	
import random
 
def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X
###################################
#	https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def Wk(mu, clusters):
		K = len(mu)
		return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \ for i in range(K) for c in clusters[i]])
				
def bounding_box(X):
		xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
		ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
		return (xmin,xmax), (ymin,ymax)
 
def gap_statistic(X):
		(xmin,xmax), (ymin,ymax) = bounding_box(X)
		# Dispersion for real distribution
		ks 		= range(1,10)
		Wks 	= zeros(len(ks))
		Wkbs 	= zeros(len(ks))
		sk 		= zeros(len(ks))
		for indk, k in enumerate(ks):
				mu, clusters 	= find_centers(X,k)
				Wks[indk] 		= np.log(Wk(mu, clusters))
				# Create B reference datasets
				B 				= 10
				BWkbs 			= zeros(B)
				for i in range(B):
						Xb 		= []
						for n in range(len(X)):
								Xb.append([random.uniform(xmin,xmax),random.uniform(ymin,ymax)])
						Xb 				= np.array(Xb)
						mu, clusters 	= find_centers(Xb,k)
						BWkbs[i] 		= np.log(Wk(mu, clusters))
				Wkbs[indk] 		= sum(BWkbs)/B
				sk[indk] 		= np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
		sk = sk*np.sqrt(1+1/B)
		return(ks, Wks, Wkbs, sk)

X = init_board_gauss(200,3)
ks, logWks, logWkbs, sk = gap_statistic(X)

#http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

######################################################################################################################
--------------------------------------
#http://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

def compute_bic(kmeans,X):
		#Computes the BIC metric for a given clusters. Parameters:
			#kmeans:  List of clustering object from scikit learn
			#X     :  multidimension np array of data points
			#Returns: BIC value
		#####################
		# assign centers and labels
		centers 	= [kmeans.cluster_centers_]
		labels  	= kmeans.labels_
		#number of clusters
		m 			= kmeans.n_clusters
		# size of the clusters
		n 			= np.bincount(labels)
		#size of data set
		N, d 		= X.shape
		#compute variance for all clusters beforehand
		cl_var 		= (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])
		const_term 	= 0.5 * m * np.log(N) * (d+1)
		BIC 		= np.sum([n[i]*np.log(n[i]) - n[i]*np.log(N) - ((n[i] * d)/2)*np.log(2*np.pi*cl_var) - ((n[i] - 1)*d/2) for i in range(m)]) - const_term
		return(BIC)

#IRIS DATA
iris 	= sklearn.datasets.load_iris()
X 		= iris.data[:, :4]  # extract only the features
#Xs = StandardScaler().fit_transform(X)
Y 		= iris.target
ks 		= range(1,10)
# run 9 times kmeans and save each result in the KMeans object
KMeans 	= [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]
# now run for each cluster the BIC computation
#a=df1.as_matrix()
BIC 	= [compute_bic(kmeansi,X) for kmeansi in KMeans]
print BIC

#[-901.8088330799194, -562.67814893720902, -442.4179569307467, -401.31661808222532, -373.70396994638168, -367.27568113462917, -369.13543294596866, -351.7636856213748, -360.97885983416268]

plt.plot(ks,BIC,'r-o')
plt.title("iris data  (cluster vs BIC)")
plt.xlabel("# clusters")
plt.ylabel("# BIC")
plt.show()

#######################################################################################################################################
--------------------------------
# https://www.linkedin.com/pulse/finding-k-k-means-clustering-jaganadh-gopinadhan
import pylab as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris 				= load_iris()
k 					= range(1,11)
clusters 			= [cluster.KMeans(n_clusters = c,init = 'k-means++').fit(iris.data) for c in k]
centr_lst 			= [cc.cluster_centers_ for cc in clusters]
k_distance 			= [cdist(iris.data, cent, 'euclidean') for cent in centr_lst]
clust_indx 			= [np.argmin(kd,axis=1) for kd in k_distance]
distances 			= [np.min(kd,axis=1) for kd in k_distance]
avg_within 			= [np.sum(dist)/iris.data.shape[0] for dist in distances]
with_in_sum_square 	= [np.sum(dist ** 2) for dist in distances]
to_sum_square 		= np.sum(pdist(iris.data) ** 2)/iris.data.shape[0]
bet_sum_square 		= to_sum_square - with_in_sum_square

kidx 				= 2
fig 				= plt.figure()
ax 					= fig.add_subplot(111)
ax.plot(k, avg_within, 'g*-')
ax.plot(k[kidx], avg_within[kidx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering (IRIS Data)')

########################################################################################################################################
---------------------------------------------
#	http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(dataSet, k):
	
    # Initialize centroids randomly
    numFeatures 	= dataSet.getNumFeatures()
    centroids 		= getRandomCentroids(numFeatures, k)    
    # Initialize book keeping vars.
    iterations 		= 0
    oldCentroids 	= None
    
    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations):
			# Save old centroids for convergence test. Book keeping.
			oldCentroids 	= centroids
			iterations 		+= 1        
			# Assign labels to each datapoint based on centroids
			labels 			= getLabels(dataSet, centroids)        
			# Assign centroids based on datapoint labels
			centroids 		= getCentroids(dataSet, labels, k)        
    # We can get the labels too by calling getLabels(dataSet, centroids)
    return centroids
# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(oldCentroids, centroids, iterations):
		if iterations 		> MAX_ITERATIONS: return True
		return oldCentroids == centroids
# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset. 
def getLabels(dataSet, centroids):
    # For each element in the dataset, chose the closest centroid. 
    # Make that centroid the element's label.
# Function: Get Centroids
# -------------
# Returns k random centroids, each of dimension n.
def getCentroids(dataSet, labels, k):
    # Each centroid is the geometric mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.
	
#######################################################################################################################
#######################################################################################################################
##	https://gist.github.com/jaganadhg/ddbf0956a7921b83ceef90b8a81dfaee
"""
Author : Jaganadh Gopinadhan
Licence : Apahce 2
e-mail jaganadhg at gmail dot com 
"""
import scipy

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

import pandas as pd


class TWHGapStat(object):
    """
    Implementation of Gap Statistic from Tibshirani, Walther, Hastie to determine the 
    inherent number of clusters in a dataset with k-means clustering.
    Ref Paper : https://web.stanford.edu/~hastie/Papers/gap.pdf
    """
    
    def generate_random_data(self, X):
        """
        Populate reference data.
        
        Parameters
        ----------
        X : Numpy Array
            The base data from which random sample has to be generated
        
        Returns
        -------
        reference : Numpy Array
            Reference data generated using the Numpy/Scipy random utiity .
            NUmber of diamensions in the data will be as same as the base
            dataset. 
        """
        reference = scipy.random.random_sample(size=(X.shape[0], X.shape[1]))
        return reference
    
    def _fit_cluster(self,X, n_cluster, n_iter=5):
        """
        Fit cluster on reference data and return inertia mean.
        
        
        Parameters
        ----------
        X : numpy array
            The base data 
            
        n_cluster : int 
            The number of clusters to form 
            
        n_iter : int, default = 5
            number iterative lustering experiments has to be perfromed in the data.
            If the data is large keep it less than 5, so that the run time will be less.
        
        Returns
        -------
        mean_nertia : float 
            Returns the mean intertia value. 
        """
        iterations = range(1, n_iter + 1)
        
        ref_inertias = pd.Series(index=iterations)
        
        for iteration in iterations:
            clusterer = KMeans(n_clusters=n_cluster, n_init=3, n_jobs=-1)
            # If you are using Windows server n_jobs = -1 will be dangerous. So the 
            # value should be set to max cores - 3 . If we use all the cores available
            # in Windows server sklearn tends to throw memory error 
            clusterer.fit(X)
            ref_inertias[iteration] = clusterer.inertia_
        
        mean_nertia = ref_inertias.mean()
        
        return mean_nertia
    
    def fit(self,X,max_k):
        """
        Compute Gap Statistics
        Parameters
        ----------
        X : numpy array
            The base data 
        max_k :int 
            Maximum value to which we are going to test the 'k' in k-means algorithmn 
        Returns
        -------
        gap_stat : Pandas Series
            For eack k in max_k range gap stat value is returned as a Pandas Sereies.
            Index is K and valuess correspondes to gap stattistics for each K
        """
        
        k_range = range(1,max_k + 1)
        gap_stat = pd.Series(index=k_range)
        
        ref_data = self.generate_random_data(X)
        
        for k in k_range:
            base_clusterer = KMeans(n_clusters=k,n_init = 3, n_jobs = -1)
            base_clusterer.fit(X)
            
            ref_intertia = self._fit_cluster(ref_data,k)
            
            cur_gap = scipy.log(ref_intertia - base_clusterer.inertia_)
            
            gap_stat[k] = cur_gap
        
        return gap_stat

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data 
    
    gap_stat = TWHGapStat()
    gs = gap_stat.fit(X,5)
    print gs
	