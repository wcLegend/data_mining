"""
=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

	- TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
		frequent words to features indices and hence compute a word occurrence
		frequency (sparse) matrix. The word frequencies are then reweighted using
		the Inverse Document Frequency (IDF) vector collected feature-wise over
		the corpus.

	- HashingVectorizer hashes word occurrences to a fixed dimensional space,
		possibly with collisions. The word count vectors are then normalized to
		each have l2-norm equal to one (projected to the euclidean unit-ball) which
		seems to be important for k-means to work in high dimensional space.

		HashingVectorizer does not provide IDF weighting as this is a stateless
		model (the fit method does nothing). When IDF weighting is needed it can
		be added by pipelining its output to a TfidfTransformer instance.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce
dimensionality and discover latent patterns in the data.

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans,AffinityPropagation,MeanShift,SpectralClustering,AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
										format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
							dest="n_components", type="int",
							help="Preprocess documents with latent semantic analysis.") #使用潜在语义分析对文档进行预处理
op.add_option("--no-minibatch",
							action="store_false", dest="minibatch", default=True,
							help="Use ordinary k-means algorithm (in batch mode).")  #使用普通的k均值算法（在批处理模式下）
op.add_option("--no-idf",
							action="store_false", dest="use_idf", default=True,
							help="Disable Inverse Document Frequency feature weighting.")  #禁用TF-IDF加权
op.add_option("--use-hashing",
							action="store_true", default=False,
							help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
							help="Maximum number of features (dimensions)"
									 " to extract from text.")
op.add_option("--verbose",
							action="store_true", dest="verbose", default=False,
							help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()


def is_interactive():
		return not hasattr(sys.modules['__main__'], '__file__')




def document_clustering(estimator,name,X):

	#print("Clustering sparse data with %s" % estimator)
	t0 = time()
	estimator.fit(X.toarray())#  用toarray也行 但用了svd降维   	  
	#print("done in %0.3fs" % (time() - t0))
	#print()   
	if hasattr(estimator, 'labels_'):
			y_pred = estimator.labels_
	else:
			y_pred = estimator.predict(X)

	print('%-9s\t\t\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
				% (name, (time() - t0),
					 metrics.normalized_mutual_info_score(labels, y_pred), 
					 metrics.homogeneity_score(labels, y_pred),
					 metrics.completeness_score(labels, y_pred),
					 metrics.v_measure_score(labels, y_pred),
					 metrics.adjusted_rand_score(labels, y_pred),
					 metrics.adjusted_mutual_info_score(labels,  y_pred,
																							average_method='arithmetic')))


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
		op.error("this script takes no arguments.")
		sys.exit(1)


# #############################################################################
# Load some categories from the training set
categories = [
		'alt.atheism',
		'talk.religion.misc',
		'comp.graphics',
		'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
														 shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset "
			"using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
		if opts.use_idf:
				# Perform an IDF normalization on the output of HashingVectorizer
				hasher = HashingVectorizer(n_features=opts.n_features,
																	 stop_words='english', alternate_sign=False,
																	 norm=None, binary=False)
				vectorizer = make_pipeline(hasher, TfidfTransformer())
		else:
				vectorizer = HashingVectorizer(n_features=opts.n_features,
																			 stop_words='english',
																			 alternate_sign=False, norm='l2',
																			 binary=False)
else:
		vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
																 min_df=2, stop_words='english',
																 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
		print("Performing dimensionality reduction using LSA")
		t0 = time()
		# Vectorizer results are normalized, which makes KMeans behave as
		# spherical k-means for better results. Since LSA/SVD results are
		# not normalized, we have to redo the normalization.
		svd = TruncatedSVD(opts.n_components)
		normalizer = Normalizer(copy=False)
		lsa = make_pipeline(svd, normalizer)

		X = lsa.fit_transform(X)

		print("done in %fs" % (time() - t0))

		explained_variance = svd.explained_variance_ratio_.sum()
		print("Explained variance of the SVD step: {}%".format(
				int(explained_variance * 100)))

		print()


# #############################################################################
# Do the actual clustering
print(82 * '_')
print('init\t\t\t\ttime\tNMI\t\thomo\tcompl\tv-meas\tARS\t\tAMI')

mbkm = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
												 init_size=1000, batch_size=1000, verbose=opts.verbose)
document_clustering(mbkm,'MBKMeans',X)

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
								verbose=opts.verbose)
document_clustering(km,'KMeans',X)

af = AffinityPropagation(preference=-50,convergence_iter = 10,verbose=opts.verbose) #效果很差
document_clustering(af,'AfPro',X)


ms = MeanShift(bandwidth=2)
document_clustering(ms,'MeanS',X)


whc  = AgglomerativeClustering(n_clusters=4,linkage='ward')
document_clustering(whc,'Ward/Aggclu',X)

chc = AgglomerativeClustering(n_clusters=4,linkage='complete')
document_clustering(chc,'comp/Aggclu',X)

ahc = AgglomerativeClustering(n_clusters=4,linkage='average')
document_clustering(ahc,'ave/Aggclu',X)

shc = AgglomerativeClustering(n_clusters=4,linkage='single')  #效果很差
document_clustering(shc,'sin/Aggclu',X)

sc = SpectralClustering(n_clusters=4, eigen_solver='arpack', affinity="nearest_neighbors")
document_clustering(sc,'SpeClu',X)

db = DBSCAN() #效果很差  当SVD取K=100的时候效果还行
document_clustering(db,'DBSCAN',X)

gau = GaussianMixture(n_components = 4, covariance_type='full')
document_clustering(gau,'GauMix',X)


"""
print("NMI: %0.3f"%metrics.normalized_mutual_info_score(labels, km.labels_))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
			% metrics.adjusted_rand_score(labels, km.labels_))
print()
"""

"""
if not opts.use_hashing:
		print("Top terms per cluster:")

		if opts.n_components:
				original_space_centroids = svd.inverse_transform(km.cluster_centers_)
				order_centroids = original_space_centroids.argsort()[:, ::-1]
		else:
				order_centroids = km.cluster_centers_.argsort()[:, ::-1]

		terms = vectorizer.get_feature_names()
		for i in range(true_k):
				print("Cluster %d:" % i, end='')
				for ind in order_centroids[i, :10]:
						print(' %s' % terms[ind], end='')
				print()
"""