print(__doc__)
import warnings
warnings.filterwarnings("ignore")
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,SpectralClustering,AgglomerativeClustering,DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture


np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\t\t\ttime\tNMI\t\thomo\tcompl\tv-meas\tARI\t\tAMI')


def bench_Clustering(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    #print('start')
    if hasattr(estimator, 'labels_'):
        y_pred = estimator.labels_
    else:
        y_pred = estimator.predict(data)

    print('%-9s\t\t\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0),
             metrics.normalized_mutual_info_score(labels, y_pred), 
             metrics.homogeneity_score(labels, y_pred),
             metrics.completeness_score(labels, y_pred),
             metrics.v_measure_score(labels, y_pred),
             metrics.adjusted_rand_score(labels, y_pred),
             metrics.adjusted_mutual_info_score(labels,  y_pred,
                                                average_method='arithmetic')))

#bench_Clustering(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),name="k-means++", data=data)

# n_clusters 代表族数  init代表初始化的方法   n_init 代表在不同种子下的运行次数

#bench_Clustering(KMeans(init='random', n_clusters=n_digits, n_init=10),name="random", data=data)
 
# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
#bench_Clustering(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),name="PCA-based",data=data)

af = AffinityPropagation(preference=-50,convergence_iter = 10)#convergence_iter 停止收敛的估计簇数没有变化的迭代数
#bench_Clustering(af,name="AfPro",data=data)
ms = MeanShift(bandwidth=2)# 带宽参数 没看懂什么意思加上效果很好
#bench_Clustering(ms,name="MeanShift",data=data)
#sc = SpectralClustering(n_clusters=10,assign_labels="discretize",random_state=0)


whc = AgglomerativeClustering(n_clusters=10,linkage='ward')
#bench_Clustering(whc,name="Ward/Aggclu",data=data)

chc = AgglomerativeClustering(n_clusters=10,linkage='complete')
#bench_Clustering(chc,name="comp/Aggclu",data=data)

ahc = AgglomerativeClustering(n_clusters=10,linkage='average')
#bench_Clustering(ahc,name="ave/Aggclu",data=data)

shc = AgglomerativeClustering(n_clusters=10,linkage='single')
#bench_Clustering(shc,name="sin/Aggclu",data=data)


sc = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="nearest_neighbors")

#n_clusters 投影空间  random_state 类似seed 确定随机性  assign_labels 拉普拉斯分配标签的策略，有两种离散discretize和kmeans 默认kmeans
#bench_Clustering(sc,name="SpeClu",data=data)


db = DBSCAN(eps=4, min_samples=4)
#bench_Clustering(db,name="DBSCAN",data=data)
# eps 两个邻域之间两个样本的最大距离，最重要的参数
# min_samples 当一个点周围有这些数量的同类点时被视为中心点

gau = GaussianMixture(n_components = 10, covariance_type='full')
#bench_Clustering(gau,name="GauMix",data=data)

print(82 * '_')








#"""
# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
#method = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
#method = AffinityPropagation(preference=-50,convergence_iter = 10)
method = MeanShift(bandwidth=2)
method.fit(reduced_data)


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = method.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = method.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
#"""