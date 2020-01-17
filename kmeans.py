import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from pandas.plotting import parallel_coordinates

#============ K OPTIM ================= 
# import csv

_data = pd.read_csv('falldetection.csv')

# categorical_features = data[['ACTIVITY']]
# continuous_features = data[['TIME','SL', 'EEG','BP', 'HR', 'CIRCLUATION']]

# for col in categorical_features:
#     dummies = pd.get_dummies(data[col], prefix=col)
#     data = pd.concat([data, dummies], axis=1)
#     data.drop(col, axis=1, inplace=True)

# mms = MinMaxScaler()
# mms.fit(data)
# data_transformed = mms.transform(data)

# Sum_of_squared_distances = []
# K = range(1,15)
# for k in K:
#     km = KMeans(n_clusters=k)
#     km = km.fit(data_transformed)
#     Sum_of_squared_distances.append(km.inertia_)

# plt.plot(K, Sum_of_squared_distances, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Sum_of_squared_distances')
# plt.title('Elbow Method For Optimal k')
# plt.show()

#============= GRAFIC IN FUNCTIE DE VALORI =====================

# sampled_df = data[(data['ID'] % 10) == 0]
# print(sampled_df.shape)

# features = ['ACTIVITY', 'TIME', 'SL', 'EEG', 'BP', 'HR','CIRCLUATION']

# select_df = sampled_df[features]

# X = StandardScaler().fit_transform(select_df)

# kmeans = KMeans(n_clusters=5)
# model = kmeans.fit(X)

# centers = model.cluster_centers_

# def pd_centers(featuresUsed, centers):
# 	colNames = list(featuresUsed)
# 	colNames.append('prediction')

# 	# Zip with a column called 'prediction' (index)
# 	Z = [np.append(A, index) for index, A in enumerate(centers)]

# 	# Convert to pandas data frame for plotting
# 	P = pd.DataFrame(Z, columns=colNames)
# 	P['prediction'] = P['prediction'].astype(int)
# 	return P

# def parallel_plot(data):
# 	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
# 	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-8,+8])
# 	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')



# P = pd_centers(features, centers)

# parallel_plot(P[P['BP'] < 100])
# plt.show()

#===========================================================================

# features = list(_data.columns)

# data = _data[features]

# clustering_kmeans = KMeans(n_clusters=5, precompute_distances="auto", n_jobs=-1)
# data['clusters'] = clustering_kmeans.fit_predict(data)

# reduced_data = PCA(n_components=2).fit_transform(data)
# results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

# sns.scatterplot(x="pca1", y="pca2", hue=data['clusters'], data=results)
# plt.title('K-means Clustering with 2 dimensions')
# plt.show()

#===========================================
