import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist # computing the distance
from scipy.cluster.hierarchy import inconsistent
from scipy.cluster.hierarchy import fcluster


class Principle_Component_Analysis:
    def __init__(self, data, var_per):
        self.data = data
        self.pca = PCA(var_per, random_state = 0)
        self.PCA = self.pca.fit(self.Standard_Scaler_Preprocess().drop(['PLAYER', 'TEAM', 'POSITION'], axis = 1))
        
    def Standard_Scaler_Preprocess(self):    
        std_scale = StandardScaler()
        std_scale_data = std_scale.fit_transform(self.data.drop(['PLAYER', 'TEAM', 'POSITION'], axis = 1))
        std_scale_data = pd.DataFrame(std_scale_data, columns = self.data.drop(['PLAYER', 'TEAM', 'POSITION'], axis = 1).columns.tolist())
        std_scale_data['PLAYER'] = self.data['PLAYER']
        std_scale_data['TEAM'] = self.data['TEAM']
        std_scale_data['POSITION'] = self.data['POSITION']
        return std_scale_data
    
    def PCA_name(self):
        PCA_name = []
        for i in range(1, self.PCA.n_components_ + 1):
            PCA_name += ['PC' + str(i)]
        return PCA_name
    
    def PCA_variance(self):
        pca_variance = pd.DataFrame({"Variance Explained" : self.PCA.explained_variance_,
                                     'Percentage of Variance Explained' : self.PCA.explained_variance_ratio_}, index = self.PCA_name())
        pca_variance['Percentage of Variance Explained'] = (pca_variance['Percentage of Variance Explained'] * 100).round(0)
        pca_variance['Cumulative Percentage of Variance Explained'] = pca_variance['Percentage of Variance Explained'].cumsum()
        return pca_variance
    
    def PCA_transform(self, n):
        pca_data = self.pca.fit_transform(self.Standard_Scaler_Preprocess().drop(['PLAYER', 'TEAM', 'POSITION'], axis = 1))
        pca_data = pd.DataFrame(pca_data, columns = self.PCA_name())
        index = []
        for i in range(1, n+1):
            index += ['PC' + str(i)]
        pca_data = pca_data[index]
        pca_data['PLAYER'] = self.Standard_Scaler_Preprocess()['PLAYER']
        pca_data['TEAM'] = self.Standard_Scaler_Preprocess()['TEAM']
        pca_data['POSITION'] = self.Standard_Scaler_Preprocess()['POSITION']
        return pca_data
    
    def Heatmap(self): 
        pca_eigen = pd.DataFrame(self.PCA.components_, columns = self.Standard_Scaler_Preprocess().drop(['PLAYER', 'TEAM', 'POSITION'], axis = 1).columns.tolist(), index = self.PCA_name()).T
        plt.figure(figsize = (10,10))
        sns.heatmap(pca_eigen.abs(), vmax = 0.5, vmin = 0)
        
    def PCA_sorted_eigen(self, PC):
        pca_eigen = pd.DataFrame(self.PCA.components_, columns = self.Standard_Scaler_Preprocess().drop(['PLAYER', 'TEAM', 'POSITION'], axis = 1).columns.tolist(), index = self.PCA_name()).T
        return pca_eigen.loc[pca_eigen[PC].abs().sort_values(ascending = False).index][PC]


def HeatMap(df, vert_min, vert_max):
    plt.figure(figsize = (10,10))
    sns.heatmap(df.corr(),
                vmin = vert_min, vmax = vert_max, center = 0,
                cmap = sns.diverging_palette(20, 220, n = 200),
                square = True)

def Standard_Scaler_Preprocess(data):    
    std_scale = StandardScaler()
    std_scale_data = std_scale.fit_transform(data.drop(['PLAYER', 'TEAM', 'POSITION'], axis = 1))
    std_scale_data = pd.DataFrame(std_scale_data, columns = data.drop(['PLAYER', 'TEAM', 'POSITION'], axis = 1).columns.tolist())
    std_scale_data['PLAYER'] = data['PLAYER']
    std_scale_data['TEAM'] = data['TEAM']
    std_scale_data['POSITION'] = data['POSITION']
    return std_scale_data


class Cluster:
    def __init__(self, df, method):
        self.df = df
        self.method = method
        self.linked = linkage(self.df, self.method)

    def cophenet_value(self):
        c, coph_dists = cophenet(self.linked, pdist(self.df))
        return c

    def dendrogram_plot(self):
        plt.figure(figsize=(15, 6)) 
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('distance')
        dendrogram(self.linked,
                   orientation='top', #The direction to plot the dendrogram
                              #The root at the top, and descendent links going downwards
                   #labels=statesList,
                   distance_sort='descending',
                   show_leaf_counts=True)
        plt.show()

    def dendrogram_truncated(self, n, y_min = 0, max_d = 0):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendro = dendrogram(
                    self.linked,
                    truncate_mode='lastp',  # show only the last p merged clusters
                    p=n,  # show only the last p merged clusters
                    leaf_rotation=90.,
                    leaf_font_size=12.,
                    show_contracted=True,  # to get a distribution impression in truncated branches
                )

        for i, d, c in zip(dendro['icoord'], dendro['dcoord'], dendro['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            #if y > annotate_above:
            plt.plot(x, y, 'o', c=c)
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                         textcoords='offset points',
                         va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')

        plt.ylim(ymin = y_min)
        plt.show()

    def inconsistency(self):
        depth = 3
        incons = inconsistent(self.linked, depth)
        return incons[-15:]

    def elbow_plot(self, cut = 0):
        last = self.linked[(-1*cut):, 2]
        last_rev = last[::-1]
        idxs = np.arange(1, len(last) + 1)
        plt.plot(idxs, last_rev)

        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        self.acceleration_rev = acceleration[::-1]
        plt.plot(idxs[:-2] + 1, self.acceleration_rev)
        plt.show()
    
    def elbow_point(self):
        k = self.acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
        return k

    def create_cluster(self, max_d):
        clusters = fcluster(self.linked, max_d, criterion='distance')
        return clusters

print('done')