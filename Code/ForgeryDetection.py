import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class ForgeryDetection:
    display = False
    # clusters = None
    
    def __init__(self, img):
        self.img = img.copy()
        # self.clusters = None

    def KeyPointDetector(self, img1, filename):
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp1, descriptors_1 = sift.detectAndCompute(gray1, None)
        dst = cv.drawKeypoints(img1, kp1, img1)
        filename = filename.split('.')
        self.filename = filename[0] + "_KeyPoints."+filename[1]
        
        return self.img, dst, self.filename
        

    def ForgeryDetect(self, img1, filename):
        #img1 = cv.imread(filename)
        print(img1.shape)
        # gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp1, descriptors_1 = sift.detectAndCompute(img1, None)
        self.filename = filename
        self.img = img1
        dst2, name = self.locateForgery(kp1, descriptors_1)
        
        return dst2, name
    
        
    # def displayPlot(self):
    #     eps = 40
    #     min_sample = 2
    #     gray1 = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
    #     sift = cv.SIFT_create()
    #     kp1, descriptors_1 = sift.detectAndCompute(gray1, None)
    #     clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(descriptors_1) # Find self.clusters using DBSCAN
    #     n_noise_ = list(clusters.labels_).count(-1)
    #     n_clusters_ = len(set(clusters.labels_)) - (1 if -1 in clusters.labels_ else 0)
    #     print(n_noise_)
    #     print(n_clusters_)
    #     unique_labels = set(clusters.labels_)
    #     core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
    #     core_samples_mask[clusters.core_sample_indices_] = True

    #     colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #     for k, col in zip(unique_labels, colors):

    #         class_member_mask = (clusters.labels_ == k)

    #         xy = descriptors_1[class_member_mask & core_samples_mask]
    #         plt.plot(
    #             xy[:, 0],
    #             xy[:, 1],
    #             "o",
    #             markerfacecolor=tuple(col),
    #             markeredgecolor="k",
    #             markersize=14,
    #         )

    #         xy = descriptors_1[class_member_mask & ~core_samples_mask]
    #         """plt.plot(
    #             xy[:, 0],
    #             xy[:, 1],
    #             "x",
    #             markerfacecolor=tuple(col),
    #             markeredgecolor="k",
    #             markersize=6,
    #         )"""

    #     plt.title(f"Estimated number of self.clusters: {n_clusters_}")
    #     plt.show()
    

    def locateForgery(self, kp, descriptors, eps = 40, min_sample = 2):
        from sklearn.metrics import silhouette_score

        
        epsilon_list = [i for i in range(1,50)]
        print(epsilon_list)
        minPt_list = [2, 5, 10, 20, 25]

        best_score = -1
        best_eps = None
        best_min_sample = None

        for eps in epsilon_list:
            for min_sample in minPt_list:
                clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(descriptors)
                n_clusters_ = len(set(clusters.labels_)) - (1 if -1 in clusters.labels_ else 0)
                if n_clusters_ > 1:
                    score = silhouette_score(descriptors, clusters.labels_)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_sample = min_sample

        print("Best eps:", best_eps)
        print("Best min_sample:", best_min_sample)
        
        if best_eps == best_min_sample == None:
            return None, None
        
        
        self.clusters = DBSCAN(eps=best_eps, min_samples=best_min_sample).fit(descriptors)

        # print(n_noise_)
        # print(n_clusters_)
        # print(len(set(self.clusters.labels_)) - (1 if -1 in self.clusters.labels_ else 0))
        
        
        size=np.unique(self.clusters.labels_).shape[0]-1 
        if (size==0) and (np.unique(self.clusters.labels_)[0]==-1):
            print('Pas de falsification !')
            return None, None                                           
        if size==0:
            size=1
        
        
        cluster_list = [[] for i in range(size)]
        for index in range(len(kp)):
            if self.clusters.labels_[index] != -1: 
                cluster_list[self.clusters.labels_[index]].append((int(kp[index].pt[0]),int(kp[index].pt[1])))
            for points in cluster_list:
                if len(points) > 1:
                    for idx1 in range(1,len(points)):
                        cv.line(self.img, points[0], points[idx1], (255,0,0), 5) 
        
        filename = self.filename.split('.')
        name = filename[0] + "_ForgeryDetected."+filename[1]
        cv.imwrite("falsifiee.png", self.img)
        return self.img, name
    
    
    
    
    
    
    """Reduire la dimension des descripteurs SIFT
        mean, _ = cv.PCACompute(descriptors, mean=None)
        # Centrage des données
        descriptors_centered = descriptors - mean
        # Calcul des composantes principales et des valeurs propres
        _, eigenvecs = cv.PCACompute(descriptors_centered, mean=None, maxComponents=64)
        # Réduction de dimension
        descriptors_reduced = np.dot(descriptors_centered, eigenvecs.T)
        # Affichage de la nouvelle shape
        print(descriptors_reduced.shape)
        """""""""""""""""""""""""""""""""""""""""""""
        
        
"""def voisinage(self, descrip, pt, epsilon):
    neighbors = []
    for i,point in enumerate(descrip):
        distance = np.linalg.norm(point - pt)
        if distance <= epsilon:
            neighbors.append(i)
    
    return neighbors


def extend_cluster(self, D, P, PtsVoisins, cluster_index, eps, min_sample, labels, visited):
    labels[P.astype(dtype=int)] = cluster_index
    #i = 0
    for i,pt in enumerate(PtsVoisins):
        P_prime = i
        if not visited[P_prime]:
            visited[P_prime] = True
            PtsVoisins_prime = self.voisinage(D, D[P_prime], eps)
            if len(PtsVoisins_prime) >= min_sample:
                PtsVoisins.extend(PtsVoisins_prime)
        if labels[P_prime] == 0:
            labels[P_prime] = cluster_index
        #i += 1

def dbscan(self, D, eps, min_sample):
    
    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=int)
    cluster_index = 0
    
    # Parcours des points non visités
    for i,pt in enumerate(D):
        if not visited[i]:
            visited[i] = True
            PtsVoisins = self.voisinage(D, pt, eps)
            print(len(PtsVoisins))
            if len(PtsVoisins) < min_sample:
                labels[i] = -1
            else:
                cluster_index += 1
                self.extend_cluster(D, pt, PtsVoisins, cluster_index, eps, min_sample, labels, visited)
                
    return labels
"""