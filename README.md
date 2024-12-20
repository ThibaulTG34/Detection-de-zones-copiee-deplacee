# Détection de zones copiées-déplacées dans des images

## Objectifs
L'objectif de ce projet en binome était de développer un algorithme permettant de détecter des zones copiées-déplacées dans des images. Ensuite, nous devions implémenté une interface pour montrer le résultat. Ce projet a été réalisé en Python avec la libraire OpenCV pour la partie algorithme et nous avions utilisé PyQt pour la partie interface.

## Méthode utilisée
Nous avions décliner l'algorithme de détection en trois étapes : 
- Détection des points d'intérêts avec l'algo SIFT (OpenCV)
- Algorithme de clustering - DBSCAN
- Résultats

### Détection des points d'intérêts
L'image de gauche est l'image originale.
<div align="center"><img src="https://github.com/ThibaulTG34/Detection-de-zones-copiee-deplacee/blob/main/Code/R%C3%A9sultats/sift_keypoints.jpg" alt="image" style="width:350px;height:auto;">
<img src="https://github.com/ThibaulTG34/Detection-de-zones-copiee-deplacee/blob/main/Code/R%C3%A9sultats/sift_keypoints_forged.jpg" alt="image" style="width:350px;height:auto;">
</div>

### Algorithme de clustering
<div align="center"><img src="https://github.com/ThibaulTG34/Detection-de-zones-copiee-deplacee/blob/main/Code/R%C3%A9sultats/sift_keypoints.jpg" alt="image" style="width:350px;height:auto;">
<img src="https://github.com/ThibaulTG34/Detection-de-zones-copiee-deplacee/blob/main/Code/R%C3%A9sultats/sift_keypoints_forged.jpg" alt="image" style="width:350px;height:auto;">
</div>
