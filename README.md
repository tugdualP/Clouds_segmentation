# Projet Clouds segmentation (Segmentation de régions nuageuses)

Ce répertoire Git contient 8 fichers codés en Python (2 scripts et 6 notebooks).
Attention, pour éviter les problèmes de compatibilité, ces fichiers doivent être exécutés en utilisant la version 2.3.1 de Keras et la version 2.0 de Tensorflow

Voici le détail de ces fichiers:

* 2 scripts python pour généraliser les fonctions utiles à tous les notebooks: 

  * clouds_graph_functions: script python avec toutes les fonctions utiles à l'affichage
  * clouds_utilities_functions: script python avec toutes les fonctions utiles non liées à l'affichage
  
* 6 notebooks:
ces notebooks nécessitent l'ajout des 2 scripts ci-dessus en utility script et l'ajout des données d'input du projet Kaggle 
https://www.kaggle.com/c/understanding_cloud_organization/ (soit 2 répertoires d'images test_images, train_images et un fichier csv, train.csv)
  
    1. Partie analyse des données et data visualisation:
          * clouds-init-dataviz : notebook initial de prise en main et analyse des données + datavisualisation
          
    2. Partie classification multilabels:
          * clouds-classification : notebook de classification multilabels (différents modèles peuvent être testés)
          
    3. Partie segmentation sémantique
          * clouds-segmentation-unet : notebook de segmentation avec le modèle U-Net
          * clouds-segmentation-fpn : notebook de segmentation avec le modèle FPN
          * clouds-segmentation-segnet : notebook de segmentation avec le modèle SegNet
          
    4. Partie segmentation d'instance (détection objets):
          * clouds-segmentation-mask-rcnn : notebook de détection d'objets avec le modèle mask R-CNN
          
