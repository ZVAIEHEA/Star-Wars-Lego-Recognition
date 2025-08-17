# Créer une fonction qui va aléatoirement enlever des pixels de chaque image afin de pouvoir entrainer le modèle. Utiliser l'exemple en bookmark sur kaggle sur minifigs lego.

import os
import math
import random

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
import albumentations as A
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2 as tf_mobilenet_v2
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import models as tf_models
from tensorflow.keras import callbacks as tf_callbacks
from sklearn import metrics as sk_metrics
from sklearn import model_selection as sk_model_selection

def load_image_safely(image_path, target_size=(256, 256)):
    """
    Charge une image de manière sécurisée et la redimensionne.
    
    Args:
        image_path (str): Chemin vers l'image
        target_size (tuple): Taille cible (largeur, hauteur)
    
    Returns:
        numpy.ndarray: Image sous forme d'array NumPy ou None si erreur
    """
    if not os.path.exists(image_path):
        print(f"Erreur: Le fichier {image_path} n'existe pas")
        return None
    
    # Charger l'image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return None
    
    # Convertir de BGR à RGB (OpenCV charge en BGR par défaut)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionner l'image
    image = cv2.resize(image, target_size)
    
    # Normaliser les valeurs des pixels (0-255 -> 0-1)
    image = image.astype(np.float32) / 255.0
    
    return image

def apply_data_augmentation(image):
    """
    Applique des transformations aléatoires à une image pour l'augmentation de données.
    
    Args:
        image (numpy.ndarray): Image sous forme d'array NumPy
    
    Returns:
        numpy.ndarray: Image transformée
    """
    if image is None:
        return None
    
    # Définir les transformations d'augmentation
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
    ])
    
    # Appliquer les transformations
    transformed = transform(image=image)
    return transformed['image']

def load_minifig_dataset(image_dir="minifig_images", target_size=(224, 224)):
    """
    Charge toutes les images de minifigures et crée un dataset.
    
    Args:
        image_dir (str): Répertoire contenant les images
        target_size (tuple): Taille cible pour les images
    
    Returns:
        tuple: (images, labels, label_names)
    """
    images = []
    labels = []
    label_names = []
    
    if not os.path.exists(image_dir):
        print(f"Erreur: Le répertoire {image_dir} n'existe pas")
        return np.array([]), np.array([]), []
    
    # Obtenir les noms uniques des personnages à partir des noms de fichiers
    files = os.listdir(image_dir)
    character_names = set()
    
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Extraire le nom du personnage du nom de fichier
            character_name = ''.join([c for c in file if not c.isdigit()]).replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').strip()
            character_names.add(character_name)
    
    label_names = sorted(list(character_names))
    print(f"Personnages détectés: {label_names}")
    
    # Charger les images
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, file)
            image = load_image_safely(image_path, target_size)
            
            if image is not None:
                # Extraire le label du nom de fichier
                character_name = ''.join([c for c in file if not c.isdigit()]).replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').strip()
                
                if character_name in label_names:
                    label_index = label_names.index(character_name)
                    images.append(image)
                    labels.append(label_index)
    
    print(f"Chargé {len(images)} images pour {len(label_names)} personnages")
    
    return np.array(images), np.array(labels), label_names

def create_augmented_dataset(images, labels, augment_factor=3):
    """
    Crée un dataset augmenté en appliquant des transformations aléatoires.
    
    Args:
        images (numpy.ndarray): Images originales
        labels (numpy.ndarray): Labels correspondants
        augment_factor (int): Nombre d'images augmentées à créer par image originale
    
    Returns:
        tuple: (images_augmentées, labels_augmentés)
    """
    augmented_images = []
    augmented_labels = []
    
    # Ajouter les images originales
    for i, image in enumerate(images):
        augmented_images.append(image)
        augmented_labels.append(labels[i])
        
        # Créer des versions augmentées
        for _ in range(augment_factor):
            aug_image = apply_data_augmentation(image)
            if aug_image is not None:
                augmented_images.append(aug_image)
                augmented_labels.append(labels[i])
    
    print(f"Dataset augmenté: {len(augmented_images)} images au total")
    
    return np.array(augmented_images), np.array(augmented_labels)