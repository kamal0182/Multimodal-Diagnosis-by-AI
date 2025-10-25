import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pandas as pd
import os
import io

# Importation du modèle GoogLeNet standard de torchvision
# C'est la structure complète requise pour charger les poids
from torchvision.models import googlenet, GoogLeNet_Weights
import torchvision.transforms as transforms

# =================================================================
# 1. MODEL ARCHITECTURE DEFINITION (The Blueprint)
# =================================================================
# ATTENTION: Nous utilisons le modèle standard de PyTorch et modifions 
# sa tête de classification (la couche 'fc') pour correspondre à vos 4 classes.
# Ceci garantit que toutes les couches intermédiaires sont présentes.

class CustomGoogLeNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Charger le modèle GoogLeNet pré-entraîné (ici, on utilise weights=None 
        # car on va charger nos propres poids)
        # Note: 'weights=None' signifie qu'il crée l'architecture avec des poids aléatoires
        self.base_model = googlenet(weights=None, aux_logits=False)
        
        # Le modèle standard se termine par une couche 'fc' (Linear).
        # On récupère la taille de l'entrée de cette couche (1024 features pour GoogLeNet standard)
        num_ftrs = self.base_model.fc.in_features
        
        # Remplacer la couche finale (classification) pour nos 4 classes
        # On renomme cette couche 'fc' pour correspondre au nom attendu par les poids du checkpoint.
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        
        # ATTENTION: GoogLeNet a deux couches de classification auxiliaires (aux_logits) 
        # que nous devons désactiver ou gérer si elles sont incluses dans vos poids sauvegardés.
        # En définissant aux_logits=False lors de l'instanciation, nous les ignorons.

    def forward(self, x):
        # Utiliser la méthode forward du modèle de base
        return self.base_model(x)

# =================================================================
# 2. CONFIGURATION ET CACHE
# =================================================================

# Définissez le chemin du fichier de poids de votre modèle PyTorch
# ATTENTION: Utilisez un chemin relatif ou assurez-vous que ce chemin est accessible
# J'ai mis à jour le chemin pour pointer vers le dossier Models comme vous l'avez suggéré précédemment.
MODEL_FILE = os.path.join(os.getcwd(), "Models", "model3.pth") 
NUM_CLASSES = 4
CLASS_LABELS = ["Type 1 (Early Pre-B)", "Type 2 (Benign)", "Type 3 (Pre-B)", "Type 4 (Pro-B)"]

@st.cache_resource
def load_pytorch_model(model_class, path_to_weights, num_classes):
    """Charge et met en cache le modèle PyTorch."""
    if not os.path.exists(path_to_weights):
        st.error(f"❌ Fichier de modèle non trouvé à: {path_to_weights}")
        st.error(f"Chemin recherché: {os.path.abspath(path_to_weights)}")
        return None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instancier le modèle avec la classe CustomGoogLeNet
    model = model_class(num_classes=num_classes)
    
    st.info(f"Chargement des poids sur {device}...")
    
    try:
        # Tente de charger les poids (strict=False est conservé pour ignorer les couches inattendues/manquantes)
        state_dict = torch.load(path_to_weights, map_location=device)
        
        # Si vous avez sauvegardé SEULEMENT le modèle (model.state_dict()), cela fonctionne.
        # Si vous avez sauvegardé un dictionnaire 'checkpoint' contenant 'model_state_dict', décommentez ceci:
        # if 'model_state_dict' in state_dict:
        #     state_dict = state_dict['model_state_dict']
            
        model.base_model.load_state_dict(state_dict, strict=False)

    except RuntimeError as e:
        # Si l'erreur de taille persiste ou si d'autres erreurs apparaissent
        st.error(f"Erreur lors du chargement des poids: {e}")
        st.error("L'architecture du modèle dans le script ne correspond pas exactement au fichier de poids.")
        return None
        
    model.to(device)
    model.eval() # Mode évaluation
    st.success("✅ Modèle chargé avec succès!")
    return model

# Charge le modèle (cette ligne est exécutée une seule fois grâce à @st.cache_resource)
model = load_pytorch_model(CustomGoogLeNet, MODEL_FILE, NUM_CLASSES)

# =================================================================
# 3. LOGIQUE DE PRÉDICTION
# =================================================================

def preprocess_image(image: Image.Image):
    """Prétraitement de l'image (Redimensionnement et conversion en Tensor)."""
    # NOTE IMPORTANTE: Le prétraitement doit correspondre à ce que vous avez fait à l'entraînement!
    
    # 1. Redimensionnement (GoogLeNet utilise souvent 224x224)
    image_resized = image.resize((224, 224))
    
    # 2. Conversion en tableau NumPy (H, W, C) et float32
    img_array = np.array(image_resized, dtype=np.float32)
    
    # 3. Normalisation (Ajout des étapes de normalisation typiques)
    # Si votre modèle a été entraîné avec la normalisation ImageNet:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Normalisation entre 0 et 1 (si non déjà fait par dtype=np.float32)
    img_array = img_array / 255.0
    
    # Normalisation par les stats ImageNet
    img_array = (img_array - mean) / std

    
    # 4. Conversion en Tensor et changement d'ordre des axes (PyTorch: C, H, W)
    # (H, W, C) -> (C, H, W)
    input_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0) # Ajout de la dimension batch
    
    return input_tensor

def predict_image(model, input_tensor):
    """Exécute l'inférence PyTorch."""
    device = next(model.parameters()).device 
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        
    # Appliquer Softmax pour obtenir les probabilités et convertir en NumPy
    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
    predicted_index = np.argmax(probabilities)

    return predicted_index, probabilities

# =================================================================
# 4. INTERFACE UTILISATEUR STREAMLIT
# =================================================================

st.subheader("Classification d'Images Médicales avec GoogLeNet")

if model is None:
    st.warning("Veuillez vérifier que le fichier de modèle (.pth) est présent et que la classe GoogLeNet est correctement définie.")
else:
    uploaded_file = st.file_uploader("Téléchargez une image (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB") # Assure 3 canaux pour GoogLeNet
            st.image(image, caption='Image Téléchargée', width=300)
            st.markdown("---")

            # Prétraitement de l'image
            input_tensor = preprocess_image(image)

            with st.spinner('Analyse par le modèle GoogLeNet...'):
                # Exécution de la prédiction
                predicted_index, probabilities = predict_image(model, input_tensor)

            # Affichage des résultats
            st.subheader("Résultat de la Prédiction")
            st.success(f"Class Prédite: **{CLASS_LABELS[predicted_index]}**")

            # Affichage des probabilités
            results_df = pd.DataFrame({
                'Classe': CLASS_LABELS,
                'Probabilité': probabilities
            }).set_index('Classe')

            st.bar_chart(results_df, color="#37A8E0")
            
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du traitement de l'image: {e}")
            st.warning("Assurez-vous que l'image est un format valide et que votre fonction de prétraitement est correcte.")
