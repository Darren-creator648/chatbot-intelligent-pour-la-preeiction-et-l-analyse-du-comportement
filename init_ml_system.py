import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- Fonctions utilitaires ---

def load_data(file_path):
    """Charge les données depuis un fichier CSV."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{file_path}' est introuvable. Veuillez vérifier le chemin.")
        return None

def save_model(model, filename):
    """Sauvegarde un modèle dans un fichier."""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, os.path.join('models', filename))
    print(f"✅ Modèle '{filename}' sauvegardé avec succès dans le dossier 'models/'.")

# --- Fonction principale d'entraînement et de sauvegarde (avec fuite de données) ---

def train_and_save_leaky_model_correlated():
    """Entraîne et sauvegarde un modèle après avoir ré-échantillonné AVANT la division, en utilisant les 5 variables les plus corrélées."""
    try:
        # 1. Charger les données
        df = load_data('data/Students Performance Dataset.csv')
        if df is None:
            return
            
        print("✅ Données chargées avec succès pour la prédiction du grade.")

        # 2. Définir les features et la cible
        target = 'Grade'
        ignore_cols = [
            'Student_ID', 
            'First_Name', 
            'Last_Name', 
            'Email', 
            'Final_Score', 
            'Total_Score'
        ]
        
        y = df[target]
        X = df.drop(columns=ignore_cols + [target], errors='ignore')

        # Encodage de la cible pour le calcul de corrélation
        label_encoder = LabelEncoder()
        y_encoded_for_corr = label_encoder.fit_transform(y)
        
        # Identifier les variables numériques
        numerical_features_all = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features_all = X.select_dtypes(include=['object']).columns.tolist()

        # 3. Calculer les corrélations et sélectionner les meilleures variables
        print("\n⏳ Calcul des corrélations pour la sélection de variables...")
        df_temp = X.copy()
        df_temp[target] = y_encoded_for_corr
        
        correlations = df_temp.corr(method='spearman', numeric_only=True)[target].abs().sort_values(ascending=False)
        
        # Sélectionner les 5 plus corrélées, en excluant la cible elle-même
        correlated_features = correlations.index[1:6].tolist()
        print(f"✅ 5 variables numériques les plus corrélées avec le grade: {correlated_features}")
        
        # 4. Définir les nouvelles listes de features pour le prétraitement
        numerical_features = correlated_features
        # On garde toutes les variables catégorielles
        categorical_features = categorical_features_all

        # 5. Création du pré-processeur
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        # Transformation des données complètes pour appliquer SMOTE
        X_preprocessed = preprocessor.fit_transform(X)
        
        print("\n⏳ Application de SMOTE sur l'ensemble complet de données (FUITE DE DONNÉES)...")
        # 6. Application de SMOTE sur l'ensemble de données complet
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y_encoded_for_corr)
        
        print(f"✅ Jeu de données ré-échantillonné. Taille initiale: {len(X)}. Nouvelle taille: {len(X_resampled)}")

        # 7. Division des données ré-échantillonnées
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # 8. Entraînement d'un classifieur sans pipeline Imb
        classifier = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        classifier.fit(X_train, y_train)

        # 9. Évaluation finale
        y_pred = classifier.predict(X_test)
        
        # Inverser l'encodage pour l'affichage des résultats
        y_test_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)

        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        report = classification_report(y_test_labels, y_pred_labels)

        print(f"\n--- Évaluation du Modèle (Résultats trompeurs en raison de la fuite de données) ---")
        print(f"Précision (Accuracy) finale: {accuracy:.4f}")
        print("\nRapport de classification:\n", report)

        # 10. Sauvegarde du modèle et du pré-processeur
        save_model(classifier, 'leaky_model_correlated_classifier.joblib')
        save_model(preprocessor, 'leaky_model_correlated_preprocessor.joblib')
        
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")

if __name__ == "__main__":
    train_and_save_leaky_model_correlated()