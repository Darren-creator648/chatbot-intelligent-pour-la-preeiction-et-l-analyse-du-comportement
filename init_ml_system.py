import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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

def train_and_save_grade_model():
    """Entraîne et sauvegarde le modèle de prédiction du grade."""
    try:
        # 1. Charger les données
        df = load_data('data/Students Performance Dataset.csv')
        if df is None:
            return
            
        print("✅ Données chargées avec succès pour la prédiction du grade.")

        # --- Définition des features et de la cible ---
        target = 'Grade'
        
        # Colonnes à ignorer pour l'entraînement
        ignore_cols = [
            'Student_ID', 
            'First_Name', 
            'Last_Name', 
            'Email', 
            'Final_Score', 
            'Total_Score'
        ]
        
        # Séparer les features (X) et la cible (y)
        y = df[target]
        X = df.drop(columns=ignore_cols + [target], errors='ignore')

        # Identifier automatiquement les variables numériques et catégorielles
        numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nVariables numériques détectées: {numerical_features}")
        print(f"Variables catégorielles détectées: {categorical_features}")

        # --- Création du pipeline de prétraitement ---
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # --- Création du pipeline d'entraînement avec SMOTE ---
        # Utiliser l'option 'all' pour suréchantillonner toutes les classes minoritaires
        model_pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(sampling_strategy='all', random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # --- Entraînement et évaluation du modèle ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraîner le pipeline complet
        model_pipeline.fit(X_train, y_train)

        # Faire des prédictions sur le jeu de test
        y_pred = model_pipeline.predict(X_test)

        # Évaluer le modèle
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\n--- Évaluation du modèle de Prédiction du Grade ---")
        print(f"Précision (Accuracy): {accuracy:.2f}")
        print("\nRapport de classification:\n", report)

        # Sauvegarder le modèle entraîné
        save_model(model_pipeline, 'grade_prediction_model.joblib')
        
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")

if __name__ == "__main__":
    train_and_save_grade_model()