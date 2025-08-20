import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
from sklearn.impute import SimpleImputer

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

def train_and_save_anxiety_model():
    """Entraîne et sauvegarde le modèle de prédiction de l'anxiété."""
    try:
        # 1. Charger les données
        df_health = load_data('data/Student Mental health.csv')
        if df_health is None:
            return
            
        print("✅ Données chargées avec succès pour la prédiction de l'anxiété.")

        # 2. Nettoyage et prétraitement
        df_health.rename(columns={
            'Choose your gender': 'sexe',
            'What is your course?': 'filiere',
            'Your current year of Study': 'annee_etude',
            'What is your CGPA?': 'mgp',
            'Marital status': 'status',
            'Do you have Depression?': 'depression',
            'Do you have Anxiety?': 'anxiete',
            'Do you have Panic attack?': 'crise_de_panique',
            'Did you seek any specialist for a treatment?': 'traitement_spe'
        }, inplace=True)

        df_health = df_health.drop(columns=['Timestamp'], errors='ignore')
        
        # Le mapping des colonnes binaires est toujours nécessaire
        binary_cols = ['anxiete', 'depression', 'crise_de_panique', 'traitement_spe']
        for col in binary_cols:
            df_health[col] = df_health[col].map({'Yes': 1, 'No': 0})
        
        def convert_cgpa(cgpa_range):
            if isinstance(cgpa_range, str) and ' - ' in cgpa_range:
                low, high = map(float, cgpa_range.split(' - '))
                return (low + high) / 2
            return None
        df_health['mgp'] = df_health['mgp'].apply(convert_cgpa)
        
        # --- Définition des features et de la cible ---
        target = 'anxiete'
        
        # --- CORRECTION : Définir toutes les colonnes comme features, sauf la cible ---
        X = df_health.drop(columns=[target], errors='ignore')
        y = df_health[target]

        numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Les colonnes de santé mentale sont des variables numériques
        # Vous pouvez les traiter comme des features numériques
        # Assurez-vous simplement qu'elles ne sont pas dans la cible
        
        print(f"\nVariables numériques détectées: {numerical_features}")
        print(f"Variables catégorielles détectées: {categorical_features}")

        # --- Création du pipeline de prétraitement ---
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # --- Création du pipeline d'entraînement avec SMOTE ---
        model_pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(sampling_strategy='all', random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # --- Entraînement et évaluation du modèle ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['No Anxiety', 'Anxiety'])

        print(f"\n--- Évaluation du modèle de Prédiction de l'Anxiété ---")
        print(f"Précision (Accuracy): {accuracy:.2f}")
        print("\nRapport de classification:\n", report)

        save_model(model_pipeline, 'anxiety_prediction_model.joblib')
        
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")

if __name__ == "__main__":
    train_and_save_anxiety_model()