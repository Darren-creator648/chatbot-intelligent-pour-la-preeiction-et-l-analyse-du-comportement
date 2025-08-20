import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class StudentPredictionModel:
    def __init__(self):
        self.score_model = None
        self.grade_model = None
        self.dropout_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def load_and_prepare_data(self, json_file_path):
        """Charge et prépare les données depuis le fichier JSON"""
        students_data = []
        
        print("📖 Chargement des données...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    student = json.loads(line.strip())
                    students_data.append(student)
                except Exception as e:
                    print(f"❌ Ligne ignorée: {e}")
        
        df = pd.DataFrame(students_data)
        print(f"✅ {len(df)} étudiants chargés")
        
        return self._preprocess_data(df)
    
    def _preprocess_data(self, df):
        """Préprocessing des données"""
        print("🔄 Préprocessing des données...")
        
        # Création de nouvelles features
        df['Sleep_Study_Ratio'] = df['Sleep_Hours_per_Night'] / (df['Study_Hours_per_Week'] / 7)
        df['Stress_Performance_Index'] = df['Stress_Level (1-10)'] * (100 - df['Final_Score']) / 100
        df['Engagement_Score'] = (df['Attendance (%)'] + df['Participation_Score']) / 2
        
        # Définir le risque d'abandon basé sur plusieurs critères
        df['Dropout_Risk'] = (
            (df['Final_Score'] < 50) | 
            (df['Attendance (%)'] < 70) | 
            (df['Participation_Score'] < 40) |
            (df['Stress_Level (1-10)'] > 7)
        ).astype(int)
        
        # Features catégorielles à encoder
        categorical_features = ['Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
        
        return df
    
    def prepare_features_and_targets(self, df):
        """Prépare les features et les targets pour l'entraînement"""
        
        # Features numériques
        numeric_features = [
            'Age', 'Attendance (%)', 'Study_Hours_per_Week', 
            'Sleep_Hours_per_Night', 'Stress_Level (1-10)', 'Participation_Score',
            'Sleep_Study_Ratio', 'Stress_Performance_Index', 'Engagement_Score'
        ]
        
        # Features catégorielles encodées
        categorical_encoded = [
            'Gender_encoded', 'Department_encoded', 
            'Extracurricular_Activities_encoded', 'Internet_Access_at_Home_encoded'
        ]
        
        # Toutes les features
        all_features = numeric_features + categorical_encoded
        available_features = [f for f in all_features if f in df.columns]
        
        X = df[available_features]
        
        # Targets
        y_score = df['Final_Score']
        y_grade = df['Grade']
        y_dropout = df['Dropout_Risk']
        
        return X, y_score, y_grade, y_dropout
    
    def train_models(self, json_file_path):
        """Entraîne les modèles de prédiction"""
        print("🚀 Début de l'entraînement des modèles...")
        
        # Chargement et préparation des données
        df = self.load_and_prepare_data(json_file_path)
        X, y_score, y_grade, y_dropout = self.prepare_features_and_targets(df)
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Division train/test
        X_train, X_test, y_score_train, y_score_test = train_test_split(
            X_scaled, y_score, test_size=0.2, random_state=42
        )
        
        _, _, y_grade_train, y_grade_test = train_test_split(
            X_scaled, y_grade, test_size=0.2, random_state=42
        )
        
        _, _, y_dropout_train, y_dropout_test = train_test_split(
            X_scaled, y_dropout, test_size=0.2, random_state=42
        )
        
        # 1. Modèle pour prédire le score final
        print("📊 Entraînement du modèle de prédiction de score...")
        self.score_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.score_model.fit(X_train, y_score_train)
        
        # 2. Modèle pour prédire le grade
        print("🎓 Entraînement du modèle de prédiction de grade...")
        le_grade = LabelEncoder()
        y_grade_encoded = le_grade.fit_transform(y_grade)
        self.label_encoders['Grade'] = le_grade
        
        _, _, y_grade_encoded_train, y_grade_encoded_test = train_test_split(
            X_scaled, y_grade_encoded, test_size=0.2, random_state=42
        )
        
        self.grade_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.grade_model.fit(X_train, y_grade_encoded_train)
        
        # 3. Modèle pour prédire le risque d'abandon
        print("⚠️ Entraînement du modèle de prédiction de risque d'abandon...")
        self.dropout_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'  # Pour gérer le déséquilibre des classes
        )
        self.dropout_model.fit(X_train, y_dropout_train)
        
        # Évaluation des modèles
        self._evaluate_models(X_test, y_score_test, y_grade_encoded_test, y_dropout_test)
        
        self.is_trained = True
        print("✅ Entraînement terminé avec succès!")
        
        return self
    
    def _evaluate_models(self, X_test, y_score_test, y_grade_test, y_dropout_test):
        """Évalue la performance des modèles"""
        print("\n📈 ÉVALUATION DES MODÈLES")
        print("=" * 50)
        
        # Prédictions
        score_pred = self.score_model.predict(X_test)
        grade_pred = self.grade_model.predict(X_test)
        dropout_pred = self.dropout_model.predict(X_test)
        
        # Métriques Score
        score_rmse = np.sqrt(mean_squared_error(y_score_test, score_pred))
        score_mae = np.mean(np.abs(y_score_test - score_pred))
        print(f"🎯 SCORE - RMSE: {score_rmse:.2f}, MAE: {score_mae:.2f}")
        
        # Métriques Grade
        grade_acc = accuracy_score(y_grade_test, grade_pred)
        print(f"🎓 GRADE - Précision: {grade_acc:.3f}")
        
        # Métriques Dropout
        dropout_acc = accuracy_score(y_dropout_test, dropout_pred)
        print(f"⚠️ DROPOUT - Précision: {dropout_acc:.3f}")
        
        print("\nRapport détaillé - Risque d'abandon:")
        print(classification_report(y_dropout_test, dropout_pred, 
                                    target_names=['Pas de risque', 'Risque d\'abandon']))
    
    def predict_student(self, student_data):
        """Prédit le score, grade et risque d'abandon pour un étudiant"""
        if not self.is_trained:
            raise Exception("Le modèle n'est pas encore entraîné!")
        
        # Préparation des données de l'étudiant
        student_features = self._prepare_student_features(student_data)
        student_scaled = self.scaler.transform([student_features])
        
        # Prédictions
        predicted_score = self.score_model.predict(student_scaled)[0]
        predicted_grade_encoded = self.grade_model.predict(student_scaled)[0]
        predicted_grade = self.label_encoders['Grade'].inverse_transform([predicted_grade_encoded])[0]
        dropout_risk = self.dropout_model.predict(student_scaled)[0]
        dropout_probability = self.dropout_model.predict_proba(student_scaled)[0][1]
        
        return {
            'predicted_score': round(predicted_score, 2),
            'predicted_grade': predicted_grade,
            'dropout_risk': bool(dropout_risk),
            'dropout_probability': round(dropout_probability, 3),
            'confidence_level': self._get_confidence_level(dropout_probability)
        }
    
    def _prepare_student_features(self, student_data):
        """Prépare les features d'un étudiant pour la prédiction"""
        # Features calculées
        sleep_study_ratio = student_data['Sleep_Hours_per_Night'] / (student_data['Study_Hours_per_Week'] / 7)
        stress_performance_index = student_data['Stress_Level (1-10)'] * (100 - student_data.get('Final_Score', 75)) / 100
        engagement_score = (student_data['Attendance (%)'] + student_data['Participation_Score']) / 2
        
        # Features de base
        features = [
            student_data['Age'],
            student_data['Attendance (%)'],
            student_data['Study_Hours_per_Week'],
            student_data['Sleep_Hours_per_Night'],
            student_data['Stress_Level (1-10)'],
            student_data['Participation_Score'],
            sleep_study_ratio,
            stress_performance_index,
            engagement_score
        ]
        
        # Encodage des features catégorielles
        categorical_features = ['Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home']
        for feature in categorical_features:
            if feature in student_data and feature in self.label_encoders:
                try:
                    encoded_value = self.label_encoders[feature].transform([str(student_data[feature])])[0]
                    features.append(encoded_value)
                except:
                    # Valeur inconnue, utiliser la plus fréquente (0)
                    features.append(0)
        
        return features
    
    def _get_confidence_level(self, probability):
        """Détermine le niveau de confiance"""
        if probability < 0.3:
            return "Faible risque"
        elif probability < 0.6:
            return "Risque modéré"
        else:
            return "Risque élevé"
    
    def get_feature_importance(self):
        """Retourne l'importance des features pour chaque modèle"""
        if not self.is_trained:
            return None
        
        feature_names = [
            'Age', 'Attendance', 'Study_Hours', 'Sleep_Hours', 
            'Stress_Level', 'Participation', 'Sleep_Study_Ratio',
            'Stress_Performance_Index', 'Engagement_Score',
            'Gender', 'Department', 'Extracurricular', 'Internet_Access'
        ]
        
        return {
            'score_importance': dict(zip(feature_names, self.score_model.feature_importances_)),
            'grade_importance': dict(zip(feature_names, self.grade_model.feature_importances_)),
            'dropout_importance': dict(zip(feature_names, self.dropout_model.feature_importances_))
        }
    
    def save_models(self, base_path="models/"):
        """Sauvegarde les modèles entraînés"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        joblib.dump(self.score_model, f"{base_path}score_model.pkl")
        joblib.dump(self.grade_model, f"{base_path}grade_model.pkl")
        joblib.dump(self.dropout_model, f"{base_path}dropout_model.pkl")
        joblib.dump(self.scaler, f"{base_path}scaler.pkl")
        joblib.dump(self.label_encoders, f"{base_path}label_encoders.pkl")
        
        print(f"✅ Modèles sauvegardés dans {base_path}")
    
    def load_models(self, base_path="models/"):
        """Charge les modèles pré-entraînés"""
        try:
            self.score_model = joblib.load(f"{base_path}score_model.pkl")
            self.grade_model = joblib.load(f"{base_path}grade_model.pkl")
            self.dropout_model = joblib.load(f"{base_path}dropout_model.pkl")
            self.scaler = joblib.load(f"{base_path}scaler.pkl")
            self.label_encoders = joblib.load(f"{base_path}label_encoders.pkl")
            self.is_trained = True
            print("✅ Modèles chargés avec succès!")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer et entraîner le modèle
    model = StudentPredictionModel()
    
    # Entraîner avec vos données
    model.train_models("data/Students Performance Dataset.json")
    
    # Sauvegarder les modèles
    model.save_models()
    
    # Exemple de prédiction pour un nouvel étudiant
    nouvel_etudiant = {
        'Age': 20,
        'Gender': 'Male',
        'Department': 'Engineering',
        'Attendance (%)': 85,
        'Study_Hours_per_Week': 15,
        'Sleep_Hours_per_Night': 7,
        'Stress_Level (1-10)': 6,
        'Participation_Score': 70,
        'Extracurricular_Activities': 'Sports',
        'Internet_Access_at_Home': 'Yes'
    }
    
    predictions = model.predict_student(nouvel_etudiant)
    print("\n🔮 PRÉDICTIONS POUR LE NOUVEL ÉTUDIANT:")
    print("=" * 50)
    print(f"Score prédit: {predictions['predicted_score']}/100")
    print(f"Grade prédit: {predictions['predicted_grade']}")
    print(f"Risque d'abandon: {'OUI' if predictions['dropout_risk'] else 'NON'}")
    print(f"Probabilité d'abandon: {predictions['dropout_probability']:.1%}")
    print(f"Niveau de risque: {predictions['confidence_level']}")
    
    # Afficher l'importance des features
    print("\n📊 IMPORTANCE DES FEATURES:")
    importance = model.get_feature_importance()
    for model_type, features in importance.items():
        print(f"\n{model_type}:")
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features[:5]:
            print(f"  {feature}: {score:.3f}")