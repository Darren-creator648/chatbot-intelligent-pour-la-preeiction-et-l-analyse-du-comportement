import json
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess
import re
from sentence_transformers import SentenceTransformer
import language_tool_python
from config import Config
import joblib 
import pandas as pd
import google.generativeai as genai 
import openai 

class AIService:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.student_profiles = []
        self.raw_students_data = []
        # Renommer les attributs pour correspondre aux modèles entraînés
        self.grade_prediction_model = None
        self.grade_prediction_preprocessor = None # <-- Ajout du pré-processeur
        self.anxiety_prediction_model = None
        
        # Configuration de l'API OpenRouter
        try:
            if Config.OPENROUTER_API_KEY:
                self.api_client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=Config.OPENROUTER_API_KEY,
                    
                )
                print("✅ Client OpenRouter configuré avec succès.")
            else:
                print("⚠️ Clé API OpenRouter non configurée. La génération d'analyse ne fonctionnera pas.")
                self.api_client = None
        except Exception as e:
            print(f"❌ Erreur lors de la configuration de l'API OpenRouter : {e}")
            self.api_client = None
        
    
    def load_students_from_json(self, json_file):
        """Charge les données étudiants depuis un fichier JSON Ligne par Ligne"""
        students = []
        raw_students_data = []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    student = json.loads(line.strip())
                    raw_students_data.append(student)
                    
                    desc = (
                        f"{student['First_Name']} {student['Last_Name']} ({student['Gender']}, {student['Age']} ans), "
                        f"du département {student['Department']}, a un taux de présence de {student['Attendance (%)']}%, "
                        f"une note finale de {student['Final_Score']}/100, une participation de {student['Participation_Score']}/100, "
                        f"et un niveau de stress de {student['Stress_Level (1-10)']}/10. "
                        f"Étudie {student['Study_Hours_per_Week']} heures/semaine, dort {student['Sleep_Hours_per_Night']}h/nuit. "
                        f"Activités extrascolaires : {student['Extracurricular_Activities']}. "
                        f"Accès internet : {student['Internet_Access_at_Home']}. "
                        f"Note globale : {student['Grade']}."
                    )
                    students.append(desc)
                except Exception as e:
                    print(f"❌ Ligne ignorée (erreur : {e})")
        
        self.raw_students_data = raw_students_data
        return students

    # === Découpage des textes en chunks ===
    def split_text(self, texts, chunk_size=None):
        """Découpe les textes en chunks"""
        if chunk_size is None:
            chunk_size = Config.CHUNK_SIZE
        
        chunks = []
        for text in texts:
            words = text.split()
            chunks.extend([" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)])
        return chunks

    # === Création de l'index FAISS ===
    def create_faiss_index(self, chunks, model):
        """Crée l'index FAISS"""
        vectors = model.encode(chunks)
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return index, vectors

    # === Récupération du contexte ===
    def retrieve_context(self, question, chunks=None, model=None, index=None, k=None):
        """Récupère le contexte pertinent pour une question et retourne le score de similarité max"""
        if chunks is None:
            chunks = self.chunks
        if model is None:
            model = self.model
        if index is None:
            index = self.index
        if k is None:
            k = Config.CONTEXT_K
            
        question_vec = model.encode([question])
        question_vec = np.array(question_vec).astype('float32')

        distances, indices = index.search(question_vec, k * 5)

        candidate_indices = [i for i in indices[0] if i >= 0]
        candidate_chunks = [chunks[i] for i in candidate_indices]

        candidate_embeddings = model.encode(candidate_chunks)
        candidate_embeddings = np.array(candidate_embeddings).astype('float32')

        sim_scores = cosine_similarity(question_vec, candidate_embeddings)[0]

        sorted_candidates = sorted(zip(candidate_chunks, sim_scores), key=lambda x: x[1], reverse=True)

        top_chunks = [chunk for chunk, score in sorted_candidates[:k]]
        
        max_similarity_score = float(sorted_candidates[0][1]) if sorted_candidates else 0.0

        context = "\n".join(top_chunks)
        return context, max_similarity_score
    
    # === NOUVELLE MÉTHODE AVEC OPENROUTER ===
    def generate_with_openrouter(self, prompt: str) -> str:
        """Génère une réponse avec l'API d'OpenRouter."""
        if not self.api_client:
            return "Une erreur est survenue lors de l'initialisation du service d'IA."
        
        try:
            response = self.api_client.chat.completions.create(
                model=Config.OPENROUTER_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Erreur lors de l'appel à l'API OpenRouter : {e}")
            return "Une erreur est survenue lors de la génération de l'analyse."
            
    # === Post-traitement de la réponse ===
    
    def postprocess_response(self, response: str) -> str:
        """Corrige la réponse avec LanguageTool"""
        try:
            tool = language_tool_python.LanguageTool('fr-FR')
            matches = tool.check(response)
            corrected = language_tool_python.utils.correct(response, matches)
            return corrected
        except Exception as e:
            print(f"⚠️ Erreur pendant la correction locale : {e}")
            return response
    
    def is_nonsense(self, message):
        """Détecte les messages sans sens"""
        message = message.strip().lower()
        if len(message) < 4:
            return True
        if not re.search(r'[aeiouy]', message):
            return True
        if re.fullmatch(r'(.)\1{2,}', message):
            return True
        if re.fullmatch(r'([a-z]{1,2})\1{2,}', message):
            return True
        return False

    # --- NOUVEAU : Chargement des modèles de prédiction ---
    def load_prediction_models(self):
        """Charge les modèles de prédiction depuis le disque."""
        try:
            # Charger les modèles avec les noms de fichiers corrects
            self.grade_prediction_preprocessor = joblib.load('models/leaky_model_correlated_preprocessor.joblib')
            self.grade_prediction_model = joblib.load('models/leaky_model_correlated_classifier.joblib')
            
            self.anxiety_prediction_model = joblib.load('models/anxiety_prediction_model.joblib')
            
            print("✅ Modèles de prédiction chargés avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles de prédiction : {e}")
            
    # --- NOUVEAU : Méthodes de prédiction ---
    def predict_grade(self, input_data):
        """
        Prédit le grade à partir des données d'entrée.
        Utilise le pré-processeur et le classifieur distincts.
        """
        if self.grade_prediction_model is None or self.grade_prediction_preprocessor is None:
            return {"error": "Modèle de prédiction de grade non chargé."}
        
        try:
            df = pd.DataFrame([input_data])
            # Appliquer le pré-processeur avant de faire la prédiction
            df_preprocessed = self.grade_prediction_preprocessor.transform(df)
            
            grade_numerical = self.grade_prediction_model.predict(df_preprocessed)[0]
            
            # Inverser l'encodage pour afficher la lettre du grade
            grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
            predicted_grade = grade_map.get(grade_numerical, 'Indéterminé')
            
            return {"grade": predicted_grade}
        
        except Exception as e:
            return {"error": f"Erreur lors de la prédiction du grade : {e}"}

    def predict_anxiety(self, input_data):
        """Prédit le niveau d'anxiété à partir des données d'entrée."""
        if self.anxiety_prediction_model is None:
            return {"error": "Modèle d'anxiété non chargé."}
        
        try:
            df = pd.DataFrame([input_data])
            # Le modèle est un pipeline, il gère le prétraitement tout seul
            prediction = self.anxiety_prediction_model.predict(df)[0]
            
            # Convertir la prédiction numérique en étiquette textuelle
            result = "anxiété" if prediction == 1 else "pas d'anxiété"
            
            return {"anxiety_level": result}
            
        except Exception as e:
            return {"error": f"Erreur lors de la prédiction de l'anxiété : {e}"}
            

    def initialize_system(self):
        """Initialise le système IA"""
        try:
            print("🔍 Initialisation du système...")
            
            # --- Chargement des modèles de prédiction en premier ---
            self.load_prediction_models()

            print("🔍 Chargement des profils étudiants...")
            self.student_profiles = self.load_students_from_json(Config.STUDENTS_DATA_FILE)
            
            print("🔍 Découpage des descriptions en chunks...")
            self.chunks = self.split_text(self.student_profiles)
            
            print("🔍 Chargement du modèle d'embeddings...")
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            
            print("🔍 Création de l'index FAISS...")
            self.index, _ = self.create_faiss_index(self.chunks, self.model)
            
            print("✅ Système initialisé avec succès!")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation : {e}")
            return False
    
    def is_initialized(self):
        """Vérifie si le système est initialisé"""
        return all([self.model, self.index, self.chunks, self.grade_prediction_model, self.anxiety_prediction_model])
    # --- NOUVELLE MÉTHODE : Génération d'analyse et de recommandations ---
    def generate_analysis(self, input_data, grade_prediction, anxiety_prediction):
        """
        Génère une analyse textuelle et des recommandations personnalisées
        à partir des données d'entrée et des prédictions.
        """
        # Construction du contexte pour le prompt
        context = (
            f"Analyse pour un étudiant avec les caractéristiques suivantes :\n"
            f"- Âge : {input_data.get('Age')} ans\n"
            f"- Sexe : {input_data.get('Gender')}\n"
            f"- Département : {input_data.get('Department')}\n"
            f"- Heures d'étude par semaine : {input_data.get('Study_Hours_per_Week')}h\n"
            f"- Heures de sommeil par nuit : {input_data.get('Sleep_Hours_per_Night')}h\n"
            f"- Niveau de stress : {input_data.get('Stress_Level (1-10)')}/10\n"
            f"- Activités parascolaires : {input_data.get('Extracurricular_Activities')}\n"
            f"- Accès internet : {input_data.get('Internet_Access_at_Home')}\n"
            f"- Note mi-session : {input_data.get('Midterm_Score')}\n"
            f"- Moyenne des devoirs : {input_data.get('Assignments_Avg')}\n"
            f"- Moyenne des quiz : {input_data.get('Quizzes_Avg')}\n"
            f"- Score de participation : {input_data.get('Participation_Score')}\n"
            f"- Score des projets : {input_data.get('Projects_Score')}\n"
            f"- Taux de présence : {input_data.get('Attendance (%)')}%\n"
            f"- Année d'étude : {input_data.get('annee_etude')}\n"
            f"- Moyenne générale (MGP) : {input_data.get('mgp')}\n"
            f"- Statut matrimonial : {input_data.get('status')}\n"
            f"- Dépression : {'Oui' if input_data.get('depression') == 1 else 'Non'}\n"
            f"- Crises de panique : {'Oui' if input_data.get('crise_de_panique') == 1 else 'Non'}\n"
            f"- Traitement spécialisé : {'Oui' if input_data.get('traitement_spe') == 1 else 'Non'}\n\n"
            f"Le grade prédit est : {grade_prediction}\n"
            f"Le niveau d'anxiété prédit est : {anxiety_prediction}\n\n"
        )
        
        # Instructions pour le modèle de langage
        instructions = (
            "En te basant sur ces informations, rédige une analyse claire, concise et professionnelle. "
            "Fournis des recommandations personnalisées et pratiques pour l'étudiant, en te concentrant sur "
            "les aspects de la santé mentale et des performances académiques. "
            "Structure ta réponse de la manière suivante :\n\n"
            "**Analyse et Synthèse :** Explique brièvement les points forts et faibles de la situation de l'étudiant. "
            "**Recommandations :** Propose 3 à 5 recommandations personnalisées pour améliorer son grade et sa santé mentale. "
            "Utilise un ton encourageant et empathique. Rédige en français. Assure-toi que les recommandations sont directement liées aux données fournies (par exemple, si le stress est élevé, recommande des techniques de gestion du stress ; si le sommeil est faible, recommande d'améliorer la qualité du sommeil)."
        )

        prompt = context + instructions
        
        try:
            generated_text = self.generate_with_openrouter(prompt)
            # Post-traitement pour améliorer la qualité du texte
            corrected_text = self.postprocess_response(generated_text)
            return corrected_text
        except Exception as e:
            print(f"❌ Erreur lors de la génération de l'analyse : {e}")
            return "Une erreur est survenue lors de la génération de l'analyse."

# Instance globale
ai_service = AIService()