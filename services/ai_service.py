import json
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess
import re
from sentence_transformers import SentenceTransformer
import language_tool_python
from config import Config

class AIService:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.student_profiles = []
        self.raw_students_data = []
    
    def load_students_from_json(self, json_file):
        """Charge les données étudiants depuis un fichier JSON"""
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
        """Récupère le contexte pertinent pour une question"""
        if chunks is None:
            chunks = self.chunks
        if model is None:
            model = self.model
        if index is None:
            index = self.index
        if k is None:
            k = Config.CONTEXT_K
            
        # Encodage question
        question_vec = model.encode([question])
        question_vec = np.array(question_vec).astype('float32')

        # Recherche FAISS (k * 5 pour avoir plus de candidats)
        distances, indices = index.search(question_vec, k * 5)

        # Récupérer les chunks candidats
        candidate_indices = [i for i in indices[0] if i >= 0]
        candidate_chunks = [chunks[i] for i in candidate_indices]

        # Embeddings candidats (à calculer ou récupérer si stockés)
        candidate_embeddings = model.encode(candidate_chunks)
        candidate_embeddings = np.array(candidate_embeddings).astype('float32')

        # Calcul similarité cosinus
        sim_scores = cosine_similarity(question_vec, candidate_embeddings)[0]

        # Trier candidats par similarité cosinus décroissante
        sorted_candidates = sorted(zip(candidate_chunks, sim_scores), key=lambda x: x[1], reverse=True)

        # Garde les top k avec meilleur score cosinus
        top_chunks = [chunk for chunk, score in sorted_candidates[:k]]

        return "\n".join(top_chunks)
    
    def generate_with_ollama(self, prompt, model_name=None):
        """Génère une réponse avec Ollama"""
        if model_name is None:
            model_name = Config.OLLAMA_MODEL
            
        try:
            result = subprocess.run(
                ["ollama", "run", model_name],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
            )
            return result.stdout.decode("utf-8").strip()
        except subprocess.TimeoutExpired:
            return "Désolé, la génération a pris trop de temps."
        except Exception as e:
            return f"Erreur lors de la génération : {str(e)}"
    
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
    
    def initialize_system(self):
        """Initialise le système IA"""
        try:
            print("🔍 Initialisation du système...")
            
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
        return all([self.model, self.index, self.chunks])

# Instance globale
ai_service = AIService()