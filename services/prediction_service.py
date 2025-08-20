# services/prediction_service.py

from services.ai_service import ai_service

def handle_predictions(data):
    """
    Gère la logique de prédiction et renvoie un dictionnaire de résultats ou une erreur.
    """
    if not ai_service.is_initialized():
        return {"error": "Le système IA n'est pas complètement initialisé. Veuillez réessayer plus tard."}

    grade_prediction = ai_service.predict_grade(data)
    anxiety_prediction = ai_service.predict_anxiety(data)
   

    if "error" in grade_prediction:
        return {"error": grade_prediction["error"]}
    if "error" in anxiety_prediction:
        return {"error": anxiety_prediction["error"]}
    

    return {
        "grade_prediction": grade_prediction["grade"],
        "anxiety_prediction": anxiety_prediction["anxiety_level"],
    }