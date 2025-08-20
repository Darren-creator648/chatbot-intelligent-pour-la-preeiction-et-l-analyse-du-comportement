from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from collections import Counter
from services.ai_service import ai_service
from database import db_instance
from config import Config
from services.prediction_service import handle_predictions

stats_bp = Blueprint('stats', __name__)

@stats_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_stats():
    return jsonify({
        "total_students": len(ai_service.student_profiles),
        "total_chunks": len(ai_service.chunks),
        "model_name": Config.EMBEDDING_MODEL,
        "system_status": "operational" if ai_service.is_initialized() else "initializing"
    })

@stats_bp.route('/analytics', methods=['GET'])
@jwt_required()
def get_analytics():
    try:
        user_id = get_jwt_identity()
        analytics = db_instance.get_analytics_data(user_id)
        return jsonify(analytics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@stats_bp.route('/student-analytics', methods=['GET'])
@jwt_required()
def get_student_analytics():
    try:
        if not ai_service.raw_students_data:
            return jsonify({"error": "Données étudiants non chargées"}), 503
        
        # Analyse des données étudiantes
        departments = Counter(student['Department'] for student in ai_service.raw_students_data)
        grades = Counter(student['Grade'] for student in ai_service.raw_students_data)
        gender_distribution = Counter(student['Gender'] for student in ai_service.raw_students_data)
        
        scores = [student['Final_Score'] for student in ai_service.raw_students_data]
        attendance = [student['Attendance (%)'] for student in ai_service.raw_students_data]
        stress_levels = [student['Stress_Level (1-10)'] for student in ai_service.raw_students_data]
        
        performance_data = []
        for student in ai_service.raw_students_data:
            performance_data.append({
                'name': f"{student['First_Name']} {student['Last_Name']}",
                'attendance': student['Attendance (%)'],
                'final_score': student['Final_Score'],
                'stress_level': student['Stress_Level (1-10)'],
                'study_hours': student['Study_Hours_per_Week'],
                'sleep_hours': student['Sleep_Hours_per_Night'],
                'department': student['Department'],
                'grade': student['Grade']
            })
        
        at_risk_students = [
            student for student in ai_service.raw_students_data
            if student['Final_Score'] < 60 or student['Attendance (%)'] < 70
        ]
        
        return jsonify({
            'departments': dict(departments),
            'grades': dict(grades),
            'gender_distribution': dict(gender_distribution),
            'performance_stats': {
                'avg_score': sum(scores) / len(scores),
                'avg_attendance': sum(attendance) / len(attendance),
                'avg_stress': sum(stress_levels) / len(stress_levels)
            },
            'performance_data': performance_data,
            'at_risk_count': len(at_risk_students),
            'total_students': len(ai_service.raw_students_data)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@stats_bp.route('/analytics/detailed', methods=['GET'])
@jwt_required()
def get_detailed_analytics():
    """Analytics détaillées avec plus de métriques"""
    try:
        user_id = get_jwt_identity()
        
        if not ai_service.raw_students_data:
            return jsonify({"error": "Données étudiants non chargées"}), 503
        
        students = ai_service.raw_students_data
        
        score_ranges = {
            'excellent': len([s for s in students if s['Final_Score'] >= 80]),
            'good': len([s for s in students if 60 <= s['Final_Score'] < 80]),
            'average': len([s for s in students if 40 <= s['Final_Score'] < 60]),
            'poor': len([s for s in students if s['Final_Score'] < 40])
        }
        
        stress_distribution = Counter(student['Stress_Level (1-10)'] for student in students)
        
        high_stress_low_performance = len([
            s for s in students 
            if s['Stress_Level (1-10)'] >= 7 and s['Final_Score'] < 60
        ])
        
        low_attendance_low_performance = len([
            s for s in students 
            if s['Attendance (%)'] < 70 and s['Final_Score'] < 60
        ])
        
        user_analytics = db_instance.get_analytics_data(user_id)
        
        return jsonify({
            'score_ranges': score_ranges,
            'stress_distribution': dict(stress_distribution),
            'correlations': {
                'high_stress_low_performance': high_stress_low_performance,
                'low_attendance_low_performance': low_attendance_low_performance
            },
            'user_activity': user_analytics,
            'data_quality': {
                'total_records': len(students),
                'complete_records': len([s for s in students if all(s.values())]),
                'departments_count': len(set(s['Department'] for s in students))
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Route pour les prédictions des modèles ---
@stats_bp.route('/predict', methods=['POST'])
@jwt_required()
def predict_student_data():
    """
    Accepte les données d'un étudiant et retourne les prédictions des modèles.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données JSON manquantes."}), 400
        
        # Appel au service de prédiction qui renvoie un dictionnaire
        predictions = handle_predictions(data)

        # Gérer les erreurs de prédiction
        if "error" in predictions:
            status_code = 503 if "initialisé" in predictions["error"] else 500
            return jsonify(predictions), status_code
        
        # Renvoyer la réponse combinée avec un message de succès
        predictions["message"] = "Prédictions générées avec succès."
        return jsonify(predictions)
    
    except Exception as e:
        # Gérer les erreurs inattendues, comme des données d'entrée incorrectes
        return jsonify({"error": f"Une erreur interne est survenue : {str(e)}"}), 500