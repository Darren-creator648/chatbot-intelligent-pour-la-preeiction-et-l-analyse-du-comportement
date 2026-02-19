import React, { useState } from 'react';
import axios from 'axios';
import {
  Container,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  CircularProgress,
  Box,
  MenuItem,
  Snackbar,
  Alert
} from '@mui/material';
import Markdown from 'react-markdown'; // <-- Import de la librairie pour afficher le Markdown
import rehypeRaw from 'rehype-raw'; // <-- Pour le rendu HTML si nécessaire

// Remarque : Remplacez par l'URL de votre API si elle est différente
const API_BASE_URL = 'http://localhost:5000/api'; 

const PredictionForm = ({ token }) => {
  const [formData, setFormData] = useState({
    // Données pour le modèle de prédiction de grade
    Age: '',
    Gender: '',
    Department: '',
    Parent_Education_Level: '',
    Family_Income_Level: '',
    'Attendance (%)': '',
    Midterm_Score: '',
    Assignments_Avg: '',
    Quizzes_Avg: '',
    Participation_Score: '',
    Projects_Score: '',
    Study_Hours_per_Week: '',
    Sleep_Hours_per_Night: '',
    'Stress_Level (1-10)': '',
    Extracurricular_Activities: '',
    Internet_Access_at_Home: '',

    // Données pour le modèle de prédiction d'anxiété
    sexe: '', // Ce champ peut être dérivé de Gender
    filiere: '', // Ce champ peut être dérivé de Department
    annee_etude: '',
    mgp: '',
    status: '',
    depression: '', 
    crise_de_panique: '', 
    traitement_spe: '', 
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [snackbarOpen, setSnackbarOpen] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({ ...prevData, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);

    const combinedData = {
        // Le backend gère la sélection des 5 meilleures variables pour le grade
        // On envoie donc toutes les données nécessaires aux deux modèles
        Age: parseFloat(formData.Age),
        Gender: formData.Gender,
        Department: formData.Department,
        Parent_Education_Level: formData.Parent_Education_Level,
        Family_Income_Level: formData.Family_Income_Level,
        'Attendance (%)': parseFloat(formData['Attendance (%)']),
        Midterm_Score: parseFloat(formData.Midterm_Score),
        Assignments_Avg: parseFloat(formData.Assignments_Avg),
        Quizzes_Avg: parseFloat(formData.Quizzes_Avg),
        Participation_Score: parseFloat(formData.Participation_Score),
        Projects_Score: parseFloat(formData.Projects_Score),
        Study_Hours_per_Week: parseFloat(formData.Study_Hours_per_Week),
        Sleep_Hours_per_Night: parseFloat(formData.Sleep_Hours_per_Night),
        'Stress_Level (1-10)': parseFloat(formData['Stress_Level (1-10)']),
        Extracurricular_Activities: formData.Extracurricular_Activities,
        Internet_Access_at_Home: formData.Internet_Access_at_Home,
        // Données pour le modèle d'anxiété
        sexe: formData.Gender,
        filiere: formData.Department,
        annee_etude: formData.annee_etude,
        mgp: parseFloat(formData.mgp),
        status: formData.status,
        depression: parseFloat(formData.depression), 
        crise_de_panique: parseFloat(formData.crise_de_panique), 
        traitement_spe: parseFloat(formData.traitement_spe), 
    };

    try {
      const config = {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      };
      
      const response = await axios.post(`${API_BASE_URL}/predict`, combinedData, config);
      setPrediction(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "Une erreur est survenue lors de la prédiction.");
      setSnackbarOpen(true);
    } finally {
      setLoading(false);
    }
  };

  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };

  return (
    <Container component="main" maxWidth="md" sx={{ my: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography component="h1" variant="h4" align="center" gutterBottom>
          Prédictions des performances et de la santé mentale
        </Typography>
        <Typography variant="body1" align="center" paragraph>
          Entrez les informations d'un étudiant pour obtenir des prédictions sur son grade et son niveau d'anxiété.
        </Typography>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            {/* Champs pour les données de performance (Grade) */}
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Informations de Performance Académique</Typography>
            </Grid>
            {/* Les champs correspondants aux 5 variables corrélées restent importants */}
            <Grid item xs={12} sm={6}>
              <TextField label="Âge" name="Age" type="number" fullWidth required value={formData.Age} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField select label="Genre" name="Gender" fullWidth required value={formData.Gender} onChange={handleChange}>
                <MenuItem value="Male">Homme</MenuItem>
                <MenuItem value="Female">Femme</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField select label="Département" name="Department" fullWidth required value={formData.Department} onChange={handleChange}>
                <MenuItem value="Science">Science</MenuItem>
                <MenuItem value="Engineering">Ingénierie</MenuItem>
                <MenuItem value="Arts">Arts</MenuItem>
                <MenuItem value="Commerce">Commerce</MenuItem>
                <MenuItem value="Business">Business</MenuItem>
                <MenuItem value="Law">Droit</MenuItem>
                <MenuItem value="Medicine">Médecine</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField label="Présence (%)" name="Attendance (%)" type="number" fullWidth required value={formData['Attendance (%)']} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField label="Note mi-session" name="Midterm_Score" type="number" fullWidth required value={formData.Midterm_Score} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField label="Moyenne des devoirs" name="Assignments_Avg" type="number" fullWidth required value={formData.Assignments_Avg} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField label="Moyenne des quiz" name="Quizzes_Avg" type="number" fullWidth required value={formData.Quizzes_Avg} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField label="Score de participation" name="Participation_Score" type="number" fullWidth required value={formData.Participation_Score} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField label="Score des projets" name="Projects_Score" type="number" fullWidth required value={formData.Projects_Score} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField label="Heures d'étude par semaine" name="Study_Hours_per_Week" type="number" fullWidth required value={formData.Study_Hours_per_Week} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField label="Heures de sommeil par nuit" name="Sleep_Hours_per_Night" type="number" fullWidth required value={formData.Sleep_Hours_per_Night} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField label="Niveau de stress (1-10)" name="Stress_Level (1-10)" type="number" fullWidth required value={formData['Stress_Level (1-10)']} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField select label="Activités parascolaires" name="Extracurricular_Activities" fullWidth required value={formData.Extracurricular_Activities} onChange={handleChange}>
                <MenuItem value="Yes">Oui</MenuItem>
                <MenuItem value="No">Non</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField select label="Accès Internet à la maison" name="Internet_Access_at_Home" fullWidth required value={formData.Internet_Access_at_Home} onChange={handleChange}>
                <MenuItem value="Yes">Oui</MenuItem>
                <MenuItem value="No">Non</MenuItem>
              </TextField>
            </Grid>
            
            {/* Champs pour les données de santé mentale (Anxiété) */}
            <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>Informations sur la Santé Mentale</Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField select label="Année d'étude" name="annee_etude" fullWidth required value={formData.annee_etude} onChange={handleChange}>
                <MenuItem value="Year 1">1ère année</MenuItem>
                <MenuItem value="Year 2">2ème année</MenuItem>
                <MenuItem value="Year 3">3ème année</MenuItem>
                <MenuItem value="Year 4">4ème année</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
                <TextField label="Moyenne Générale (MGP)" name="mgp" type="number" fullWidth required value={formData.mgp} onChange={handleChange} />
            </Grid>
            <Grid item xs={12} sm={6}>
                <TextField select label="Statut matrimonial" name="status" fullWidth required value={formData.status} onChange={handleChange}>
                    <MenuItem value="Single">Célibataire</MenuItem>
                    <MenuItem value="Married">Marié(e)</MenuItem>
                </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
                <TextField select label="Souffrez-vous de dépression ?" name="depression" fullWidth required value={formData.depression} onChange={handleChange}>
                    <MenuItem value="1">Oui</MenuItem>
                    <MenuItem value="0">Non</MenuItem>
                </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
                <TextField select label="Avez-vous des crises de panique ?" name="crise_de_panique" fullWidth required value={formData.crise_de_panique} onChange={handleChange}>
                    <MenuItem value="1">Oui</MenuItem>
                    <MenuItem value="0">Non</MenuItem>
                </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
                <TextField select label="Avez-vous consulté un spécialiste ?" name="traitement_spe" fullWidth required value={formData.traitement_spe} onChange={handleChange}>
                    <MenuItem value="1">Oui</MenuItem>
                    <MenuItem value="0">Non</MenuItem>
                </TextField>
            </Grid>
          </Grid>
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
            <Button type="submit" variant="contained" color="primary" disabled={loading}>
              {loading ? <CircularProgress size={24} /> : 'Prédire'}
            </Button>
          </Box>
        </form>

        {prediction && (
          <Box sx={{ mt: 4, textAlign: 'center' }}>
            <Typography variant="h5" color="text.secondary">Résultats de la prédiction</Typography>
            <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
              <Typography variant="h6">Grade Prédit : {prediction.grade_prediction}</Typography>
              <Typography variant="h6">Niveau d'Anxiété Prédit : {prediction.anxiety_prediction}</Typography>
              
              {/* NOUVEAU : Affichage de l'analyse générée */}
              <Box sx={{ mt: 3, textAlign: 'left' }}>
                <Typography variant="h6">Analyse Personnalisée :</Typography>
                <Markdown rehypePlugins={[rehypeRaw]}>{prediction.analysis}</Markdown>
              </Box>
            </Paper>
          </Box>
        )}
        
        
        <Snackbar open={snackbarOpen} autoHideDuration={6000} onClose={handleSnackbarClose}>
          <Alert onClose={handleSnackbarClose} severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        </Snackbar>
      </Paper>
    </Container>
  );
};

export default PredictionForm;