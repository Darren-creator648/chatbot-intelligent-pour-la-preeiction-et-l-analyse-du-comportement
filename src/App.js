import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, AlertCircle, CheckCircle, Loader, BarChart3, Users, Brain, Download, TrendingUp, PieChart, Activity, LogOut, UserCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart as RechartsPieChart, Pie, Cell, ScatterChart, Scatter } from 'recharts';
// Import du nouveau composant
import PredictionForm from './PredictionForm';

const StudentAnalysisApp = () => {
  // États d'authentification
  const [isAuthenticated, setIsAuthenticated] = useState(false);//Gère l’état de connexion de l’utilisateur
  const [user, setUser] = useState(null);// Gère les informations de l'utilisateur connecté
  const [authLoading, setAuthLoading] = useState(true);// Gère l'état de chargement de l'authentification
  const [showLogin, setShowLogin] = useState(true);// Gère l'affichage du formulaire de connexion ou d'inscription

  // États de l'application principale
  const [messages, setMessages] = useState([]);// Gère les messages de la conversation
  const [inputMessage, setInputMessage] = useState('');// Gère le message saisi par l'utilisateur
  const [isLoading, setIsLoading] = useState(false);// Gère l'état de chargement lors de l'envoi d'un message
  const [systemStatus, setSystemStatus] = useState({ status: 'checking', model_loaded: false });// Gère l'état du système (santé et chargement du modèle)
  const [stats, setStats] = useState(null);// Gère les statistiques générales de l'application
  const [sessionId, setSessionId] = useState(null);// Gère l'ID de session pour la conversation en cours
  const [activeTab, setActiveTab] = useState('chat');// Gère l'onglet actif (chat ou analytics)
  const [analytics, setAnalytics] = useState(null);// Gère les données d'analyse de l'application
  const [studentAnalytics, setStudentAnalytics] = useState(null);// Gère les données d'analyse des étudiants
  const messagesEndRef = useRef(null);// Référence pour le défilement automatique vers le bas de la conversation
  const inputRef = useRef(null);// Référence pour le champ de saisie du message

  // États pour les formulaires d'authentification
  const [authForm, setAuthForm] = useState({
    email: '',
    password: '',
    full_name: '',
    role: 'student'
  });// Gère les données du formulaire d'authentification
  const [authError, setAuthError] = useState('');

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4', '#84CC16', '#F97316'];
  const API_BASE = 'http://localhost:5000/api';

  // Vérification de l'authentification au chargement
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      fetchProfile(token);
    } else {
      setAuthLoading(false);
    }
  }, []);

  // Initialisation après authentification
  useEffect(() => {
    if (isAuthenticated) {
      const newSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      setSessionId(newSessionId);
      checkSystemHealth();
      fetchStats();
      fetchAnalytics();
      fetchStudentAnalytics();
      
      setMessages([{
        id: 1,
        type: 'bot',
        content: `🎓 Bienvenue ${user?.full_name} ! Je suis votre assistant IA spécialisé dans l'analyse des performances étudiantes. Posez-moi vos questions !`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    }// eslint-disable-next-line
  }, [isAuthenticated, user]);
  

  // Auto-scroll et focus
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (activeTab === 'chat' && inputRef.current && !inputMessage && isAuthenticated) {
      setTimeout(() => inputRef.current.focus(), 100);
    }
  }, [activeTab, inputMessage, isAuthenticated]);

  // === Fonctions d'authentification ===
  const makeAuthenticatedRequest = async (url, options = {}) => {
    const token = localStorage.getItem('auth_token');
    return fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
        ...options.headers
      }
    });
  };
// === Récupération du profil utilisateur ===
  const fetchProfile = async (token) => {
    try {
      const response = await fetch(`${API_BASE}/profile`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
        setIsAuthenticated(true);
      } else {
        localStorage.removeItem('auth_token');
      }
    } catch (error) {
      console.error('Erreur lors de la récupération du profil:', error);
      localStorage.removeItem('auth_token');
    } finally {
      setAuthLoading(false);
    }
  };
// === Fonctions de connexion et d'inscription ===
  const handleLogin = async (e) => {
    e.preventDefault();
    setAuthError('');
    setAuthLoading(true);

    try {
      const response = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          email: authForm.email,
          password: authForm.password
        })
      });
// Vérification de la réponse du serveur
      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('auth_token', data.access_token);
        setUser(data.user);
        setIsAuthenticated(true);
        setAuthForm({ email: '', password: '', full_name: '', role: 'student' });
      } else {
        setAuthError(data.error || 'Erreur de connexion');
      }
    } catch (error) {
      setAuthError('Erreur de connexion au serveur');
    } finally {
      setAuthLoading(false);
    }
  };
// === Fonction d'inscription ===
  const handleRegister = async (e) => {
    e.preventDefault();
    setAuthError('');
    setAuthLoading(true);

    try {
      const response = await fetch(`${API_BASE}/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(authForm)
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('auth_token', data.access_token);
        setUser(data.user);
        setIsAuthenticated(true);
        setAuthForm({ email: '', password: '', full_name: '', role: 'student' });
      } else {
        setAuthError(data.error || "Erreur lors de l'inscription");
      }
    } catch (error) {
      setAuthError('Erreur de connexion au serveur');
    } finally {
      setAuthLoading(false);
    }
  };
// === Fonction de déconnexion ===
  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    setUser(null);
    setIsAuthenticated(false);
    setMessages([]);
    setAuthForm({ email: '', password: '', full_name: '', role: 'student' });
  };

  // === Fonctions de l'application principale ===
  const checkSystemHealth = async () => {
    try {
      const response = await makeAuthenticatedRequest(`${API_BASE}/health`);
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      setSystemStatus({ status: 'error', model_loaded: false });
    }
  };
  // === Récupération des statistiques et des données d'analyse ===
  const fetchStats = async () => {
    try {
      const response = await makeAuthenticatedRequest(`${API_BASE}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Erreur lors de la récupération des stats:', error);
    }
  };
// === Récupération des données d'analyse ===
  const fetchAnalytics = async () => {
    try {
      const response = await makeAuthenticatedRequest(`${API_BASE}/analytics`);
      const data = await response.json();
      setAnalytics(data);
    } catch (error) {
      console.error('Erreur lors de la récupération des analytics:', error);
    }
  };
// === Récupération des données d'analyse des étudiants ===
  const fetchStudentAnalytics = async () => {
    try {
      const response = await makeAuthenticatedRequest(`${API_BASE}/student-analytics`);
      const data = await response.json();
      setStudentAnalytics(data);
    } catch (error) {
      console.error('Erreur lors de la récupération des analytics étudiants:', error);
    }
  };
// === Envoi de message ===
  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !sessionId) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString()
    };
// Ajout du message utilisateur à la conversation
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await makeAuthenticatedRequest(`${API_BASE}/chat`, {
        method: 'POST',
        body: JSON.stringify({ 
          message: inputMessage,
          session_id: sessionId
        })
      });
// Vérification de la réponse du serveur
      const data = await response.json();

      if (response.ok) {
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: data.response,
          messageType: data.type,
          contextUsed: data.context_used,
          timestamp: new Date().toLocaleTimeString()
        };
        setMessages(prev => [...prev, botMessage]);
        fetchAnalytics();
      } else {
        throw new Error(data.message || 'Erreur serveur');
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: `❌ Erreur: ${error.message}. Vérifiez que le serveur Flask est démarré.`,
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
// === Export de la conversation ===
  const exportConversation = () => {
    const conversationText = messages.map(msg => 
      `[${msg.timestamp}] ${msg.type === 'user' ? 'Utilisateur' : 'Assistant'}: ${msg.content}`
    ).join('\n\n');
    
    const blob = new Blob([conversationText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${sessionId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
// === Gestion des événements de saisie ===
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };
// === Gestion de la saisie du message ===
  const handleInputChange = (e) => {
    setInputMessage(e.target.value);
  };
// === Icônes et messages d'état ===
  const getStatusIcon = () => {
    if (systemStatus.status === 'healthy' && systemStatus.model_loaded) {
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    } else if (systemStatus.status === 'error') {
      return <AlertCircle className="w-4 h-4 text-red-500" />;
    } else {
      return <Loader className="w-4 h-4 text-yellow-500 animate-spin" />;
    }
  };
// === Icône de type de message ===
  const getMessageTypeIcon = (messageType) => {
    switch (messageType) {
      case 'analysis':
        return <Brain className="w-4 h-4 text-purple-500" />;
      case 'greeting':
        return <Bot className="w-4 h-4 text-blue-500" />;
      case 'info':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      default:
        return <Bot className="w-4 h-4 text-gray-500" />;
    }
  };
  // visionnage des des differents utilisateur par l'administrateur
 
// === Préparation des données pour les graphiques ===
  const prepareChartData = () => {
    if (!studentAnalytics) return {};

    const departmentData = Object.entries(studentAnalytics.departments || {}).map(([name, value]) => ({
      name,
      value,
      percentage: Math.round((value / (studentAnalytics.total_students || 1)) * 100)
    }));
// === Préparation des données de distribution des notes ===
    const gradeData = Object.entries(studentAnalytics.grades || {}).map(([grade, count]) => ({
      grade,
      count,
      percentage: Math.round((count / (studentAnalytics.total_students || 1)) * 100)
    }));
// === Préparation des données de performance ===
    const performanceData = studentAnalytics.performance_data?.slice(0, 50) || [];

    const dailyActivityData = analytics?.daily_activity?.map(([date, count]) => ({
      date: new Date(date).toLocaleDateString('fr-FR', { month: 'short', day: 'numeric' }),
      conversations: count
    })) || [];

    return {
      departmentData,
      gradeData,
      performanceData,
      dailyActivityData
    };
  };
// === Préparation des données pour les graphiques ===
  const chartData = prepareChartData();

  // === Composant d'authentification ===
  const AuthForm = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center px-4">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-4 rounded-xl inline-block mb-4">
            <BarChart3 className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-gray-900">
            {showLogin ? 'Connexion' : 'Inscription'}
          </h2>
          <p className="mt-2 text-gray-600">
            {showLogin ? "Accédez à votre espace d'analyse" : 'Créez votre compte'}
          </p>
        </div>

        <form className="mt-8 space-y-6" onSubmit={showLogin ? handleLogin : handleRegister}>
          {authError && (
            <div className="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg">
              {authError}
            </div>
          )}

          <div className="space-y-4">
            {!showLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Nom complet
                </label>
                <input
                  type="text"
                  required={!showLogin}
                  value={authForm.full_name}
                  onChange={(e) => setAuthForm({...authForm, full_name: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Votre nom complet"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email
              </label>
              <input
                type="email"
                required
                value={authForm.email}
                onChange={(e) => setAuthForm({...authForm, email: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="votre@email.com"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Mot de passe
              </label>
              <input
                type="password"
                required
                value={authForm.password}
                onChange={(e) => setAuthForm({...authForm, password: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="••••••••"
                minLength="6"
              />
            </div>

            {!showLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Rôle
                </label>
                <select
                  value={authForm.role}
                  onChange={(e) => setAuthForm({...authForm, role: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="student">Étudiant</option>
                  <option value="teacher">Enseignant</option>
                  <option value="admin">Administrateur</option>
                </select>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={authLoading}
            className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 rounded-lg hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 transition-all transform hover:scale-105 flex items-center justify-center"
          >
            {authLoading ? (
              <Loader className="w-5 h-5 animate-spin" />
            ) : (
              showLogin ? 'Se connecter' : "S'inscrire"
            )}
          </button>

          <div className="text-center">
            <button
              type="button"
              onClick={() => {
                setShowLogin(!showLogin);
                setAuthError('');
                setAuthForm({ email: '', password: '', full_name: '', role: 'student' });
              }}
              className="text-blue-600 hover:text-blue-500 text-sm"
            >
              {showLogin ? "Pas de compte ? Inscrivez-vous" : "Déjà un compte ? Connectez-vous"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
  

  // === Composants de l'application principale ===
  const ChatTab = () => (
    <div className="bg-white rounded-2xl shadow-xl overflow-hidden h-[calc(100vh-200px)]">
      <div className="flex-1 overflow-y-auto p-6 space-y-4 h-[calc(100%-140px)]">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3xl flex ${
                message.type === 'user' 
                  ? 'flex-row-reverse bg-gradient-to-r from-blue-500 to-purple-600 text-white' 
                  : 'flex-row bg-gray-50 text-gray-800'
              } rounded-2xl p-4 shadow-sm`}
            >
              <div className={`flex-shrink-0 ${message.type === 'user' ? 'ml-3' : 'mr-3'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  message.type === 'user' ? 'bg-white bg-opacity-20' : 'bg-blue-100'
                }`}>
                  {message.type === 'user' ? (
                    <User className="w-5 h-5" />
                  ) : (
                    getMessageTypeIcon(message.messageType)
                  )}
                </div>
              </div>
              
              <div className="flex-1">
                <div className="whitespace-pre-wrap leading-relaxed">
                  {message.content}
                </div>
                
                <div className={`flex items-center justify-between mt-2 text-xs ${
                  message.type === 'user' ? 'text-white text-opacity-70' : 'text-gray-500'
                }`}>
                  <span>{message.timestamp}</span>
                  {message.contextUsed && (
                    <span className="bg-blue-100 text-blue-600 px-2 py-1 rounded-full">
                      {message.contextUsed} sources utilisées
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-50 rounded-2xl p-4 shadow-sm flex items-center space-x-3">
              <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                <Bot className="w-5 h-5 text-blue-600" />
              </div>
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t bg-gray-50 p-4">
        <div className="flex items-center space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={handleInputChange}
              onKeyDown={handleKeyPress}
              placeholder="Posez votre question sur les étudiants... (Ex: Quels étudiants sont à risque?)"
              className="w-full resize-none rounded-xl border border-gray-200 p-3 pr-12 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 focus:outline-none transition-all"
              rows="2"
              disabled={isLoading || !systemStatus.model_loaded}
              style={{ minHeight: '56px' }}
            />
          </div>
          
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading || !systemStatus.model_loaded}
            className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-3 rounded-xl hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105"
          >
            {isLoading ? (
              <Loader className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>

          <button
            onClick={exportConversation}
            disabled={messages.length <= 1}
            className="bg-gray-500 text-white p-3 rounded-xl hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            title="Exporter la conversation"
          >
            <Download className="w-5 h-5" />
          </button>
        </div>
        
        <div className="mt-3 flex flex-wrap gap-2">
          {['Quels étudiants sont à risque?', 'Analyse du taux de présence', 'Étudiants avec bon niveau'].map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => setInputMessage(suggestion)}
              className="text-sm bg-white border border-gray-200 px-3 py-1 rounded-full hover:bg-blue-50 hover:border-blue-300 transition-colors"
              disabled={isLoading}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  const AnalyticsTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Étudiants</p>
              <p className="text-2xl font-bold text-gray-900">{studentAnalytics?.total_students || 0}</p>
            </div>
            <Users className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">À Risque</p>
              <p className="text-2xl font-bold text-red-500">{studentAnalytics?.at_risk_count || 0}</p>
            </div>
            <AlertCircle className="w-8 h-8 text-red-500" />
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Score Moyen</p>
              <p className="text-2xl font-bold text-green-500">
                {studentAnalytics?.performance_stats?.avg_score?.toFixed(1) || 0}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Conversations</p>
              <p className="text-2xl font-bold text-purple-500">{analytics?.total_conversations || 0}</p>
            </div>
            <Activity className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <PieChart className="w-5 h-5 mr-2 text-blue-500" />
            Distribution par Département
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RechartsPieChart>
              <Pie
                data={chartData.departmentData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({name, percentage}) => `${name} (${percentage}%)`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {chartData.departmentData?.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </RechartsPieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-green-500" />
            Distribution des Notes
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData.gradeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="grade" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#10B981" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-purple-500" />
            Présence vs Performance
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={chartData.performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="attendance" name="Présence (%)" />
              <YAxis dataKey="final_score" name="Score Final" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Étudiants" data={chartData.performanceData} fill="#8B5CF6" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-orange-500" />
            Activité des Conversations
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData.dailyActivityData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="conversations" stroke="#F97316" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  // Rendu principal
  if (authLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-8 h-8 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-gray-600">Chargement...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <AuthForm />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-lg border-b">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-3 rounded-xl">
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-800">Analyse Comportementale Étudiants</h1>
                <p className="text-gray-600">Assistant IA pour l'analyse des performances</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Navigation par onglets */}
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === 'chat' 
                      ? 'bg-white text-blue-600 shadow-sm' 
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                >
                  💬 Chat
                </button>
                <button
                  onClick={() => setActiveTab('analytics')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === 'analytics' 
                      ? 'bg-white text-blue-600 shadow-sm' 
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                >
                  📊 Analytics
                </button>
                {/* Nouveau bouton d'onglet pour les prédictions */}
                <button
                    onClick={() => setActiveTab('predict')}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      activeTab === 'predict' 
                        ? 'bg-white text-blue-600 shadow-sm' 
                        : 'text-gray-600 hover:text-gray-800'
                    }`}
                >
                    🔮 Prédictions
                </button>
              </div>

              {/* Profil utilisateur */}
              <div className="flex items-center space-x-3 bg-gray-50 px-3 py-2 rounded-lg">
                <UserCircle className="w-5 h-5 text-gray-600" />
                <span className="text-sm font-medium text-gray-700">{user?.full_name}</span>
                <span className="text-xs bg-blue-100 text-blue-600 px-2 py-1 rounded-full">
                  {user?.role}
                </span>
              </div>

              {/* Statut du système */}
              <div className="flex items-center space-x-2 bg-gray-50 px-3 py-2 rounded-lg">
                {getStatusIcon()}
                <span className="text-sm font-medium">
                  {systemStatus.status === 'healthy' && systemStatus.model_loaded ? 'En ligne' : 
                   systemStatus.status === 'error' ? 'Hors ligne' : 'Initialisation...'}
                </span>
              </div>
              
              {/* Stats */}
              {stats && (
                <div className="flex items-center space-x-4 text-sm text-gray-600">
                  <div className="flex items-center space-x-1">
                    <Users className="w-4 h-4" />
                    <span>{stats.total_students} étudiants</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Brain className="w-4 h-4" />
                    <span>{stats.total_chunks} segments</span>
                  </div>
                </div>
              )}

              {/* Bouton de déconnexion */}
              <button
                onClick={handleLogout}
                className="bg-red-500 text-white p-2 rounded-lg hover:bg-red-600 transition-colors"
                title="Se déconnecter"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-4">
        {activeTab === 'chat' && <ChatTab />}
        {activeTab === 'analytics' && <AnalyticsTab />}
        {activeTab === 'predict' && <PredictionForm token={localStorage.getItem('auth_token')} />}
      </div>
    </div>
  );
};

export default StudentAnalysisApp;