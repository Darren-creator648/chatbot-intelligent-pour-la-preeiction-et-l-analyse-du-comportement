// AuthContainer.jsx
import React, { useState } from 'react';
import { Loader, BarChart3 } from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';

const AuthContainer = ({ onAuthSuccess, authLoading, setAuthLoading }) => {
  const [showLogin, setShowLogin] = useState(true);
  const [authForm, setAuthForm] = useState({
    email: '',
    password: '',
    full_name: '',
    role: 'student'
  });
  const [authError, setAuthError] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    setAuthError('');
    setAuthLoading(true);

    try {
      const response = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: authForm.email, password: authForm.password })
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('auth_token', data.access_token);
        onAuthSuccess(data.user);
      } else {
        setAuthError(data.error || 'Erreur de connexion');
      }
    } catch (error) {
      setAuthError('Erreur de connexion au serveur');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setAuthError('');
    setAuthLoading(true);

    try {
      const response = await fetch(`${API_BASE}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(authForm)
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('auth_token', data.access_token);
        onAuthSuccess(data.user);
      } else {
        setAuthError(data.error || "Erreur lors de l'inscription");
      }
    } catch (error) {
      setAuthError('Erreur de connexion au serveur');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleAuthFormChange = (e) => {
    const { name, value } = e.target;
    setAuthForm(prev => ({ ...prev, [name]: value }));
  };

  return (
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
                <label className="block text-sm font-medium text-gray-700 mb-1">Nom complet</label>
                <input
                  type="text"
                  name="full_name"
                  required
                  value={authForm.full_name}
                  onChange={handleAuthFormChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Votre nom complet"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
              <input
                type="email"
                name="email"
                required
                value={authForm.email}
                onChange={handleAuthFormChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="votre@email.com"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Mot de passe</label>
              <input
                type="password"
                name="password"
                required
                value={authForm.password}
                onChange={handleAuthFormChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="••••••••"
                minLength="6"
              />
            </div>

            {!showLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Rôle</label>
                <select
                  name="role"
                  value={authForm.role}
                  onChange={handleAuthFormChange}
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
};

export default AuthContainer;