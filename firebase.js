// firebase.js

// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyASOMO-k4AbDWqum8Q7HNxlobcNNd_23do",
  authDomain: "chatbot-24f45.firebaseapp.com",
  projectId: "chatbot-24f45",
  storageBucket: "chatbot-24f45.firebasestorage.app",
  messagingSenderId: "35498108209",
  appId: "1:35498108209:web:e238b13d3133c60f043c10",
  measurementId: "G-9V4D3XCY8F"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);


// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);

// Initialize Cloud Firestore and get a reference to the service
export const db = getFirestore(app);

export default app;
// Import the functions you need from the SDKs you need
