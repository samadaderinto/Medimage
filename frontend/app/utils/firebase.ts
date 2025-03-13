// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import {
  browserLocalPersistence,
  getAuth,
  setPersistence,
} from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getFunctions, httpsCallable } from "firebase/functions";
import { getVertexAI, getGenerativeModel } from "firebase/vertexai"; // TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries
// import dotenv from "dotenv";
// Your web app's Firebase configuration

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: "medical-image-processor.firebaseapp.com",
  projectId: "medical-image-processor",
  storageBucket: "medical-image-processor.firebasestorage.app",
  messagingSenderId: "270012789273",
  appId: "1:270012789273:web:2a4b21c4ea67c59dbef088",
};

// Initialize Firebase
const myApp = initializeApp(firebaseConfig);
const myDb = getFirestore(myApp);
const auth = getAuth(myApp);

setPersistence(auth, browserLocalPersistence)
  .then(() => {
    console.log("Auth persistence set to local storage");
  })
  .catch((error) => {
    console.error("Error setting persistence:", error);
  });

const functions = getFunctions(myApp);
const vertexAI = getVertexAI(myApp);

const model = getGenerativeModel(vertexAI, { model: "gemini-2.0-flash" });

export { myApp, myDb, functions, httpsCallable, vertexAI, model, auth };
