import pickle
import numpy as np

with open('models/severity_map.plk', 'rb') as f:
      severity_map = pickle.load(f)
      
      
def predict_disease(symptoms_list, model, top_k=5):
    # Encode the symptoms based on their severity
    encoded_symptoms = [severity_map[symptom] for symptom in symptoms_list]

    for i in range(len(encoded_symptoms), 17):
        encoded_symptoms.append(0)

    # Create a numpy array from the encoded symptoms
    symptoms_array = np.array(encoded_symptoms).reshape(1, -1)

    # Use the trained model to predict the probabilities of all diseases
    disease_probs = model.predict_proba(symptoms_array)[0]

    # Sort the predicted probabilities in descending order
    sorted_probs_idx = np.argsort(disease_probs)[::-1]

    # Get the top-k predicted diseases and their probabilities
    top_k_diseases = [(model.classes_[idx], disease_probs[idx])
                      for idx in sorted_probs_idx[:top_k]]

    return top_k_diseases


    
    

