import requests

url = "http://localhost:9696/predict"

patient = {"gender": "Female",
           "age": 30.0,
           "hypertension": 0,
           "heart_disease": 0, 
           "ever_married": "No",
           "work_type": "Self-employed",
           "Residence_type": "Urban",
           "avg_glucose_level": 156.57,
           "bmi": 27.0,
           "smoking_status": "never smoked"} 

response = requests.post(url, json = patient).json()
print(response)


