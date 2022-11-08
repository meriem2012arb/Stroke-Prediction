# Stroke Prediction
# Project Description 

#### About stroke : A stroke, sometimes called a brain attack, occurs when something blocks blood supply to part of the brain or when a blood vessel in the brain bursts.
![alt text](https://github.com/meriem2012arb/Midterm-Project--ML-zoomcamp-/blob/main/stroke.jpg)
    
    
Goal : 
Our objective is to predict whether a patient is likely to get stroke (Target feature : ```stroke``` (0/1)) based on the features from given data using ML techniques.

Data :
The dataset dowloaded is  from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Attribute Information :
```
 id: unique identifier
 gender: "Male", "Female" or "Other"
 age: age of the patient 
 hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
 heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
 ever_married: "No" or "Yes"
 work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
 Residence_type: "Rural" or "Urban"
 avg_glucose_level: average glucose level in blood
 bmi: body mass index
 smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
 stroke: 1 if the patient had a stroke or 0 if not
 *Note: "Unknown" in smoking_status means that the information is unavailable for this patient
 ```
### Table of Contents :
-----
### Run the Code


Pipenv creates an enviroment with the name of the current folder.

Install 'pipenv' running in shell:

```pip install pipenv```

Activate the environment running in shell:

```pipenv shell```

When then environment is activated, install everything using 'pipenv' instead of 'pip', for this project, to creat the Pipfile and the Piplock.file, we run (since you have them already in the folder, you do not need to run the following command line):

``` pipenv install numpy scikit-learn==1.1.2 xgboost flask gunicorn``` 

The Pipfile records what you have installed (thus only run the packages installation once) and in the Pipfile.lock are the packages checksums.

Close the environment with Crt + d

To use the environment, run pipenv shell and deploy the model as said in the next section.

### Apply the deployment

In the active environment and open the web server by running:

```gunicorn --bind 0.0.0.0:9696 predict_deposit:app```

(use 'waitress' instead of 'gunicorn' if you are in Windows).

The data of a new patient are written in 'predict_test.py'. Test the deployment by running it in other shell:

```python3 predict_test_deposit.py```

The output (if that client will open a deposit or not and the probability) will be written in the shell.

Close the web server with Ctrl + c.

### Docker

We do not need to install packages, activate environments, train models,... everytime we want to know if a new patient will get stroke or not. We can skip the former sections using a Docker container.

First, create a Docker image locally by running in shell (the enviroment does not need to be activated):

```docker run -it --rm --entrypoint=bash python:3.8.12-slim```

Exit the container shell with Ctrl + d.

The Dockerfile is this folder installs python, runs pipenv to install packages and dependencies, runs the predict.py script to open the web server and the  model and deploys it using gunicorn

```docker built -t docker-deposit```
(the last point means 'here', i.e., run it in the environment folder).

Run the docker container with:

```docker run -it --rm -p 9696:9696 docker-deposit```

and the model will be deployed and ready to use.

To send a new request, open a new shell in the enviroment directory and directly run:

python predict_test.py

and you will see if the patient will get a stroke or not and its probability.

Close the container with Ctrl + c.








