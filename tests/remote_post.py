import json
import requests

site = 'https://udacity-census-api.onrender.com/predict'

payload = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Never-married",
    "occupation": "Prof-speciality",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 14000,
    "capital-loss": 0,
    "hours-per-week": 55,
    "native-country": "United-States",
    "salary": ">50K"
}

headers = {'Content-Type':'application/json'}

if __name__ == '__main__':
    response = requests.post(site, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        result = response.json()['result']
        print(f'The prediction for the given data is: {result}')

    else:
        print(f'Bad request (response code: {response.status_code})')