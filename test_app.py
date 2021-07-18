from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_pred_True():
    # defining a sample payload for the testcase
    payload = {
        'duration_in_month': 6,
        'credit_history': 'A34',
        'purpose': 'A43',
        'credit_amount': 1169,
        'present_employment_since_in_yrs': 7,
        'installment_rate_in_percent_of_income': 4,
        'job': 'A173',
        'housing': 'A152',
        'no_of_people_liable_to_provide_maintainence_for': 1,
        'foreign_worker': 'A201',
        'loan_status': 1
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica"}

# def test_pred_setosa():
#     payload = {
#         "sepal_length": 4.8,
#         "sepal_width": 3,
#         "petal_length": 1.4,
#         "petal_width": 0.1,
#     }
#     with TestClient(app) as client:
#         response = client.post("/predict_flower", json=payload)
#         # asserting the correct response is received
#         assert response.status_code == 200
#         assert response.json() == {"flower_class": "Iris Setosa"}


# def test_pred_versicolor():
#     payload = {
#         "sepal_length": 6,
#         "sepal_width": 2.9,
#         "petal_length": 4.5,
#         "petal_width": 1.5,
#     }
#     with TestClient(app) as client:
#         response = client.post("/predict_flower", json=payload)
#         # asserting the correct response is received
#         assert response.status_code == 200
#         assert response.json() == {"flower_class": "Iris Versicolour"}