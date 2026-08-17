import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200


def test_prediction():

    response = client.post(
        "/predict",
        json={
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.5,
            "TotalCharges": 1074.0
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert (
        "churn_probability"
        in result
    )