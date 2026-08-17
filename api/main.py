from fastapi import FastAPI, HTTPException

from api.schemas import CustomerInput

from api.model_loader import (
    get_model,
    get_threshold,
)


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "ML API for predicting "
        "customer churn risk"
    ),
    version="1.0.0"
)


model = get_model()

threshold = get_threshold()



@app.get("/")
def home():

    return {
        "message":
        "Customer Churn API running"
    }



@app.post("/predict")
def predict(
    customer: CustomerInput
):

    try:

        input_data = (
            customer
            .model_dump()
        )


        import pandas as pd


        df = pd.DataFrame(
            [input_data]
        )


        probability = (
            model
            .predict_proba(df)
            [:,1][0]
        )


        prediction = int(
            probability
            >= threshold
        )


        return {

            "churn_probability":
                round(
                    float(probability),
                    4
                ),

            "decision_threshold":
                round(
                    float(threshold),
                    4
                ),

            "prediction":
                prediction,

            "result":
                (
                    "Churn Risk"
                    if prediction == 1
                    else
                    "No Churn"
                )
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )