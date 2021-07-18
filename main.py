import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from mlutils import load_model, predict, retrain
from typing import List
from datetime import datetime, time

# defining the main app
app = FastAPI(title="GermanCreditScoring", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", load_model)

# class which is expected in the payload
class QueryIn(BaseModel):
    duration_in_month: str
    credit_history: str
    purpose: str
    credit_amount: str
    present_employment_since_in_yrs: str
    installment_rate_in_percent_of_income: str
    job: str
    housing: str
    no_of_people_liable_to_provide_maintainence_for: str
    foreign_worker: str
# class which is returned in the response
class QueryOut(BaseModel):
    loan_status: str
    time_stamp: time

# class which is expected in the payload while re-training
class FeedbackIn(BaseModel):
    duration_in_month: str
    credit_history: str
    purpose: str
    credit_amount: str
    present_employment_since_in_yrs: str
    installment_rate_in_percent_of_income: str
    job: str
    housing: str
    no_of_people_liable_to_provide_maintainence_for: str
    foreign_worker: str
    loan_status: str
# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/predict", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def predict_loan_status(query_data: QueryIn):
    return {"loan_status": predict(query_data),
            "time_stamp": datetime.now().time()}

@app.post("/feedback_loop", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct flower class
# Response: Dict with detail confirming success (200)
def feedback_loop(data: List[FeedbackIn]):
    retrain(data)
    return {"detail": "Feedback loop successful",
            "time_stamp": datetime.now().time()}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
