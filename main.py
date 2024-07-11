from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from src.preprocessing.data_management import load_model
from src.predict import inference

saved_file_name ="two_input_xor_nn.pkl"
loaded_model = load_model(saved_file_name)

app = FastAPI(title="Two Input XOR Function Implementor",
              description="A Two input neural network to implement XOR Function",
              version="0.1")

origins = ["*"]

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

class two_input_xor_gate(BaseModel):
    X1:float
    X2:float

@app.get("/")
def index():
    return {'message':'A Web App For Serving the output of a two input XOR function implemented through Neural Network'}


@app.post("/Generate Response")
def generate_response(trigger:two_input_xor_gate):
    input1 = trigger.X1
    input2 = trigger.X2

    input_to_nn = np.array([input1,input2])
    nn_out = inference(input_to_nn,loaded_model["params"]["biases"],loaded_model["params"]["weights"])

    return nn_out


if __name__=='__main__':
    uvicorn.run(app)
