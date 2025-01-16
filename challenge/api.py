from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

app = FastAPI()
class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class FlightsRequest(BaseModel):
    flights: List[Flight]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def predict(request: FlightsRequest):
    valid_opera = ["American Airlines", "Sky Airline", "Grupo LATAM", "Copa Air", "Other"]
    valid_tipovuelo = ["N", "I"]
    valid_mes = list(range(1, 12))

    for flight in request.flights:
        if flight.TIPOVUELO not in valid_tipovuelo:
            raise HTTPException(status_code=400, detail=f"Invalid TIPOVUELO value: {flight.TIPOVUELO}")
        if flight.OPERA not in valid_opera:
            raise HTTPException(status_code=400, detail=f"Invalid OPERA value: {flight.OPERA}")
        if flight.MES not in valid_mes:
            raise HTTPException(status_code=400, detail=f"Invalid MES value: {flight.MES}")


    predictions = [0 for _ in request.flights]
    return {"predict": predictions}

