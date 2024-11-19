from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

rf_model = joblib.load("best_random_forest_regressor_model.pkl")
label_encoders_guardados = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

class PredictionInput(BaseModel):
    Game_Title: str
    Year: float
    Publisher: str
    North_America: float
    Europe: float
    Japan: float
    Rest_of_World: float
    Global: float
    Number_of_Reviews: float
    Summary: str
    Wishlist: float
    Platform: str
    Genre: str

def convert_to_float(value):
    if isinstance(value, str):
        if "K" in value:
            return float(value.replace("K", "")) * 1_000
        elif "M" in value:
            return float(value.replace("M", "")) * 1_000_000
    return float(value)

def align_column_names(df):
    column_mapping = {
        "Game_Title": "Game Title",
        "Year": "Year",
        "Publisher": "Publisher",
        "North_America": "North America",
        "Europe": "Europe",
        "Japan": "Japan",
        "Rest_of_World": "Rest of World",
        "Global": "Global",
        "Number_of_Reviews": "Number of Reviews",
        "Summary": "Summary",
        "Wishlist": "Wishlist",
        "Platform": "Platform",
        "Genre": "Genre",
    }
    df.rename(columns=column_mapping, inplace=True)
    return df

def preprocess_input(data):
    df = pd.DataFrame([data.dict()])

    df = align_column_names(df)

    required_columns = [
        "Genre", "Platform", "Publisher", "Wishlist", "Number of Reviews",
        "Year", "North America", "Europe", "Japan", "Rest of World", "Global"
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0 

    df["Wishlist"] = df["Wishlist"].apply(convert_to_float)
    df["Number of Reviews"] = df["Number of Reviews"].apply(convert_to_float)
    df["Publisher"] = df["Publisher"].fillna("Unknown")
    df["Year"] = df["Year"].fillna(0)

    for col in ["Genre", "Publisher", "Platform"]:
        df[col] = df[col].astype(str)
        if col in label_encoders_guardados:
            df[col] = df[col].map(
                lambda s: label_encoders_guardados[col].transform([s])[0]
                if s in label_encoders_guardados[col].classes_
                else 0
            )

    expected_columns = scaler.feature_names_in_
    df = df.reindex(columns=expected_columns)
    print("Columnas finales para el escalador:", df.columns)
    print("Primeras filas escaladas:", df.head())
    df = scaler.transform(df)

    return df

@app.post("/predict_rating")
def predict(input_data: PredictionInput):
    processed_data = preprocess_input(input_data)
    prediction = rf_model.predict(processed_data)

    return {"predicted_rating": prediction[0]}