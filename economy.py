import pandas as pd
import numpy as np
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. State Definition
class EconomicState(TypedDict):
    fileName: str
    data: Optional[pd.DataFrame]
    model: Optional[object]
    accuracy: float
    admin_rankings: Optional[pd.DataFrame]


# 2. Node: Fetch & Preprocess Data
def loadAndPrepData(state: EconomicState) -> dict:
    df = pd.read_csv(state["fileName"])

    # Fill missing value in 2000 GDP Change with 0.0
    df["GDP Change ($T)"] = df["GDP Change ($T)"].fillna(0.0)

    # Define "Good Economy" target (Composite metric threshold):
    # High Real Growth (>2.5%) AND Low Unemployment (<5.5%) AND Low Inflation (<3.5%)
    good_growth = df["Real GDP Growth (%)"] >= 2.5
    low_unemp = df["Unemployment (%)"] <= 5.5
    low_inflation = df["Inflation (%)"] <= 3.5

    # Target: 1 = Good economic year, 0 = Challenging economic year
    df["target"] = (good_growth & low_unemp & low_inflation).astype(int)

    return {"data": df}


# 3. Node: Train Ensemble & Evaluate Administrations
def trainAndPredict(state: EconomicState) -> dict:
    df = state["data"]

    # Select numerical features for modeling
    features = [
        "Debt/GDP (%)",
        "GDP ($T)",
        "GDP Change ($T)",
        "Real GDP Growth (%)",
        "Inflation (%)",
        "Unemployment (%)",
        "Deficit ($T)",
        "Interest ($T)",
    ]

    X = df[features]
    y = df["target"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize Ensemble Models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=1000)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)], voting="soft"
    )

    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    # Predict probabilities for the ENTIRE dataset to rate administrations
    # Column [1] represents probability of a "Good Economic Year"
    df["good_econ_probability"] = ensemble.predict_proba(X)[:, 1]

    # Group by Administration to rank overall economic performance
    admin_summary = (
        df.groupby("Administration")
        .agg(
            Avg_Probability_Good_Econ=("good_econ_probability", "mean"),
            Avg_Real_GDP_Growth=("Real GDP Growth (%)", "mean"),
            Avg_Unemployment=("Unemployment (%)", "mean"),
            Avg_Inflation=("Inflation (%)", "mean"),
            Total_Years=("Year", "count"),
        )
        .sort_values(by="Avg_Probability_Good_Econ", ascending=False)
        .reset_index()
    )

    return {
        "model": ensemble,
        "accuracy": accuracy,
        "admin_rankings": admin_summary,
    }


# 4. Graph Construction
if __name__ == "__main__":
    graph = StateGraph(EconomicState)

    graph.add_node("loadAndPrepData", loadAndPrepData)
    graph.add_node("trainAndPredict", trainAndPredict)

    graph.add_edge(START, "loadAndPrepData")
    graph.add_edge("loadAndPrepData", "trainAndPredict")
    graph.add_edge("trainAndPredict", END)

    app = graph.compile()

    # Save the dataset to data.csv before executing
    result = app.invoke(
        {
            "fileName": "data.csv",
            "data": None,
            "model": None,
            "accuracy": 0.0,
            "admin_rankings": None,
        }
    )

    print(f"=== Model Test Accuracy: {result['accuracy'] * 100:.1f}% ===")
    print("\n=== Administration Economic Ranking (ML Probability) ===")
    print(result["admin_rankings"].to_string(index=False))