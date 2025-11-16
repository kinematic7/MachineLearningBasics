import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import chromadb
from chromadb.config import Settings
from ollama import embeddings, chat
import hashlib
import json

#==============================================================
# Utility: SHA256 hash (for stable unique document IDs)
#==============================================================
def get_hash_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

#==============================================================
# Population Data Loader
#==============================================================
class PopulationDataLoader:
    UNWANTED_COLUMNS = ["rank", "cca3", "area (km²)", "density (km²)", 
                        "world percentage", "growth rate"]
    REQUIRED = ["country", "continent", "2023 population"]

    def __init__(self, file_path):
        self.file_path = file_path

    def load_dataframe(self):
        df = pd.read_csv(self.file_path)
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df.drop(columns=self.UNWANTED_COLUMNS, errors="ignore")
        return df

    def melt_years(self, df):
        pop_cols = [c for c in df.columns if "population" in c]
        long_df = df.melt(
            id_vars=["country", "continent"],
            value_vars=pop_cols,
            var_name="year",
            value_name="population"
        )
        long_df["year"] = long_df["year"].str.extract(r"(\d{4})").astype(int)
        long_df = long_df.sort_values(["country", "year"]).reset_index(drop=True)
        return long_df

#==============================================================
# Population Predictor
#==============================================================
class PopulationPredictor:
    def __init__(self, df, n_estimators=300):
        self.df = df
        self.n_estimators = n_estimators
        self.models = {}
        self._train_models()

    def _train_models(self):
        for country in self.df["country"].unique():
            country_df = self.df[self.df["country"] == country]
            X = country_df["year"].values.reshape(-1, 1)
            y = country_df["population"].values

            model = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=42
            )
            model.fit(X, y)

            continent = country_df["continent"].iloc[0]
            self.models[country] = {
                "model": model,
                "continent": continent,
                "known_years": set(country_df["year"].values)
            }

    def predict(self, country, year):
        if country not in self.models:
            return None

        info = self.models[country]
        model, continent, known_years = info["model"], info["continent"], info["known_years"]

        # If we already have actual data for this year
        if year in known_years:
            row = self.df[(self.df['country'] == country) & (self.df['year'] == year)].iloc[0]
            return {
                "country": country,
                "continent": continent,
                "year": year,
                "population": row["population"]
            }

        # Predict new year
        pred = int(model.predict(np.array([[year]]))[0])
        return {
            "country": country,
            "continent": continent,
            "year": year,
            "population": pred
        }

#==============================================================
# MAIN PROGRAM
#==============================================================
if __name__ == "__main__":

    # Load population data
    loader = PopulationDataLoader("world_population_data.csv")
    df = loader.load_dataframe()
    long_df = loader.melt_years(df)

    # Train predictor
    predictor = PopulationPredictor(long_df)

    # Predict 2026 only (expand if needed)
    years_to_predict = [2026]
    countries = long_df["country"].unique()
    predictions = []

    for country in countries:
        for year in years_to_predict:
            pred = predictor.predict(country, year)
            if pred:
                predictions.append(pred)

    pred_df = pd.DataFrame(predictions)

    #==========================================================
    # Convert predictions → Q/A pairs
    #==========================================================
    qa_pairs = []
    for _, row in pred_df.iterrows():
        q = f"What is the population of {row['country']} in {row['year']}?"
        a = f"The population of {row['country']} in {row['year']} is {row['population']}."
        qa_pairs.append({"question": q, "answer": a})

    # Save as JSON if needed
    with open("population_qa.json", "w") as f:
        json.dump(qa_pairs, f, indent=4)

    print(f"Created {len(qa_pairs)} Q/A pairs.")

    #==========================================================
    # Initialize Chroma DB
    #==========================================================
    chroma_folder = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(chroma_folder, exist_ok=True)

    client_db = chromadb.Client(
        Settings(persist_directory=chroma_folder, anonymized_telemetry=False)
    )

    collection = client_db.get_or_create_collection("PopulationQA")

    #==========================================================
    # Insert Q/A pairs into Chroma
    #==========================================================
    for pair in qa_pairs:
        doc_id = get_hash_id(pair["question"])

        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue

        emb = embeddings("llama3:latest", pair["question"])["embedding"]

        collection.add(
            ids=[doc_id],
            documents=[pair["answer"]],
            metadatas=[pair],
            embeddings=[emb]
        )

    print("Saved Q/A pairs into Chroma DB at ./chroma_db")

    #==========================================================
    # Ask Question Function
    #==========================================================
    def ask_question(question, top_k=50):
        query_embedding = embeddings("llama3:latest", question)["embedding"]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        print (f"Retrieved {len(results['documents'][0])} documents from Chroma DB.")

        if not results["documents"]:
            return "I couldn't find any population information for that question."

        retrieved_answer = results["documents"][0][0]

        response = chat("llama3:latest", [
            {
                "role": "system",
                "content": (
                    "You are a population data assistant. "
                    "You must answer ONLY using the retrieved information. "
                    "Do not invent numbers or facts."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Retrieved info: {retrieved_answer}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer clearly and concisely."
                )
            }
        ])

        return response["message"]["content"]

    #==========================================================
    # Chatbot Loop
    #==========================================================
    print("\nPopulation Q/A Chatbot. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            answer = ask_question(question)
            print(f"Bot: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")
