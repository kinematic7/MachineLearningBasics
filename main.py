# ================================================
# Population Q/A Chatbot
# ================================================

import os
import pandas as pd
import numpy as np
import hashlib
import json
import torch
from sklearn.ensemble import RandomForestRegressor
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# ==========================================================
# Hash utility
# ==========================================================
def get_hash_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ==========================================================
# Population Data Loader
# ==========================================================
class PopulationDataLoader:
    UNWANTED_COLUMNS = ["rank", "cca3", "area (km²)", "density (km²)",
                        "world percentage", "growth rate"]
    REQUIRED = ["country", "continent"]

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

# ==========================================================
# Population Predictor
# ==========================================================
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

            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
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

        if year in info["known_years"]:
            row = self.df[(self.df["country"] == country) & (self.df["year"] == year)].iloc[0]
            return {
                "country": country,
                "continent": info["continent"],
                "year": year,
                "population": row["population"]
            }

        pred = int(info["model"].predict(np.array([[year]]))[0])
        return {
            "country": country,
            "continent": info["continent"],
            "year": year,
            "population": pred
        }

# ==========================================================
# Ask Question Function
# ==========================================================
def ask_question(question, top_k=10):
    # Encode query
    query_emb = embed_model.encode(question).tolist()

    # Retrieve nearest documents
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    if not results["documents"]:
        return "I couldn't find any population information for that question."

    retrieved_answer = results["documents"][0][0]

    FINAL_MARKER = "FINAL_ANSWER:"

    prompt = (
        "You are a population data assistant.\n"
        "Use ONLY the retrieved info below to answer the question accurately.\n\n"
        f"Retrieved info: {retrieved_answer}\n"
        f"Question: {question}\n"
        f"{FINAL_MARKER}"
    )

    output = generator(prompt, max_new_tokens=150, do_sample=False)
    generated = output[0].get("generated_text", "")

    # Clean duplicates
    lines = [line.strip() for line in generated.splitlines() if line.strip()]
    unique = []
    seen = set()
    for line in lines:
        if line not in seen:
            unique.append(line)
            seen.add(line)
    generated = "\n".join(unique)

    # Extract answer after FINAL_MARKER
    idx = generated.find(FINAL_MARKER)
    if idx != -1:
        answer_text = generated[idx + len(FINAL_MARKER):].strip()
    else:
        answer_text = generated.strip()

    # Keep first sentence only
    if "." in answer_text:
        answer_text = answer_text.split(".")[0].strip()

    return answer_text

# ==========================================================
# MAIN PROGRAM
# ==========================================================
if __name__ == "__main__":
    # ----------------------------
    # Load population data
    # ----------------------------
    loader = PopulationDataLoader("world_population_data.csv")
    df = loader.load_dataframe()
    long_df = loader.melt_years(df)

    # ----------------------------
    # Train predictor
    # ----------------------------
    predictor = PopulationPredictor(long_df)
    years_to_predict = [2026]
    predictions = []

    for country in long_df["country"].unique():
        for year in years_to_predict:
            pred = predictor.predict(country, year)
            if pred:
                predictions.append(pred)

    pred_df = pd.DataFrame(predictions)

    # ----------------------------
    # Convert predictions → Q/A pairs
    # ----------------------------
    qa_pairs = []
    for _, row in pred_df.iterrows():
        q = f"What is the population of {row['country']} in {row['year']}?"
        a = f"{row['country']} has a population of approximately {row['population']} in {row['year']}."
        qa_pairs.append({"question": q, "answer": a})

    with open("population_qa.json", "w") as f:
        json.dump(qa_pairs, f, indent=4)

    print(f"Created {len(qa_pairs)} Q/A pairs.")

    # ----------------------------
    # Initialize Chroma DB
    # ----------------------------
    chroma_folder = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(chroma_folder, exist_ok=True)

    client_db = chromadb.Client(
        Settings(persist_directory=chroma_folder, anonymized_telemetry=False)
    )
    collection = client_db.get_or_create_collection("PopulationQA")

    # ----------------------------
    # Embedding model
    # ----------------------------
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ----------------------------
    # Insert embeddings into Chroma
    # ----------------------------
    for pair in qa_pairs:
        doc_id = get_hash_id(pair["question"])
        if not collection.get(ids=[doc_id])["ids"]:
            emb = embed_model.encode(pair["question"]).tolist()
            collection.add(
                ids=[doc_id],
                documents=[pair["answer"]],
                metadatas=[pair],
                embeddings=[emb]
            )

    print("Saved Q/A pairs into Chroma DB at ./chroma_db")

    # ----------------------------
    # Load Hugging Face LLM (Gemma-2B)
    # ----------------------------
    MODEL_NAME = "google/gemma-2b"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(hf_token)
        except Exception as e:
            print(f"Warning: failed to login: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",       # Let accelerate handle GPU placement
            torch_dtype=torch.float16
        )

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            do_sample=False
        )

    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        exit(1)

    #==========================================================
    # Chat Loop
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
