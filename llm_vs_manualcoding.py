import chromadb
import pandas as pd
import re
import ollama

# Initialize Chroma client
chroma_client = chromadb.Client()

# Ollama model
#OLLAMA_MODEL = "llama3:latest"  # your installed model
OLLAMA_MODEL = "llama3.1:8b"

class ChromaDBHandler:
    def __init__(self, client):
        self.client = client

    def create_collection(self, name):
        return self.client.create_collection(name=name)

    def get_collection(self, name):
        return self.client.get_collection(name)

    def delete_collection(self, name):
        try:
            self.client.delete_collection(name=name)
        except Exception:
            pass

    def load_data(self, collection, csv_path):
        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            doc = (
                f"Country: {row['country']} is in {row['continent']}. "
                f"It has a 2023 population of {row['2023 population']} and a growth rate of {row['growth rate']}. "
                f"Area is {row['area (km²)']} km² with density {row['density (km²)']}."
            )
            collection.add(
                documents=[doc],
                metadatas=[{
                    "country": row["country"],
                    "continent": row["continent"],
                    "rank": row["rank"],
                    "population_2023": row["2023 population"],
                    "area_km2": row["area (km²)"],
                    "growth_rate": row["growth rate"]
                }],
                ids=[str(index)]
            )

    def create_population_collection(self, collection_name="population_collection", csv_path="world_population_data.csv"):
        try:
            collection = self.get_collection(collection_name)
        except Exception:
            collection = self.create_collection(collection_name)
            self.load_data(collection, csv_path)
        return collection

    def get_result_details(self, results):
        docs = results.get("documents", [[]])[0]
        return docs

    def use_ollama_for_population(self, data):
        """Calculate total population using Ollama directly and return one line."""

        prompt = f"""
                    You are given data about countries. Each line has the country name and its population, e.g.:
                    Country: Bhutan has a population of 787424
                    Country: Nepal has a population of 30000000

                    Data:                    
                    {data}

                    Question: Calculate the total population of all these countries and provide **only** the number.
                    Respond in this exact format:
                    Total population: ********
                    Do not add any other text or explanation.
            """

        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        content = response.message.content

        # Extract the line with "Total population"
        match = re.search(r"Total population: \**[\d,]+\**", content)
        if match:
            return match.group(0)
        return "Total population not found."

    def get_population_of_countries_no_LLM(self, refiltercollection):
        """
        Calculate total population from a list of document strings or metadata dicts (refiltercollection)
        without using an LLM. Returns the integer total population.
        """
        populations = []
        for item in refiltercollection:
            if not item:
                continue

            # Normalize item to text: it may be a string (document) or a dict (metadata)
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                # join metadata values into a single string
                text = " ".join(str(v) for v in item.values())
            else:
                text = str(item)

            # Try to extract the population number following the word "population"
            match = re.search(r"population(?:\s*of)?\s*[:=]?\s*([\d,]+)", text, flags=re.IGNORECASE)
            if not match:
                # Fallback: extract the first large number (likely a population)
                match = re.search(r"([\d,]{4,})", text)

            if match:
                num_str = match.group(1).replace(",", "")
                try:
                    populations.append(int(num_str))
                except ValueError:
                    # skip values that cannot be parsed as int
                    continue

        total_population = sum(populations)
        return total_population


def main():
    db_handler = ChromaDBHandler(chroma_client)
    collection = db_handler.create_population_collection()

    # Semantic query example
    result = collection.query(
        query_texts=["What is the population of Bangladesh?"],
        n_results=1,
        include=["documents", "metadatas"]
    )
    #print("Semantic Question Query Result:", db_handler.get_result_details(result))
    details = db_handler.get_result_details(result)
    print("Filtered Query Semantic Search:", str(details[0]).replace("[", "").replace("]", "").replace("'","").strip() if details else "No results found")
    #print("Semantic Question Query Result:", str(details[0]).replace("[", "").replace("]", "").replace("'","").strip() if details else "No results found")
    print("----------------")

    # Filtered query example
    result = collection.query(
        query_texts=[""],
        where={"population_2023": 787424},
        n_results=1,
        include=["documents", "metadatas"]
    )
    details = db_handler.get_result_details(result)
    print("Filtered Query Filter Result:", str(details[0]).replace("[", "").replace("]", "").replace("'","").strip() if details else "No results found")
    print("----------------")

    print("Filtered with Multiple Conditions:")

    result = collection.query(
        query_texts=[""],  # semantic query can be empty
        where={
            "$and": [      # specify and / or
                {"population_2023": {"$gt": 1000000}},
                {"population_2023": {"$lt": 2000000}},
                {"continent": "Asia"}
            ]
        },
        n_results=100,
        include=["documents", "metadatas"]
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    subsetofAsia = ""

    for doc, meta in zip(docs, metas):
        subsetofAsia += doc + "\n"
        print(f"{meta['country']}: {meta['population_2023']}")    

    print("----------------")

    total_population_line = db_handler.use_ollama_for_population(subsetofAsia)
    print("Total population on Filtered Condition (subset of Asia, calculated by Ollama):", total_population_line)

    print("----------------")

    # List all countries in Asia
    result = collection.query(
        query_texts=["List all the countries in Asia"],
        n_results=150,
        include=["documents", "metadatas"]
    )
    refiltercollection = db_handler.get_result_details(result)

    print("Countries in Asia:")

    print("----------------")

    #print("Length of populations from items:", countCountries)

    asianCountries = [countryData for countryData in refiltercollection if "Asia" in countryData]
    for countryData in asianCountries:
        print(countryData)

    #print("Length of populations from items:", len(asianCountries))

    print("----------------")

    # Use Ollama to calculate total population
    total_population_line = db_handler.use_ollama_for_population(asianCountries)
    print("Total population in Asia (calculated by Ollama):", total_population_line)

    # Calculate total population without LLM
    total_population_no_llm = db_handler.get_population_of_countries_no_LLM(asianCountries)
    print("Total population in Asia (calculated without LLM):", total_population_no_llm)

if __name__ == "__main__":
    main()
