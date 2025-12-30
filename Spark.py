# Commands you need to run
# -- This sets a virtual environment and installs pyspark --
# -- If you are using a windows machine, use WSL (Windows Subsystem for Linux) --
# -- Ensure you install python, pip and JDK installed in your WSL environment --
# nezra@muhammad:/mnt/c/Users/nezra/sparktest$ python3 -m venv ~/spark-venv
# nezra@muhammad:/mnt/c/Users/nezra/sparktest$ source ~/spark-venv/bin/activate
# (spark-venv) nezra@muhammad:/mnt/c/Users/nezra/sparktest$ pip install --upgrade pip
# (spark-venv) nezra@muhammad:/mnt/c/Users/nezra/sparktest$ pip install pyspark

from pyspark.sql import SparkSession
from typing import TypedDict, List
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from langgraph.graph import StateGraph, END


# =========================================================
# OpenAI Client (uses OPENAI_API_KEY env var)
# =========================================================
openai_client = OpenAI(
      api_key="sk-proj-yourkeyhere"  # Replace with your OpenAI API key
)

# =========================================================
# Spark: Load CSV
# =========================================================
def load_population_data(spark, path: str):
    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )


# =========================================================
# Spark → Plain Records
# =========================================================
def spark_df_to_records(df) -> List[dict]:
    records = []

    for r in df.collect():
        text = f"""
            Country: {r['country']}
            Continent: {r['continent']}
            CCA3: {r['cca3']}
            2023 Population: {r['2023 population']}
            2022 Population: {r['2022 population']}
            2020 Population: {r['2020 population']}
            Growth Rate: {r['growth rate']}
            World Percentage: {r['world percentage']}
            Area (km²): {r['area (km²)']}
            Density (km²): {r['density (km²)']}
            """.strip()

        records.append({
            "id": r["cca3"],
            "text": text,
            "metadata": {
                "country": r["country"],
                "continent": r["continent"]
            }
        })

    return records


# =========================================================
# OpenAI Embeddings
# =========================================================
def embed_texts(texts: List[str]) -> List[List[float]]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [e.embedding for e in response.data]


# =========================================================
# Chroma Vector Store (Direct API)
# =========================================================
def create_chroma_collection(records: List[dict]):
    chroma_client = chromadb.Client(
        Settings(
            persist_directory="./chroma_population",
            anonymized_telemetry=False
        )
    )

    collection = chroma_client.get_or_create_collection(
        name="world_population"
    )

    embeddings = embed_texts([r["text"] for r in records])

    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[r["metadata"] for r in records],
        embeddings=embeddings
    )

    return collection


# =========================================================
# LangGraph State
# =========================================================
class GraphState(TypedDict):
    question: str
    context: str
    answer: str


# =========================================================
# LangGraph Nodes
# =========================================================
def retrieve_node(state: GraphState, collection):
    query_embedding = embed_texts([state["question"]])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    context = "\n\n".join(results["documents"][0])
    return {"context": context}


def analyze_node(state: GraphState):
    prompt = f"""
You are a population data analyst.

Context:
{state['context']}

Question:
{state['question']}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": response.choices[0].message.content}


def filter_with_spark_sql(state: GraphState, df):
    df.createOrReplaceTempView("population")
    query = """
        SELECT country, continent, `growth rate`, `2023 population`
        FROM population
        WHERE continent = 'Asia'
          AND `growth rate` LIKE '-%'
    """
    filtered_df = df.sparkSession.sql(query)
    return filtered_df

def filter_with_spark_without_sql(state: GraphState, df):
    
    # Initial filter
    filtered_df = (
        df
        .filter(df["continent"] == "Asia")       # continent = Asia
        .filter(df["growth rate"].contains("-")) # negative growth rate
    )

    # Additional filter: remove China
    filtered_df = filtered_df.filter(filtered_df["country"] != "China")

    # Select only the columns we want
    filtered_df = filtered_df.select(
        "country",
        "continent",
        "growth rate",
        "2023 population"
    )

    return filtered_df

# =========================================================
# Main
# =========================================================
def main():
    # 1️⃣ Initialize Spark
    spark = SparkSession.builder.appName("Spark-Chroma-LangGraph").getOrCreate()

    # 2️⃣ Load CSV
    CSV_PATH = "Population.csv"
    df = load_population_data(spark, CSV_PATH)

    # 3️⃣ Convert to records for Chroma
    records = spark_df_to_records(df)
    collection = create_chroma_collection(records)

    # 4️⃣ Build LangGraph AFTER df exists
    graph = StateGraph(GraphState)

    # Add Spark filter node
    graph.add_node(
        "filter_with_spark",
        lambda s: filter_with_spark_sql(s, df)  # ✅ df now exists
    )

    graph.add_node(
        "filter_with_spark_without_sql",
        lambda s: filter_with_spark_without_sql(s, df)  # ✅ df now exists
    )

    # Add retrieve and analyze nodes
    graph.add_node("retrieve", lambda s: retrieve_node(s, collection))
    graph.add_node("analyze", analyze_node)

    # Define flow
    # You can interchange between line 221, 222 and 224, 225
    # One is an example of using Spark SQL, the other is using Spark DataFrame API
    
    graph.set_entry_point("filter_with_spark_without_sql")
    graph.add_edge("filter_with_spark_without_sql", "retrieve")    
        
    #graph.set_entry_point("filter_with_spark_sql")  
    #graph.add_edge("filter_with_spark_sql", "retrieve")    
    
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", END)

    compiled_graph = graph.compile()

    # 5️⃣ Invoke graph
    result = compiled_graph.invoke({
        "question": "Which Asian countries show negative population growth?"
    })

    print(result["answer"])
    spark.stop()

if __name__ == "__main__":
    main()



## IF you are new to Lang Graph put this in a separate file and run it

# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph as LangGraph, END, START
# import threading
# import random
# import time

# class State(TypedDict):
#     current_action: str
#     action_history: list[str]

# #
# # --- Function used for Conditional Edge from response "yes" or "no" ---
# #

# def does_vector_store_exist(state: State) -> State:
#     state["current_action"] = input("Does Vector Store Exist? (yes/no) : ")
#     state["action_history"].append(state["current_action"])
#     return state

# #
# # --- Core Functions  ---
# #

# def fetch_data_from_sql(state: State) -> State:
#     state["current_action"] = "Fetch Data from SQL Table"
#     state["action_history"].append(state["current_action"])
#     return state

# def save_data_to_vector_store(state: State) -> State:
#     state["current_action"] = "Save Data to Vector Store"
#     state["action_history"].append(state["current_action"])
#     return state

# def get_user_query(state: State) -> State:
#     state["current_action"] = "Get User Query"
#     state["action_history"].append(state["current_action"])
#     return state

# def fetch_data_from_vector_store(state: State) -> State:
#     state["current_action"] = "Fetch data from Vector Store"
#     state["action_history"].append(state["current_action"])
#     return state

# def fetch_data_from_LLM(state: State) -> State:
#     state["current_action"] = "Fetch Data from LLM"
#     state["action_history"].append(state["current_action"])

#     # Put all functions in a list
#     functions = [process_one_in_parallel, process_two_in_parallel]

#     # Create threads dynamically, pass state as a parameter
#     threads = [threading.Thread(target=f, args=(state,)) for f in functions]

#     # Start all threads
#     for t in threads: t.start()

#     # Wait for all threads to finish
#     for t in threads: t.join()
    
#     return state

# def process_one_in_parallel(state: State) -> State:  
#     delay = random.uniform(1, 3)  # Random delay between 1 and 5 seconds
#     time.sleep(delay)
#     state["action_history"].append(f"Process one done after {delay:.2f}s")  
#     return state

# def process_two_in_parallel(state: State) -> State:
#     delay = random.uniform(1, 3)  # Random delay between 1 and 5 seconds
#     time.sleep(delay)
#     state["action_history"].append(f"Process two done after {delay:.2f}s")
#     return state


# def print_action_history(state: State):
#     step = 1
#     for action in state["action_history"]:      
#         print(f"{step}. {action}")
#         step +=1

# #
# # --- Declare Graph and Nodes ---
# #

# graph = LangGraph(State)

# graph.add_node("does_vector_store_exist", does_vector_store_exist)
# graph.add_node("fetch_data_from_sql", fetch_data_from_sql)
# graph.add_node("save_data_to_vector_store", save_data_to_vector_store)
# graph.add_node("get_user_query", get_user_query)
# graph.add_node("fetch_data_from_vector_store", fetch_data_from_vector_store)
# graph.add_node("fetch_data_from_LLM", fetch_data_from_LLM)

# #
# # --- Connect Edges ---
# #

# graph.add_edge(START, "does_vector_store_exist")

# graph.add_conditional_edges(
#     "does_vector_store_exist",
#     lambda state: "get_user_query" if state["current_action"] == "yes" else "fetch_data_from_sql",
#     {
#         "get_user_query": "get_user_query",
#         "fetch_data_from_sql": "fetch_data_from_sql"
#     }
# )

# graph.add_edge("fetch_data_from_sql", "save_data_to_vector_store")
# graph.add_edge("save_data_to_vector_store", "get_user_query")
# graph.add_edge("get_user_query", "fetch_data_from_vector_store")
# graph.add_edge("fetch_data_from_vector_store", "fetch_data_from_LLM")
# graph.add_edge("fetch_data_from_LLM", END)

# app = graph.compile()

# state: State = {
#     "current_action": "",
#     "action_history": []
# }

# state = app.invoke(state)
# print_action_history(state)
