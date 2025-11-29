import pyodbc
import random
import chromadb
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

#--------------------------------
# CREATE YOUR SQL TABLE IN MSSQL
#--------------------------------

# USE [CrimsonSky]
# GO

# /****** Object:  Table [dbo].[Contact]    Script Date: 11/29/2025 5:51:59 PM ******/
# SET ANSI_NULLS ON
# GO

# SET QUOTED_IDENTIFIER ON
# GO

# CREATE TABLE [dbo].[Contact](
# 	[ContactID] [int] IDENTITY(1,1) NOT NULL,
# 	[UserID] [int] NOT NULL,
# 	[FirstName] [varchar](50) NOT NULL,
# 	[LastName] [varchar](50) NOT NULL,
# 	[Email] [varchar](100) NOT NULL,
# 	[PhoneNumber] [varchar](20) NULL,
# 	[AddressLine1] [varchar](100) NULL,
# 	[AddressLine2] [varchar](100) NULL,
# 	[City] [varchar](50) NULL,
# 	[StateOrProvince] [varchar](50) NULL,
# 	[PostalCode] [varchar](20) NULL,
# 	[Country] [varchar](50) NULL,
# PRIMARY KEY CLUSTERED 
# (
# 	[ContactID] ASC
# )WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
# ) ON [PRIMARY]
# GO



# -----------------------------
# DATABASE CONTACT GENERATOR
# -----------------------------
class ContactGenerator:
    def __init__(self, server, database, driver="ODBC Driver 17 for SQL Server"):
        self.conn_string = (
            f"Driver={{{driver}}};Server={server};Database={database};Trusted_Connection=yes;"
        )
        self.conn = None
        self.cursor = None

        self.first_names = ['James', 'Mary', 'Robert', 'Patricia', 'Michael', 'Jennifer']
        self.last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia']

        self.city_to_state = {
            "New York": "NY",
            "Los Angeles": "CA",
            "San Francisco": "CA",
            "Houston": "TX",
            "Phoenix": "AZ",
            "Fort Worth": "TX",
            "Chicago": "IL",
        }

    def connect(self):
        self.conn = pyodbc.connect(self.conn_string)
        self.cursor = self.conn.cursor()
        print("✔️ Connected to database.")

    def truncate_contacts(self):
        try:
            self.cursor.execute("TRUNCATE TABLE Contact;")
            self.conn.commit()
            print("✔️ TRUNCATE successful.")
        except Exception:
            print("⚠️ TRUNCATE failed, using DELETE...")
            self.cursor.execute("DELETE FROM Contact;")
            self.conn.commit()
            print("✔️ Table cleared (DELETE).")

    def create_random_contact(self):
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        city = random.choice(list(self.city_to_state.keys()))
        state = self.city_to_state[city]

        return (
            0,
            first,
            last,
            f"{first.lower()}.{last.lower()}@example.com",
            "".join(random.choices("0123456789", k=10)),
            f"{random.randint(1,999)} {random.choice(['Main','Oak','Elm','Maple'])} St",
            f"Apt {random.randint(1,200)}" if random.choice([True, False]) else None,
            city,
            state,
            "".join(random.choices("0123456789", k=5)),
            "USA"
        )

    def insert_contacts(self, count=50):
        for _ in range(count):
            self.cursor.execute("""
                INSERT INTO Contact 
                (UserID, FirstName, LastName, Email, PhoneNumber, AddressLine1, 
                 AddressLine2, City, StateOrProvince, PostalCode, Country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, self.create_random_contact())
        self.conn.commit()
        print(f"✔️ Inserted {count} contacts.")

    def close(self):
        if self.conn:
            self.conn.close()
            print("✔️ Connection closed.")


# -----------------------------
# DB LOAD FUNCTION
# -----------------------------
def load_contacts_from_db(server, database):
    conn = pyodbc.connect(
        f"Driver={{ODBC Driver 17 for SQL Server}};Server={server};Database={database};Trusted_Connection=yes;"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT FirstName, LastName, Email, City, StateOrProvince FROM Contact")
    contacts = [{"Name": f"{f} {l}", "Email": e, "City": c, "State": s}
                for f, l, e, c, s in cursor.fetchall()]
    conn.close()
    return contacts


# -----------------------------
# TABLE RENDERING
# -----------------------------
def print_table(data):
    if not data:
        print("\nNo results found.\n")
        return
    headers = data[0].keys()
    rows = [list(d.values()) for d in data]
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid") + "\n")


# -----------------------------
# LANGGRAPH STATE
# -----------------------------
class ContactState(TypedDict):
    query: str
    results: List[Dict[str, Any]]
    answer: str


# -----------------------------
# GRAPH NODES
# -----------------------------
def vector_search_node(state: ContactState, embed_model, collection):
    query_embedding = embed_model.encode(state["query"]).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=10)

    matches = []
    for item in results["metadatas"][0]:
        matches.append({
            "Name": item["Name"],
            "Email": item["Email"],
            "City": item["City"],
            "State": item["State"]
        })

    state["results"] = matches
    return state


def llm_answer_node(state: ContactState, llm):
    if state["results"]:
        contacts_text = "\n".join([
            f"{d['Name']}, {d['Email']}, {d['City']}, {d['State']}"
            for d in state["results"]
        ])
        prompt_text = f"The following contacts were found:\n{contacts_text}\n\nAnswer the question using ONLY this data:\n{state['query']}"
    else:
        prompt_text = f"No matching contacts found. Respond accordingly:\n{state['query']}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers based only on provided contacts."),
        ("user", prompt_text)
    ])
    chain = prompt | llm | StrOutputParser()
    state["answer"] = chain.invoke({})
    return state


# -----------------------------
# MAIN PROGRAM WITH LANGGRAPH
# -----------------------------
if __name__ == "__main__":
    SERVER = r"MUHAMMAD\SQLEXPRESS"
    DATABASE = "CrimsonSky"

    # Step 1: Populate DB
    gen = ContactGenerator(SERVER, DATABASE)
    gen.connect()
    gen.truncate_contacts()
    gen.insert_contacts(50)
    gen.close()

    # Step 2: Load DB → Chroma
    print("Loading contacts...")
    contacts = load_contacts_from_db(SERVER, DATABASE)

    client = chromadb.Client()
    collection = client.create_collection("contacts")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = [
        embed_model.encode(f"{c['Name']}, {c['Email']}, {c['City']}, {c['State']}").tolist()
        for c in contacts
    ]

    for i, c in enumerate(contacts):
        collection.add(ids=[str(i)], metadatas=[c], embeddings=[embeddings[i]])

    # Step 3: LLaMA
    llm = ChatOllama(model="llama3", temperature=0.7)

    # Step 4: Build LangGraph Pipeline
    graph = StateGraph(ContactState)
    graph.add_node("vector_search", lambda s: vector_search_node(s, embed_model, collection))
    graph.add_node("llm_answer", lambda s: llm_answer_node(s, llm))

    graph.set_entry_point("vector_search")
    graph.add_edge("vector_search", "llm_answer")
    graph.add_edge("llm_answer", END)

    app = graph.compile()

    # Step 5: Console Loop
    print("\n🤖 LangGraph Contact Assistant Ready!\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        result = app.invoke({"query": user_query})

        print("\n🔍 Vector Store Matches:")
        print_table(result["results"])

        print("\n🦙 LLaMA3 Response:")
        print(result["answer"])
