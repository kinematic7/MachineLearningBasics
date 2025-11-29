import pyodbc
import random
import chromadb
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
# CONTACT GENERATOR CLASS
# -----------------------------
class ContactGenerator:
    def __init__(self, server, database, driver="ODBC Driver 17 for SQL Server"):
        self.conn_string = (
            f"Driver={{{driver}}};"
            f"Server={server};"
            f"Database={database};"
            f"Trusted_Connection=yes;"
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
        """
        Clears all rows from Contact table.
        Attempts TRUNCATE, falls back to DELETE if needed.
        """
        try:
            self.cursor.execute("TRUNCATE TABLE Contact;")
            self.conn.commit()
            print("✔️ Contact table truncated (TRUNCATE TABLE).")
        except Exception:
            print("⚠️ TRUNCATE failed, attempting DELETE...")
            self.cursor.execute("DELETE FROM Contact;")
            self.conn.commit()
            print("✔️ Contact table cleared (DELETE FROM).")

    def create_random_contact(self, user_id=0):
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)

        city = random.choice(list(self.city_to_state.keys()))
        state = self.city_to_state[city]

        return (
            user_id,
            first_name,
            last_name,
            f"{first_name.lower()}.{last_name.lower()}@example.com",
            ''.join(random.choices("0123456789", k=10)),
            f"{random.randint(1, 999)} {random.choice(['Main', 'Oak', 'Elm', 'Maple'])} St",
            f"Apt {random.randint(1, 200)}" if random.choice([True, False]) else None,
            city,
            state,
            ''.join(random.choices("0123456789", k=5)),
            "USA"
        )

    def insert_contacts(self, count=50):
        for _ in range(count):
            data = self.create_random_contact(user_id=0)
            self.cursor.execute("""
                INSERT INTO Contact 
                (UserID, FirstName, LastName, Email, PhoneNumber, AddressLine1, 
                 AddressLine2, City, StateOrProvince, PostalCode, Country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
        self.conn.commit()
        print(f"✔️ Inserted {count} contacts.")

    def close(self):
        if self.conn:
            self.conn.close()
            print("✔️ Connection closed.")


# -----------------------------
# LOAD CONTACTS FROM DB
# -----------------------------
def load_contacts_from_db(server, database):
    conn = pyodbc.connect(
        f"Driver={{ODBC Driver 17 for SQL Server}};"
        f"Server={server};Database={database};Trusted_Connection=yes;"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT FirstName, LastName, Email, City, StateOrProvince FROM Contact")
    contacts = [{"Name": f"{f} {l}", "Email": e, "City": c, "State": s} for f, l, e, c, s in cursor.fetchall()]
    conn.close()
    return contacts

# -----------------------------
# TABLE PRINTING FUNCTION
# -----------------------------
def print_table(data):
    if not data:
        print("No results found.")
        return
    headers = data[0].keys()
    rows = [list(d.values()) for d in data]
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid") + "\n")

# -----------------------------
# CHATBOT SCRIPT
# -----------------------------
if __name__ == "__main__":
    SERVER = r"MUHAMMAD\SQLEXPRESS"
    DATABASE = "CrimsonSky"

    # 1️⃣ Insert random contacts (optional)
    generator = ContactGenerator(server=SERVER, database=DATABASE)
    generator.connect()
    generator.truncate_contacts()    
    generator.insert_contacts(50)
    generator.close()

    # 2️⃣ Load contacts from DB
    print("Loading contacts from DB...")
    contacts = load_contacts_from_db(SERVER, DATABASE)

    # 3️⃣ Initialize Chroma client and embed contacts
    client = chromadb.Client()
    collection = client.create_collection("contacts")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = [embed_model.encode(f"{c['Name']}, {c['Email']}, {c['City']}, {c['State']}").tolist() for c in contacts]

    for i, c in enumerate(contacts):
        collection.add(ids=[str(i)], metadatas=[c], embeddings=[embeddings[i]])

    # 4️⃣ Initialize LLaMA3 via Ollama
    llm = ChatOllama(model="llama3", temperature=0.7)
    output_parser = StrOutputParser()

    print("\n🤖 Chatbot ready! Type your question about contacts. Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        # -----------------------------
        # Step 1: Query ChromaDB first
        # -----------------------------
        query_embedding = embed_model.encode(user_query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=10)

        table_data = []
        for item in results['metadatas'][0]:
            table_data.append({
                "Name": item["Name"],
                "Email": item["Email"],
                "City": item["City"],
                "State": item["State"]
            })

        # -----------------------------
        # Step 2: Pass results to LLaMA3
        # -----------------------------
        if table_data:
            contacts_text = "\n".join([f"{d['Name']}, {d['Email']}, {d['City']}, {d['State']}" for d in table_data])
            prompt_text = f"The following are contacts from the database:\n{contacts_text}\n\nAnswer the question based only on these contacts:\n{user_query}"
        else:
            prompt_text = f"No matching contacts found in the database. Answer accordingly:\n{user_query}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer only based on the contacts provided."),
            ("user", prompt_text)
        ])
        chain = prompt | llm | output_parser
        response = chain.invoke({"input": ""})

        # -----------------------------
        # Step 3: Show vector store matches and LLM response
        # -----------------------------
        print("\n🔍 ChromaDB Matches:\n")
        print_table(table_data)


        print("\n🦙 LLaMA3 Response:\n")
        print(response)

   