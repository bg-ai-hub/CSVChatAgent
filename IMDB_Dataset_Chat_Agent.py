import os
from dotenv import load_dotenv
from pathlib import Path
from phi.model.openai import OpenAIChat
from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from phi.model.groq import Groq
import streamlit as st
import pandas as pd

load_dotenv()

cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")

if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

local_csv_path = "imdb_top_1000.csv"

if not os.path.exists(local_csv_path):
    raise FileNotFoundError(f"The file {local_csv_path} does not exists.")

python_agent = PythonAgent(
    #model=Groq(id="llama-3.3-70b-versatile"),
    model=OpenAIChat(id="gpt-4o-mini"),
    base_dir = tmp,
    files=[
        CsvFile(
            path=local_csv_path,
            description="A dataset containing information about movies"
        )
    ],
    markdown=True,
    show_tool_calls=False
)

data = {
        "Feature": [
            "Poster_Link", "Series_Title", "Released_Year", "Certificate", "Runtime", "Genre",
            "IMDB_Rating", "Overview", "Meta_score", "Director", "Star1", "Star2", "Star3", "Star4",
            "No_of_votes", "Gross"
        ],
        "Description": [
            "Link of the poster that IMDb is using",
            "Name of the movie",
            "Year at which that movie was released",
            "Certificate earned by that movie",
            "Total runtime of the movie",
            "Genre of the movie",
            "Rating of the movie at IMDb site",
            "Mini story/summary",
            "Score earned by the movie",
            "Name of the Director",
            "Name of the Star 1",
            "Name of the Star 2",
            "Name of the Star 3",
            "Name of the Star 4",
            "Total number of votes",
            "Money earned by that movie"
        ]
    }

df = pd.DataFrame(data)

question_list = []

def main():
    st.title("CSV Chatbot Agent")

    st.write("Welcome! Ask questions about the IMDB movie dataset, and I’ll fetch the answers for you.")

    #st.subheader("Movie Dataset Information")

    #st.table(df)

    question = st.text_area("Enter your question:", placeholder="e.g., Which movie has the highest IMDB rating?")

    if st.button("Run Flow"):
        if not question.strip():
            st.error("Please enter a valid question.")
            return

        try:
            with st.spinner("Processing your question..."):
                print(question_list)
                question_list.append(question)
                response = python_agent.run(question)  # Run the agent with the user query
                st.markdown(response.content)  # Display the response in Markdown format
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")  # Handle and display any errors
        print(question_list)

if __name__ == "__main__":
    main()
