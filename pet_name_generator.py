import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("Google API key not found. Please set it in the .env file.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY, temperature=0)

def generate_pet_name(animal_type, pet_color):
    
    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest me ten cool names for my pet."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

    response = name_chain.run({'animal_type': animal_type, 'pet_color': pet_color})

    return response


def add_custom_css():
    st.markdown("""
        <style>
        .reportview-container {
            background: #f0f2f6;
            color: black;
        }
        .sidebar .sidebar-content {
            background: #e0e4e8;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 8px 16px;
        }
        </style>
        """, unsafe_allow_html=True)
    

def main():
    add_custom_css()
    st.title("🐾 Pet Name Generator 🐾")
    st.markdown("### Get a unique name for your pet by providing its type and color!")

    animal_type = st.text_input("Enter the type of Pet (e.g., Dog, Cat):")
    pet_color = st.text_input("Enter the Color of the Pet:")

    if st.button("Generate Pet Names"):
        if animal_type and pet_color:
            with st.spinner("Generating..."):
                response = generate_pet_name(animal_type, pet_color)
                st.success("Here are some cool names for your pet:")
                st.write(response)
        else:
            st.warning("Please enter both the type and color of the pet.")

if __name__ == "__main__":
    main()
