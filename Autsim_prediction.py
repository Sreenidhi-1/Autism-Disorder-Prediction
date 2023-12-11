import streamlit as st
import pandas as pd

st.set_page_config(page_title="Autism Prediction", page_icon="📈", layout="wide")
st.write("<h1 style='text-align: center;color:#800080;-webkit-text-stroke: 0.2px black;'>Autism Spectrum Disorder Prediction</h1>", unsafe_allow_html=True)
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")



# CSS style for formatting
css_style = """
<style>
  .container {
    background-color: rgba(0, 0, 0, 0.5);
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 5px;
    margin: 20px 0;
  }

  .title {
    font-size: 24px;
    font-weight: bold;
    align:center;
    margin-bottom: 10px;
    color:#800080;
  }

  .content {
    font-size: 16px;
    margin-bottom: 15px;
  }

  .about-data {
    font-weight: bold;
    margin-top: 10px;
  }

  .italic {
    font-style: italic;
  }
</style>
"""

# HTML content with CSS
html_content = f"""
<div class="container">
  <div class="title">Context</div>
  <div class="content">
    Autism, or autism spectrum disorder (ASD), refers to a broad range of conditions characterized by challenges with:
    <ul>
      <li>Social skills</li>
      <li>Repetitive behaviors</li>
      <li>Speech and nonverbal communication</li>
    </ul>
  </div>

  <div class="title">About the Data</div>
  <div class="content">
    The dataset used in this competition is composed of survey results for more than 700 people who filled out an app form.
    There are labels portraying whether the person received a diagnosis of autism.
  </div>

  <div class="about-data">
    <span class="italic">
  </div>
</div>
"""

# Display the content using st.markdown
st.markdown(css_style, unsafe_allow_html=True)
st.markdown(html_content, unsafe_allow_html=True)






dataset = pd.read_csv("D:\College\Sem 5\ML\cleaned_autism.csv")

st.write("Dataset:")
st.write(dataset)


st.write("<h3 style='text-align: center;color:#800080;-webkit-text-stroke: 0.2px black;'>Column Description</h3>", unsafe_allow_html=True)


column_descriptions = [
    {"Column": "ID", "Description": "ID of the patient"},
    {"Column": "A1_Score to A10_Score", "Description": "Score based on Autism Spectrum Quotient (AQ) 10 item screening tool"},
    {"Column": "age", "Description": "Age of the patient in years"},
    {"Column": "gender", "Description": "Gender of the patient"},
    {"Column": "ethnicity", "Description": "Ethnicity of the patient"},
    {"Column": "jaundice", "Description": "Whether the patient had jaundice at the time of birth"},
    {"Column": "autism", "Description": "Whether an immediate family member has been diagnosed with autism"},
    {"Column": "contry_of_res", "Description": "Country of residence of the patient"},
    {"Column": "used_app_before", "Description": "Whether the patient has undergone a screening test before"},
    {"Column": "result", "Description": "Score for AQ1-10 screening test"},
    {"Column": "age_desc", "Description": "Age of the patient"},
    {"Column": "relation", "Description": "Relation of the patient who completed the test"},
    {"Column": "Class/ASD", "Description": "Classified result as 0 or 1. Here 0 represents No and 1 represents Yes. This is the target column, and during submission, submit the values as 0 or 1 only."},
]

column_descriptions_df = pd.DataFrame(column_descriptions)

st.table(column_descriptions_df)
