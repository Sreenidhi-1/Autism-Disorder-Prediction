
import streamlit as st
import pandas as pd
import pickle
import lime
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns

css_style = """
    text-align:center;
    color: #800080; 
    font-size: 1000px; 
    font-family: Arial, sans-serif;  
"""
css_style1 = """
    align:center;
    width: 300px;
    padding: 10px;
    border: 2px solid #333;
    background-color:rgba(0,0,0, 0.5);
    font-family: Arial, sans-serif;
    text-align: center; 
    margin: 0 auto; /* Center the box horizontally */
    color: yellow;
"""

st.write("<h1 style='text-align: center;color: #800080;'>Prediction</h1>", unsafe_allow_html=True)

a_scores = {}
for i in range(1, 11):
    a_scores[f"A{i}_Score"] = st.number_input(f"A{i}_Score", min_value=0, max_value=1)

age = st.number_input("Age", min_value=0.0, format="%.2f")
ethnicity = st.number_input("Ethnicity", min_value=0)
jaundice = st.radio("Jaundice", ["Yes", "No"])
autism_in_family = st.radio("Autism in Family", ["Yes", "No"])
country_of_residence = st.number_input("Country of Residence", min_value=0)

result = st.number_input("Result", min_value=0.0, format="%.2f")
def map_yes_no(value):
    return 1 if value == "Yes" else 0
with open("D:/College/Sem 5/ML/best_models.pkl", "rb") as model_file:
    models = pickle.load(model_file)
model_names = ["None"] + list(models.keys())
selected_model = st.selectbox("Select Model", model_names)

if st.button("Predict"):
 
    labels = [
         "Age", "Ethnicity",
        "Jaundice", "Autism in Family", "Country of Residence", "Result"
    ]
    values=[]
    for i in range(1, 11):
        labels.append(f"A{i}_Score")
        values.append(a_scores[f"A{i}_Score"])
    values =values+ [
        age, ethnicity,
         map_yes_no(jaundice), map_yes_no(autism_in_family), country_of_residence, result
    ]
    

    names = [
        'A1_Score',
 'A2_Score',
 'A3_Score',
 'A4_Score',
 'A5_Score',
 'A6_Score',
 'A7_Score',
 'A8_Score',
 'A9_Score',
 'A10_Score',
 'age',
 'ethnicity',
 'jaundice',
 'austim',
 'contry_of_res',
 'result'
    ]
    input_data = dict(zip(names, values))
   
    column1, column2 = st.columns(2)

    with column1:
        st.write("Information")
        st.write(labels)
        
    with column2:
        st.write("Values")
        st.write(values)
    
    input_data_df = pd.DataFrame([input_data])
    st.write(input_data_df)

    
    prediction=models[selected_model].predict(input_data_df)
    prediction_text ="No Autism" if prediction == 0 else "Person has Autism"

    st.markdown(f'<div style="{css_style1}">{prediction_text}</div>', unsafe_allow_html=True)
    
    feature_names=["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
        "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
        "age", "ethnicity", "jaundice", "austim", "contry_of_res", "result"]
    dataset = pd.read_csv("D:/College/Sem 5/ML/featured_autism.csv")
    y = dataset["Class/ASD"]
    X = dataset.drop(columns=['gender',"Class/ASD",'used_app_before','age_desc','relation'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    explainer = LimeTabularExplainer(X_train.values, mode="classification", feature_names=feature_names)
    

    instance_to_explain = input_data_df.iloc[0].values

    explanation = explainer.explain_instance(instance_to_explain,models[selected_model].predict_proba)
    st.title("LIME Explanation")
    st.write("Prediction Probability:", models[selected_model].predict_proba([input_data_df.iloc[0]])[0][1])
    #st.write(explanation.as_list())
    explanation_data = pd.DataFrame(explanation.as_list(), columns=["Feature", "Weight"])
    st.table(explanation_data)
    try:
        feature_imp = pd.Series(models[selected_model].feature_importances_, index=X.columns).sort_values(ascending=False)
        fig2=plt.figure(figsize=(10,8))
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Feature Importance")
        plt.tight_layout()
        st.pyplot(fig2)
    except:
        pass

