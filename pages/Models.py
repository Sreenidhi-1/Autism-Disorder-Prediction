import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.metrics import classification_report as cr
import matplotlib.pyplot as plt
import plotly.express as px
from yellowbrick.classifier.classification_report import classification_report
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import roc_curve, auc,confusion_matrix
import plotly.express as px
from mlxtend.plotting import plot_confusion_matrix

dataset = pd.read_csv("Dataset/featured_autism.csv")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = cr(y_test, y_pred,output_dict=True)
    return accuracy, report


st.write("<h1 style='text-align: center;color: #800080;'>Autism Detection Model Evaluation</h1>", unsafe_allow_html=True)


y = dataset["Class/ASD"]
X = dataset.drop(columns=['gender',"Class/ASD",'used_app_before','age_desc','relation']) 



with open("best_models.pkl", "rb") as model_file:
    models = pickle.load(model_file)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.sidebar.title("Select Model")

selected_model = st.sidebar.selectbox("Select a model", list(models.keys()))
image_path = "classification_report.jpeg"



if st.button("Evaluate"):
    st.write(f"**Selected Model:** {selected_model}")
    model = models[selected_model]

    accuracy, report = evaluate_model(model, X_test, y_test)
    
    y_pred = model.predict(X_test)
    #st.pyplot(ROCAUC(model, classes=[0,1]))
    
  

    st.write(f"**Accuracy:** {accuracy}")
    st.write("**Classification Report:**")

    average_report_df = pd.DataFrame(report).transpose()

    metrics = ['precision', 'recall', 'f1-score']
    st.write(average_report_df)

    st.write("**Metrics Plot:**")
    fig = px.bar(average_report_df, x=average_report_df.index, y=metrics,
            labels={'index': 'Classes', 'value': 'Metrics'},
            title='Metrics for Two Classes',
            color_discrete_sequence=['#FFC3A0', '#A0FFC3', '#C3A0FF'])  # Color for bars

    fig.update_xaxes(title_font=dict(size=14))
    fig.update_yaxes(title_font=dict(size=14))
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig)
    fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test,y_pred), figsize=(2, 2), cmap=plt.cm.Blues)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    #plt.title('Confusion Matrix', fontsize=18)
    st.pyplot(fig)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    
    fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=['False Positive Rate','True Positive Rate'],
            width=700, height=700
        )
    fig.add_shape(
            type='line',
            x0=0, x1=1, y0=0, y1=1
        )
    st.plotly_chart(fig)



        
    









