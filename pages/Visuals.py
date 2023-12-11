import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datacleaner import autoclean
import warnings
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns

warnings.filterwarnings('ignore')

asd = pd.read_csv("D:/College/Sem 5/ML/autism.csv")
#print(asd.head())

def autism_gender():
    
    fig1=plt.figure(figsize=(10,5))
    sns.histplot(x= "gender", data=asd,shrink=.8 ,hue = "gender",palette ='flare')
    fig2=plt.figure(figsize=(12,5))
    sns.histplot(data=asd, x="gender", hue="Class/ASD", multiple="dodge", palette ='flare',shrink=.8)
    plt.title(' Counnts of Autism cases by gender')
    st.pyplot(fig1)
    st.pyplot(fig2)

def autism_ethnicity():
    
    fig1=plt.figure(figsize=(10, 6))
    sns.countplot(x='ethnicity', hue='gender', data=asd,palette='BuPu_r')
    plt.xticks(rotation=90)
    plt.xlabel("Ethnicity")
    plt.ylabel("Count")
    plt.title("Countplot with Hue (Ethnicity vs. Gender)")

    fig2=plt.figure(figsize=(16,6))
    ethnicity_plot=asd.groupby(['ethnicity','Class/ASD']).size().reset_index(name='Size')
    order=["White-European","?","Asian","Middle Eastern","Black","Latino","South Asian","Others","Pasifika","Hispanic","Turkish","others"]
    sns.barplot(data=ethnicity_plot,x="ethnicity",y="Size",hue='Class/ASD', palette='BuPu_r',order=order)
    plt.title("Autism by ethnicity")

    ethnicity_plt= asd["ethnicity"].value_counts().reset_index()
    ethnicity_plt.sort_values("ethnicity",ascending=False)
    fig3=plt.figure(figsize=(16,6))
    
    sns.barplot(data=ethnicity_plt,palette="BuPu_r",y="ethnicity",x="count")

    st.pyplot(fig1)
    st.pyplot(fig2)
    st.write(ethnicity_plt)
    st.pyplot(fig3)

def autism_country_res():
    
    fig1=plt.figure(figsize=(15, 5))
    sns.countplot(data=asd, x='contry_of_res', hue='Class/ASD',palette='PRGn')
    plt.xticks(rotation=90)
    plt.xlabel("Country of Residence")
    plt.ylabel("Count")
    plt.title("Countplot by Country of Residence with Hue")

    country_plt= asd["contry_of_res"].value_counts().reset_index()
    fig2=plt.figure(figsize=(20,6))
    sns.barplot(x="count",y="contry_of_res",data=country_plt.sort_values("contry_of_res",ascending=False).iloc[:10,:],palette="PRGn")
    #####
    fig3=plt.figure(figsize=(16,6))
    country_plt=asd.groupby(['contry_of_res','Class/ASD']).size().reset_index(name='Size')
    order=["United States","United Arab Emirates","New Zealand","India","United Kingdom","Australia","Jordan","Afghanistan","Sri Lanka","Canada"]
    sns.barplot(data=country_plt,x="contry_of_res",y="Size",hue='Class/ASD', palette='PRGn',order=order)
    plt.title(' Autism by country of residence ')
    
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.write(country_plt)
    st.pyplot(fig3)

def jaundice():
    fig, ax = plt.subplots(2, 1,figsize=(13,7))
    sns.countplot(x="jaundice",data=asd,palette="GnBu",ax=ax[0])
    jaundic_plt=asd.groupby(['jaundice','Class/ASD']).size().reset_index(name='Size')
    sns.barplot(data=jaundic_plt,x="jaundice",y="Size",hue='Class/ASD', palette='GnBu',ax=ax[1])
    plt.title(' Autism by jaundice ')
    st.pyplot(fig)

def autism_family():
    fig, ax = plt.subplots(2, 1,figsize=(12,7))
    sns.countplot(x="austim",data=asd,palette="Blues",ax=ax[0])
    autusim_plt=asd.groupby(['austim','Class/ASD']).size().reset_index(name='Size')
    sns.barplot(data=autusim_plt,x="austim",y="Size",hue='Class/ASD', palette='Blues',ax=ax[1])
    plt.title('autism presence in the family ')
    st.pyplot(fig)


    
def screen():
   
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    kde_plot = sns.displot(data=asd, x='result', hue="Class/ASD", kind="kde", palette="hot_r", fill=True, height=6, aspect=1.7)
    plt.title("Screening Test Result Distribution")
    fig, ax = plt.subplots(2, 1,figsize=(11,8))
    sns.countplot(x="used_app_before",data=asd,palette="hot_r",ax=ax[0])
    used_app_before_plt=asd.groupby(['used_app_before','Class/ASD']).size().reset_index(name='Size')
    sns.barplot(data=used_app_before_plt,x="used_app_before",y="Size",hue='Class/ASD', palette='hot_r',ax=ax[1])
    plt.title('Autism by screening test')
    st.pyplot(kde_plot)
    st.pyplot(fig)

def convertAge(age):
	if age < 4:
		return 'Toddler'
	elif age < 12:
		return 'Kid'
	elif age < 18:
		return 'Teenager'
	elif age < 40:
		return 'Young'
	else:
		return 'Senior'

asd['ageGroup'] = asd['age'].apply(convertAge)

def agegrp():
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x=asd['ageGroup'], hue=asd['Class/ASD'],palette='Wistia_r')
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.title("Countplot by Age Group with Hue")
    st.pyplot(plt)

def score():
    A_feat = ['A1_Score','A6_Score', 'A2_Score', 'A7_Score','A3_Score', 'A8_Score','A4_Score','A9_Score', 'A5_Score','A10_Score',]
    i = 1
    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 13))
    for col in A_feat:
        plt.subplot(5,2,i)
        sns.countplot(x=asd[col],data=asd,palette="ocean_r")
        i += 1
    st.pyplot(fig)

def cor():
    
    
    asd1=asd.drop(['age_desc'],axis=1)
    st.title("Correlation")
    ast = autoclean(asd1)
   
    corr = ast.corr()

    plt.figure(figsize=(15, 15))
    custom_cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(data=corr, annot=True, square=True, cbar=True,cmap=custom_cmap)
    st.write(ast.corr().style.background_gradient(cmap='BuPu_r'))
    st.pyplot(plt)
    



selected_option = st.sidebar.selectbox("Select an Option:", ["None","Visualize by feature", "Correlation of Features"])
if selected_option == "Visualize by feature":
    tabs = ["A Scores","Autism by Gender", "Autism by ethnicity", "Autism by Residence","Autism by Jaundice","Autism by Family","Screening","Age"]
    tab0,tab1,tab2,tab3,tab4,tab5,tab6, tab7= st.tabs(tabs=tabs)

    with tab0:
        score()
    with tab1:
        autism_gender()
    
    with tab2:
        autism_ethnicity()
    with tab3:
        autism_country_res()
    with tab4:
        jaundice()
    with tab5:
        autism_family()

    with tab6:
        screen()
    with tab7:
        agegrp()
elif selected_option == "Correlation of Features":
     cor()
