from tkinter import *
import customtkinter
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB   

df=pd.read_csv("Data.csv")
columns_header = list(df)
symptoms_list = columns_header[:-1]
disease = df['prognosis'].unique().tolist()
OPTIONS = sorted(symptoms_list)


#TRAINING DATA
training_data = df.groupby(df['prognosis'], group_keys=False).apply(lambda x: x.sample(5))
training_data.replace({'prognosis' : dict(zip(disease, list(range(0, 100))))}, inplace=True)
symptoms_value=[]
for i in range(0,len(symptoms_list)):
    symptoms_value.append(0)
x_train= training_data[symptoms_list]
y_train = training_data[['prognosis']]
np.ravel(y_train)

# TEST DATA
test_data = df.groupby(df['prognosis'], group_keys=False).apply(lambda x: x.sample(1))
#test_data = pd.read_csv("Testing.csv")
test_data.replace({'prognosis' : dict(zip(disease, list(range(0, 100))))}, inplace=True)
x_test = test_data[symptoms_list]
y_test = test_data[["prognosis"]]
np.ravel(y_test)


def randomforest():
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(x_train, np.ravel(y_train))

    y_pred=clf4.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    accuracyy = accuracy_score(y_test, y_pred)

    top_5_symptoms = [symptom_1_option_menu.get(),symptom_2_option_menu.get(),symptom_3_option_menu.get(),symptom_4_option_menu.get(),symptom_5_option_menu.get()]

    for j in range(0,len(symptoms_list)):
        for k in top_5_symptoms:
            if(k==symptoms_list[j]):
                symptoms_value[j]=1

    input_test = [symptoms_value]
    predict = clf4.predict(input_test)
    predicted=predict[0]

    match='no'
    for l in range(0,len(disease)):
        if(predicted == l):
            match='yes'
            break

    if (match=='yes'):
        random_forest_result.delete("1.0", END)
        random_forest_result.insert(END, disease[l])
        accuracy_result.delete(accuracyy, END)
        accuracy_result.insert(END, accuracyy)
    else:
        random_forest_result.delete("1.0", END)
        random_forest_result.insert(END, "Not Found")

def NaiveBayes():
    gnb = GaussianNB()
    gnb=gnb.fit(x_train,np.ravel(y_train))  

    y_pred=gnb.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    accuracyy = accuracy_score(y_test, y_pred)
 

    top_5_symptoms = [symptom_1_option_menu.get(),symptom_2_option_menu.get(),symptom_3_option_menu.get(),symptom_4_option_menu.get(),symptom_5_option_menu.get()]
    for j in range(0,len(symptoms_list)):
        for k in top_5_symptoms:
            if(k==symptoms_list[j]):
                symptoms_value[j]=1

    inputtest = [symptoms_value]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    match='no'
    for l in range(0,len(disease)):
        if(predicted == l):
            match='yes'
            break

    if (match=='yes'):
        naive_bayes_result.delete("1.0", END)
        naive_bayes_result.insert(END, disease[l])
        accuracy_result.delete(accuracyy, END)
        accuracy_result.insert(END, accuracyy)
    else:
        naive_bayes_result.delete("1.0", END)
        naive_bayes_result.insert(END, "Not Found")



root = customtkinter.CTk()
root.title("DISEASE PREDICTOR")
root.geometry("900x600")
root.grid_columnconfigure((0, 1), weight=1)
root.grid_rowconfigure((0), weight=1)


    
sidebar_frame = customtkinter.CTkFrame(root, width=150, corner_radius=10)
sidebar_frame.grid(row=0, column=0)

sidebar_label = customtkinter.CTkLabel(sidebar_frame, text = "Select the symptoms below:", font = customtkinter.CTkFont(size=20, weight="bold"))
sidebar_label.grid(row=2, column=0, padx=20, pady=10)

symptom_1_label = customtkinter.CTkLabel(sidebar_frame, text = "Select First Symptom:", font = customtkinter.CTkFont(size=16, weight="normal"))
symptom_1_label.grid(row=3, column=0)
symptom_1_option_menu = customtkinter.CTkOptionMenu(sidebar_frame, values = OPTIONS)
symptom_1_option_menu.set("None")
symptom_1_option_menu.grid(row=4, column=0, padx=20, pady=0)

symptom_2_label = customtkinter.CTkLabel(sidebar_frame, text = "Select Second Symptom:", font = customtkinter.CTkFont(size=16, weight="normal"))
symptom_2_label.grid(row=5, column=0)
symptom_2_option_menu = customtkinter.CTkOptionMenu(sidebar_frame, values = OPTIONS)
symptom_2_option_menu.set("None")
symptom_2_option_menu.grid(row=6, column=0)

symptom_3_label = customtkinter.CTkLabel(sidebar_frame, text = "Select Third Symptom:", font = customtkinter.CTkFont(size=16, weight="normal"))
symptom_3_label.grid(row=7, column=0)
symptom_3_option_menu = customtkinter.CTkOptionMenu(sidebar_frame, values = OPTIONS)
symptom_3_option_menu.set("None")
symptom_3_option_menu.grid(row=8, column=0, padx=20)

symptom_4_label = customtkinter.CTkLabel(sidebar_frame, text = "Select Fourth Symptom:", font = customtkinter.CTkFont(size=16, weight="normal"))
symptom_4_label.grid(row=9, column=0)
symptom_4_option_menu = customtkinter.CTkOptionMenu(sidebar_frame, values = OPTIONS)
symptom_4_option_menu.grid(row=10, column=0)
symptom_4_option_menu.set("None")

symptom_5_label = customtkinter.CTkLabel(sidebar_frame, text = "Select Fifth Symptom:", font = customtkinter.CTkFont(size=16, weight="normal"))
symptom_5_label.grid(row=11, column=0)
symptom_5_option_menu = customtkinter.CTkOptionMenu(sidebar_frame, values = OPTIONS)
symptom_5_option_menu.set("None")
symptom_5_option_menu.grid(row=12, column=0)

def reset():
    symptom_1_option_menu.set("None")
    symptom_2_option_menu.set("None")
    symptom_3_option_menu.set("None")
    symptom_4_option_menu.set("None")
    symptom_5_option_menu.set("None")

reset_button = customtkinter.CTkButton(sidebar_frame, border_width=2, text = "Reset", font = customtkinter.CTkFont(size=16, weight="bold"), command=reset)
reset_button.grid(row=13, column=0, pady = 20)



button_frame = customtkinter.CTkFrame(root, width=10, corner_radius=5)
button_frame.grid(row=0, column=1)

button_label = customtkinter.CTkLabel(button_frame, text = "Press the button below for prediction results:", font = customtkinter.CTkFont(size=20, weight="bold"))
button_label.grid(row=2, column=0, padx=20, pady=10)

random_forest_button = customtkinter.CTkButton(button_frame, border_width=2, text = "Random Forest", font = customtkinter.CTkFont(size=16, weight="normal"), command=randomforest)
random_forest_button.grid(row=3, column=0, pady = 10)

naive_bayes_button = customtkinter.CTkButton(button_frame,border_width=2, text = "Naive Bayes", font = customtkinter.CTkFont(size=16, weight="normal"), command=NaiveBayes)
naive_bayes_button.grid(row=4, column=0, pady = 10)

random_forest_result_label = customtkinter.CTkLabel(button_frame, text = "Prediction result using Random Forest is:", font = customtkinter.CTkFont(size=16, weight="normal"))
random_forest_result_label.grid(row=5, column=0)
random_forest_result = customtkinter.CTkTextbox(button_frame, width = 200, height = 5)
random_forest_result.grid(row=6, column = 0)

naive_bayes_result_label = customtkinter.CTkLabel(button_frame, text = "Prediction result using Naive Bayes is:", font = customtkinter.CTkFont(size=16, weight="normal"))
naive_bayes_result_label.grid(row=7, column=0)
naive_bayes_result = customtkinter.CTkTextbox(button_frame, width = 200, height = 5)
naive_bayes_result.grid(row=8, column = 0)

accuracy_label = customtkinter.CTkLabel(button_frame, text = "Accuracy:", font = customtkinter.CTkFont(size=16, weight="bold"))
accuracy_label.grid(row=9, column=0)
accuracy_result = customtkinter.CTkTextbox(button_frame, width = 200, height = 5)
accuracy_result.grid(row=10, column = 0)

label = customtkinter.CTkLabel(button_frame, text = "")
label.grid(row=11, column=0)

root.mainloop()
