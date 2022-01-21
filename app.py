import streamlit as st
# EDA Pkg
import pandas as pd
import joblib
import os
import numpy as np
from PIL import Image

# Data Viz Pkg
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Storing Data
import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS notestable(author TEXT,title TEXT,message TEXT)')


def add_data(author,title,message):
    c.execute('INSERT INTO notestable(author,title,message) VALUES (?,?,?)',(author,title,message))
    conn.commit()


def view_all_notes():
    c.execute('SELECT * FROM notestable')
    data = c.fetchall()
    # for row in data:
    #   print(row)
    return data

class Monitor(object):
    """docstring for Monitor"""

    conn = sqlite3.connect('data.db')
    c = conn.cursor()

    def __init__(self,age=None ,workclass=None ,fnlwgt=None ,education=None ,education_num=None ,marital_status=None ,occupation=None ,relationship=None ,race=None ,sex=None ,capital_gain=None ,capital_loss=None ,hours_per_week=None ,country=None,predicted_class=None,model_class=None):
        super(Monitor, self).__init__()
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education = education
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.country = country
        self.predicted_class = predicted_class
        self.model_class = model_class

    def __repr__(self):
        # return "Monitor(age ={self.age},workclass ={self.workclass},fnlwgt ={self.fnlwgt},education ={self.education},education_num ={self.education_num},marital_status ={self.marital_status},occupation ={self.occupation},relationship ={self.relationship},race ={self.race},sex ={self.sex},capital_gain ={self.capital_gain},capital_loss ={self.capital_loss},hours_per_week ={self.hours_per_week},native_country ={self.native_country},predicted_class ={self.predicted_class},model_class ={self.model_class})".format(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class)
        "Monitor(age = {self.age},workclass = {self.workclass},fnlwgt = {self.fnlwgt},education = {self.education},education_num = {self.education_num},marital_status = {self.marital_status},occupation = {self.occupation},relationship = {self.relationship},race = {self.race},sex = {self.sex},capital_gain = {self.capital_gain},capital_loss = {self.capital_loss},hours_per_week = {self.hours_per_week},country = {self.country},predicted_class = {self.predicted_class},model_class = {self.model_class})".format(self=self)

    def create_table(self):
        self.c.execute('CREATE TABLE IF NOT EXISTS predictiontable(age NUMERIC,workclass NUMERIC,fnlwgt NUMERIC,education NUMERIC,education_num NUMERIC,marital_status NUMERIC,occupation NUMERIC,relationship NUMERIC,race NUMERIC,sex NUMERIC,capital_gain NUMERIC,capital_loss NUMERIC,hours_per_week NUMERIC,country NUMERIC,predicted_class NUMERIC,model_class TEXT)')

    def add_data(self):
        self.c.execute('INSERT INTO predictiontable(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,country,predicted_class,model_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.country,self.predicted_class,self.model_class))
        self.conn.commit()

    def view_all_data(self):
        self.c.execute('SELECT * FROM predictiontable')
        data = self.c.fetchall()
        # for row in data:
        #   print(row)
        return data







def get_value(val,my_dict):
    for key ,value in my_dict.items():
        if val == key:
            return value

# Find the Key From Dictionary
def get_key(val,my_dict):
    for key ,value in my_dict.items():
        if val == value:
            return key

# Load Models
def load_model_n_predict(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model








def main():
    """ ML App with Streamlit"""
    st.title("Salary Predictor")
    st.subheader("ML Prediction App with Streamlit")

    # Load Dataset
    df = pd.read_csv("adult.csv")
    df2 = pd.read_csv("salary_encod_dataset.csv")

    # Preview Dataset
    activity = ["eda","prediction","countries","metrics","about"]
    choice = st.sidebar.selectbox("Choose Activity",activity)


    # CHOICE FOR EDA
    if choice == 'eda':
        st.text("Exploratory Data Analysis")
        if st.checkbox("Show Dataset"):
            number  = st.number_input("Number of Dataset to Preview")
            st.dataframe(df.head(int(number)))

        if st.button("Column Names"):
            st.write(df.columns)

        if st.checkbox("Shape of Dataset"):
            st.write(df.shape)
            data_dim = st.radio("Show Dimension by",("Rows","Columns"))

            if data_dim == 'Rows':
                st.text("Number of Rows")
                st.write(df.shape[0])
            elif data_dim == 'Columns':
                st.text("Number of Columns")
                st.write(df.shape[1])

        if st.checkbox("Select Columns To Show"):
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect('Select',all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

        if st.checkbox("Select Rows to Show"):
            selected_index = st.multiselect("Select Row",df.head(10).index)
            selected_rows = df.loc[selected_index]
            st.dataframe(selected_rows)

        if st.button("Value Counts"):
            st.text("Value Counts By Target/Class")
            st.write(df.iloc[:,-1].value_counts())

        st.subheader("Data Visualization")
        # Show Correlation Plots with Matplotlib Plot
        if st.checkbox("Correlation Plot [Matplotlib]"):
            plt.matshow(df.corr())
            st.pyplot()
        # Show Correlation Plots with  Seaborn Plot
        if st.checkbox("Correlation Plot with Annotation[Seaborn]"):
            st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot()

    # CHOICE FOR PREDICTION WITH ML
    if choice == 'prediction':
        st.text("Predict Salary")

        # st.markdown('<style>' + open('icon.css').read() + '</style>', unsafe_allow_html=True)
        # st.markdown('<i class="material-icons">face</i>', unsafe_allow_html=True)
        #https://fonts.googleapis.com/icon?family=Material+Icons

        # Dictionary of Mapped Values
        d_workclass = {' Self-emp-inc': 0,
         ' Without-pay': 1,
         ' Never-worked': 2,
         ' Private': 3,
         ' State-gov': 4,
         ' Federal-gov': 5,
         ' Local-gov': 6,
         ' Self-emp-not-inc': 7,
         ' ?': 8}

        d_education = {' Bachelors': 0,
         ' Some-college': 1,
         ' 12th': 2,
         ' 7th-8th': 3,
         ' Prof-school': 4,
         ' 9th': 5,
         ' HS-grad': 6,
         ' Doctorate': 7,
         ' 10th': 8,
         ' 11th': 9,
         ' Assoc-acdm': 10,
         ' Preschool': 11,
         ' Assoc-voc': 12,
         ' 5th-6th': 13,
         ' Masters': 14,
         ' 1st-4th': 15}

        d_marital_status = {' Never-married': 0,
         ' Widowed': 1,
         ' Separated': 2,
         ' Married-AF-spouse': 3,
         ' Divorced': 4,
         ' Married-spouse-absent': 5,
         ' Married-civ-spouse': 6}

        d_occupation = {' Armed-Forces': 0,
         ' Handlers-cleaners': 1,
         ' Prof-specialty': 2,
         ' Other-service': 3,
         ' Adm-clerical': 4,
         ' Sales': 5,
         ' Exec-managerial': 6,
         ' Craft-repair': 7,
         ' Transport-moving': 8,
         ' Tech-support': 9,
         ' Priv-house-serv': 10,
         ' Protective-serv': 11,
         ' Farming-fishing': 12,
         ' Machine-op-inspct': 13,
         ' ?': 14}

        d_relationship = {' Other-relative': 0,
         ' Unmarried': 1,
         ' Wife': 2,
         ' Own-child': 3,
         ' Not-in-family': 4,
         ' Husband': 5}

        d_race = {' White': 0,
         ' Amer-Indian-Eskimo': 1,
         ' Other': 2,
         ' Asian-Pac-Islander': 3,
         ' Black': 4}

        d_sex = {' Male': 0, ' Female': 1}

        d_country = {' Philippines': 0,
         ' Puerto-Rico': 1,
         ' Peru': 2,
         ' Greece': 3,
         ' South': 4,
         ' China': 5,
         ' Hungary': 6,
         ' Laos': 7,
         ' Taiwan': 8,
         ' France': 9,
         ' Haiti': 10,
         ' Ireland': 11,
         ' Outlying-US(Guam-USVI-etc)': 12,
         ' Hong': 13,
         ' Holand-Netherlands': 14,
         ' Poland': 15,
         ' Columbia': 16,
         ' Vietnam': 17,
         ' United-States': 18,
         ' Cuba': 19,
         ' Japan': 20,
         ' Nicaragua': 21,
         ' El-Salvador': 22,
         ' ?': 23,
         ' Jamaica': 24,
         ' Thailand': 25,
         ' Portugal': 26,
         ' Scotland': 27,
         ' Dominican-Republic': 28,
         ' Trinadad&Tobago': 29,
         ' Cambodia': 30,
         ' Italy': 31,
         ' India': 32,
         ' Guatemala': 33,
         ' Canada': 34,
         ' England': 35,
         ' Honduras': 36,
         ' Mexico': 37,
         ' Iran': 38,
         ' Ecuador': 39,
         ' Germany': 40,
         ' Yugoslavia': 41}


        d_salary = {' <=50K': 0, ' >50K': 1}

        # RECEIVE USER INPUT

        age = st.slider("Select Age",16,90)

        workclass = st.selectbox("Select Work Class",tuple(d_workclass.keys()))
        fnlwgt = st.number_input("Enter FNLWGT",1.228500e+04,1.484705e+06)
        education = st.selectbox("Select Education",tuple(d_education.keys()))
        education_num = st.slider("Select Education Level",1,16)
        marital_status = st.selectbox("Select Marital-status",tuple(d_marital_status.keys()))

        occupation = st.selectbox("Select Occupation",tuple(d_occupation.keys()))
        relationship = st.selectbox("Select Relationship",tuple(d_relationship.keys()))
        race = st.selectbox("Select Race",tuple(d_race.keys()))

        sex = st.radio("Select Sex",tuple(d_sex.keys()))

        capital_gain = st.number_input("Capital Gain",0,99999)

        capital_loss = st.number_input("Capital Loss",0,4356)

        hours_per_week = st.number_input("Hours Per Week ",0,99)
        country = st.selectbox("Select  Country",tuple(d_country.keys()))
        # USER INPUT ENDS HERE

        # GET VALUES FOR EACH INPUT
        k_workclass = get_value(workclass,d_workclass)
        k_education = get_value(education,d_education)
        k_marital_status = get_value(marital_status,d_marital_status)
        k_occupation = get_value(occupation,d_occupation)
        k_relationship = get_value(relationship,d_relationship)
        k_race = get_value(race,d_race)
        k_sex = get_value(sex,d_sex)
        k_country = get_value(country,d_country)

        # RESULT OF USER INPUT
        selected_options = [age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,country]
        vectorized_result = [age ,k_workclass ,fnlwgt ,k_education ,education_num ,k_marital_status ,k_occupation ,k_relationship ,k_race ,k_sex ,capital_gain ,capital_loss ,hours_per_week ,k_country]
        sample_data = np.array(vectorized_result).reshape(1, -1)
        st.info(selected_options)
        st.text("Using this encoding for prediction")
        st.success(vectorized_result)

        prettified_result = {"age":age,
        "workclass":workclass,
        "fnlwgt":fnlwgt,
        "education":education,
        "education_num":education_num,
        "marital_status":marital_status,
        "occupation":occupation,
        "relationship":relationship,
        "race":race,
        "sex":sex,
        "capital_gain":capital_gain,
        "capital_loss":capital_loss,
        "hours_per_week":hours_per_week,
        "country":country}

        st.subheader("Prettify JSON")
        st.json(prettified_result)

        # MAKING PREDICTION
        st.subheader("Prediction")
        if st.checkbox("Make Prediction"):
            all_ml_list = ["LR","RF","DT"]
            # Model Selection
            model_choice = st.selectbox('Model Choice',all_ml_list)
            prediction_label = {">50K": 0, "<=50K": 1}
            if st.button("Predict"):
                if model_choice == 'LR':
                    model_predictor = load_model_n_predict("models/salary_logistic_model.pkl")
                    prediction = model_predictor.predict(sample_data)
                elif model_choice == 'RF':
                    model_predictor = load_model_n_predict("models/salary_rf_model.pkl")
                    prediction = model_predictor.predict(sample_data)
                elif model_choice == 'DT':
                    model_predictor = load_model_n_predict("models/salary_dt_model.pkl")
                    prediction = model_predictor.predict(sample_data)
                    # st.text(prediction)
                final_result = get_key(prediction,prediction_label)
                monitor = Monitor(age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week , country,final_result,model_choice)
                monitor.create_table()
                monitor.add_data()
                st.success("Predicted Salary as :: {}".format(final_result))





    # CHOICE FOR COUNTRIES
    if choice == 'countries':
        st.text("Demographics")
        d_country = {' Philippines': 0,
        ' Puerto-Rico': 1,
        ' Peru': 2,
        ' Greece': 3,
        ' South': 4,
        ' China': 5,
        ' Hungary': 6,
        ' Laos': 7,
        ' Taiwan': 8,
        ' France': 9,
        ' Haiti': 10,
        ' Ireland': 11,
        ' Outlying-US(Guam-USVI-etc)': 12,
        ' Hong': 13,
        ' Holand-Netherlands': 14,
        ' Poland': 15,
        ' Columbia': 16,
        ' Vietnam': 17,
        ' United-States': 18,
        ' Cuba': 19,
        ' Japan': 20,
        ' Nicaragua': 21,
        ' El-Salvador': 22,
        ' ?': 23,
        ' Jamaica': 24,
        ' Thailand': 25,
        ' Portugal': 26,
        ' Scotland': 27,
        ' Dominican-Republic': 28,
        ' Trinadad&Tobago': 29,
        ' Cambodia': 30,
        ' Italy': 31,
        ' India': 32,
        ' Guatemala': 33,
        ' Canada': 34,
        ' England': 35,
        ' Honduras': 36,
        ' Mexico': 37,
        ' Iran': 38,
        ' Ecuador': 39,
        ' Germany': 40,
        ' Yugoslavia': 41}

        selected_countries = st.selectbox("Select Country",tuple(d_country.keys()))
        st.text(selected_countries)

        df2 = pd.read_csv("adult.csv")

        result_df = df2[df2['country'].str.contains(selected_countries)]
        st.dataframe(result_df)


        if st.checkbox("Select Columns To Show"):
                result_df_columns = result_df.columns.tolist()
                selected_columns = st.multiselect('Select',result_df_columns)
                new_df = df2[selected_columns]
                st.dataframe(new_df)

                if st.checkbox("Plot"):
                    st.area_chart(df[selected_columns])
                    st.pyplot()

    # METRICS CHOICE
    if choice == 'metrics':
        st.subheader("Metrics")
        # Create your connection.
        cnx = sqlite3.connect('data.db')

        mdf = pd.read_sql_query("SELECT * FROM predictiontable", cnx)
        st.dataframe(mdf)

    # ABOUT CHOICE
    if choice == 'about':
        st.subheader("About")
        st.markdown("""
            ### Salary Predictor ML App
            #### Built with Streamlit

            ### By
            + Rajat Paliwal


            """)





if __name__ == '__main__':
    main()
