import streamlit as st
import streamlit.components.v1 as stc
import sklearn
from ml_app import run_ml_app
from data_app import run_data_app

st.write("Scikit-learn version in Cloud:", sklearn.__version__)
html_temp = """
            <div style="background-color:#3872fb;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Employee Promotion Prediction App </h1>
		    <h4 style="color:white;text-align:center;">HR Team </h4>
		    </div>
            """


def main():
    menu = ["Home", 'Machine Learning','Data Chart']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("welcome to home page")

    elif choice == 'Machine Learning':
        run_ml_app()

    elif choice == 'Data Chart':
        run_data_app("data_sensus.csv")
if __name__ == '__main__':
    main()
        