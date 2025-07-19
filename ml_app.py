import streamlit as st
import pandas as pd
import joblib

# Load model dan encoder
model = joblib.load("model.pkl")
column_order = joblib.load("columns.pkl")
label_encoders = joblib.load("label_encoders.pkl")
one_hot_encoder = joblib.load("one_hot_encoder.pkl")

# Kolom
label_cols = ['Gender']
one_hot_cols = ['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Education']

def preprocess_input(input_dict):
    df_new = pd.DataFrame([input_dict])

    # Strip (menghilangkan spasi awal dan akhir pada value string)
    for col in df_new.columns:
        if df_new[col].dtype == 'object':
            df_new[col] = df_new[col].astype(str).str.strip()

    # Label Encoding
    for col in label_cols:
        if col in df_new.columns:
            df_new[col] = label_encoders[col].transform(df_new[col].astype(str))

    # OneHot Encoding
    for col in one_hot_cols:
        if col not in df_new.columns:
            df_new[col] = 'Unknown'
        else:
            df_new[col] = df_new[col].astype(str).str.strip()

    df_ohe_part = df_new[one_hot_cols]
    ohe_array = one_hot_encoder.transform(df_ohe_part)
    ohe_columns = one_hot_encoder.get_feature_names_out(one_hot_cols)
    df_ohe = pd.DataFrame(ohe_array, columns=ohe_columns)

    # Gabung
    df_rest = df_new.drop(columns=one_hot_cols)
    final_df = pd.concat([df_rest.reset_index(drop=True), df_ohe.reset_index(drop=True)], axis=1)

    # Reindex
    final_df = final_df.reindex(columns=column_order, fill_value=0)
    return final_df

def run_ml_app():
    st.title("Income Prediction App")
    st.subheader("Masukkan data")

    input_dict = {
        'Age': st.number_input("Age", 17, 90, 30),
        'Workclass': st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked']),
        'Final Weight': st.number_input("Final Weight", 0, 1000000),
        'Education': st.selectbox("Education", ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Prof-school', 'Masters', 'Doctorate']),
        'Marital Status': st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']),
        'Occupation': st.selectbox("Occupation", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv']),
        'Relationship': st.selectbox("Relationship", ['Husband', 'Wife', 'Own-child', 'Unmarried', 'Not-in-family', 'Other-relative']),
        'Race': st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']),
        'Gender': st.radio("Gender", ['Male', 'Female']),
        'Hours Per Week': st.number_input("Hours Per Week", 1, 168),
    }

    st.write("### Data Input")
    st.json(input_dict)

    final_df = preprocess_input(input_dict)

    pred = model.predict(final_df)
    prob = model.predict_proba(final_df)

    st.subheader("Hasil Prediksi")
    income_label = label_encoders['Income'].inverse_transform(pred)[0]
    st.write(f"Prediksi Penghasilan: **{income_label}**")

    st.json({
        label_encoders['Income'].inverse_transform([0])[0]: f"{prob[0][0]:.2f}%",
        label_encoders['Income'].inverse_transform([1])[0]: f"{prob[0][1]:.2f}%"
    })

if __name__ == '__main__':
    run_ml_app()

 
    