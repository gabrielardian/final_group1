import streamlit as st
import pandas as pd
import sklearn
def run_data_app(csv_path):
    try:
        df = pd.read_csv(csv_path)

        if "Income" in df.columns:
            df["Income"] = df["Income"].astype(str).str.strip()

            # Buat kolom Age_Group
            if 'Age' in df.columns:
                age_bins = [20, 30, 40, 50, 60, 100]
                age_labels = ['20-29', '30-39', '40-49', '50-59', '60+']
                df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

            # Ambil kolom kategorikal + 'Age_Group'
            allowed_cols = [col for col in df.columns if col in ['Age_Group'] or df[col].dtype == 'object']
            allowed_cols = [col for col in allowed_cols if col != "Income"]

            if not allowed_cols:
                st.warning("Tidak ada kolom kategorikal atau numerik khusus ('Age') yang tersedia.")
            else:
                selected = st.selectbox("Pilih kolom:", allowed_cols)
                st.markdown(f"**Analisis berdasarkan '{selected}'**")

                # Jika kolom kategorikal (termasuk Age_Group)
                if df[selected].dtype == 'object' or selected == 'Age_Group':
                    grouped = df.groupby(selected)["Income"].value_counts().unstack().fillna(0)
                    st.subheader("Distribusi Income:")
                    st.bar_chart(grouped)
                else:
                    st.warning("Tipe kolom tidak dikenali.")
        else:
            st.warning("Kolom 'Income' tidak ditemukan di file CSV.")

    except Exception as e:
        st.error(f"Gagal memuat data CSV: {e}")


if __name__ == "__main__":
    csv_path = "data_sensus.csv"  # Ganti dengan path file CSV kamu
    run_data_app(csv_path)
