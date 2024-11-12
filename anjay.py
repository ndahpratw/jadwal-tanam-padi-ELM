import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from hpelm import ELM
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io

# Fungsi untuk menghitung MAPE dengan aman (menghindari pembagian dengan nol)
def safe_mape(y_true, y_pred):
    mask = y_true != 0  # Hanya menghitung MAPE ketika nilai y_true tidak nol
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

# Fungsi untuk membuat dataset time series dengan lag
def create_dataset(data, lag=1):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i:(i + lag), 0])
        y.append(data[i + lag, 0])
    return np.array(X), np.array(y)

# Fungsi untuk pengujian berbagai rasio latih dan uji serta hidden neuron
def run_experiment(X, y, hidden_neurons_range, train_test_ratios):
    results = []
    for ratio in train_test_ratios:
        train_size = int(len(X) * ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        for hidden_neurons in hidden_neurons_range:
            # Membuat Model ELM
            elm = ELM(X_train.shape[1], 1)
            elm.add_neurons(hidden_neurons, 'sigm')
            elm.train(X_train, y_train, 'r')

            # Prediksi
            y_pred_test = elm.predict(X_test)
            
            # Evaluasi dengan MAPE, MAE, dan MSE
            error_mape = safe_mape(y_test, y_pred_test)
            error_mae = mean_absolute_error(y_test, y_pred_test)
            error_mse = mean_squared_error(y_test, y_pred_test)
            
            # Simpan hasilnya
            results.append({
                'Train/Test Ratio': ratio,
                'Hidden Neurons': hidden_neurons,
                'MAPE': error_mape,
                'MAE': error_mae,
                'MSE': error_mse
            })
    
    return pd.DataFrame(results)

# Streamlit app setup
st.title("Rainfall Prediction using ELM")

# Upload dataset
uploaded_file = st.file_uploader("Upload Excel file with data", type="xlsx")

if uploaded_file:
    data = pd.read_excel(uploaded_file, parse_dates=['Tanggal'])
    
    # Membersihkan data
    data['Curah Hujan (RR)'] = data['Curah Hujan (RR)'].replace(8888, 0)
    data['Curah Hujan (RR)'] = data['Curah Hujan (RR)'].fillna(0)
    
    # Normalisasi data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Curah Hujan (RR)']])
    
    # Pengaturan parameter untuk lag, hidden neurons, dan train/test ratio
    lag = st.slider("Select lag value", 1, 30, 7)
    hidden_neurons_range = st.slider("Select hidden neuron range", 1, 20, (1, 10))
    train_test_ratios = st.multiselect("Select train/test ratios", [0.7, 0.8, 0.9], [0.7, 0.8, 0.9])
    
    if st.button("Run Experiment"):
        # Membuat dataset dengan lag
        X, y = create_dataset(data_scaled, lag)
        
        # Menjalankan eksperimen
        results = run_experiment(X, y, range(hidden_neurons_range[0], hidden_neurons_range[1] + 1), train_test_ratios)
        
        # Menampilkan hasil sebagai tabel
        st.write("Experiment Results")
        st.dataframe(results)
        
        # Menyimpan hasil pengujian ke dalam file Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            results.to_excel(writer, index=False, sheet_name="Results")
            writer.save()
        
        # Tombol untuk mengunduh file Excel hasil pengujian
        st.download_button(
            label="Download Results as Excel",
            data=output.getvalue(),
            file_name="hasil_evaluasi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Mencari parameter terbaik berdasarkan MAPE terkecil
        best_result = results.loc[results['MAPE'].idxmin()]
        st.write("Best Parameters:")
        st.write(f"Train/Test Ratio: {best_result['Train/Test Ratio']}")
        st.write(f"Hidden Neurons: {best_result['Hidden Neurons']}")
        st.write(f"MAPE: {best_result['MAPE']}")
        st.write(f"MAE: {best_result['MAE']}")
        st.write(f"MSE: {best_result['MSE']}")
        
        # Visualisasi hasil MAPE, MAE, dan MSE
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot MAPE
        for ratio in train_test_ratios:
            subset = results[results['Train/Test Ratio'] == ratio]
            ax[0].plot(subset['Hidden Neurons'], subset['MAPE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
        ax[0].set_xlabel('Hidden Neurons')
        ax[0].set_ylabel('MAPE')
        ax[0].set_title('MAPE vs Hidden Neurons')
        ax[0].legend()
        
        # Plot MAE
        for ratio in train_test_ratios:
            subset = results[results['Train/Test Ratio'] == ratio]
            ax[1].plot(subset['Hidden Neurons'], subset['MAE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
        ax[1].set_xlabel('Hidden Neurons')
        ax[1].set_ylabel('MAE')
        ax[1].set_title('MAE vs Hidden Neurons')
        ax[1].legend()
        
        # Plot MSE
        for ratio in train_test_ratios:
            subset = results[results['Train/Test Ratio'] == ratio]
            ax[2].plot(subset['Hidden Neurons'], subset['MSE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
        ax[2].set_xlabel('Hidden Neurons')
        ax[2].set_ylabel('MSE')
        ax[2].set_title('MSE vs Hidden Neurons')
        ax[2].legend()
        
        st.pyplot(fig)