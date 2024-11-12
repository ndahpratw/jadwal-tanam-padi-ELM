import numpy as np
import pandas as pd
import streamlit as st
from hpelm import ELM
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.markdown(
    "<h2 style='text-align: center;'><i>Extreame Learning Machine</i> Untuk Memprediksi Curah Hujan Dalam Penentuan Jadwal Tanam Padi</h2><br><br><br>", unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu("Main Menu", ['Dataset', 'Preprocessing', 'Modelling', 'Prediction'], default_index=3)

data = pd.read_excel('data.xlsx', parse_dates=['Tanggal'])
banyak_data = data.shape[0]
banyak_8888 = (data == 8888).sum().sum()
banyak_9999 = (data == 9999).sum().sum()
banyak_data_hilang = data.isna().sum().sum()

dataset = pd.read_excel('lagged data.xlsx')
results = pd.read_excel('hasil evaluasi.xlsx')

def create_dataset(data, lag=1):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i:(i + lag), 0])
        y.append(data[i + lag, 0])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Curah Hujan (RR)']])

if (selected == 'Dataset'):
    st.info("Data curah hujan harian diperoleh dari Badan Meteorologi, Klimatologi, dan Geofisika (BMKG). Kabupaten Bangkalan tidak memiliki stasiun pengamatan cuaca, sehingga data curah hujan yang diolah dari hasil pengamatan stasiun pengamatan cuaca terdekat, yaitu Stasiun Meteorologi Perak I Surabaya.")
    st.markdown("[Badan Meteorologi, Klimatologi, dan Geofisika (BMKG)](https://www.bmkg.go.id/)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Data Curah Hujan Harian')
        data

    with col2:
        st.subheader('Informasi Data')
        st.write("Data yang diolah merupakan data curah hujan harian dalam kurun waktu Januari 2020 – Juli 2024")
        st.write("Total = ", banyak_data, "data")
        st.markdown(f"""
        Terdapat : 
           1. Jumlah nilai 8888 (Tidak diukur) : {banyak_8888} data
           2. Jumlah nilai 9999 (Tidak diukur) : {banyak_9999} data
           3. Jumlah data kosong (None)        : {banyak_data_hilang} data
        """)

if (selected == 'Preprocessing'):
    st.info("""
    Adapun tahapan - tahapan yang akan dilakukan pada persiapan data ini adalah :
    1. Data Imputation
    2. Transformasi data
    3. Lag Feature
    4. Normalisasi Data
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Imputation", "Data Tranformation", "Normalisasi Data", "Lag Feature", "Dataset"])
    # Imputasi data
    with tab1:
        st.warning('Data Imputation')
        missing_values = data.isnull().sum()
        st.write("Jumlah Missing Values dalam Setiap Kolom : ", missing_values)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Data Sebelum')
            data

        with col2:
            st.subheader('Data Sesudah')
            data['Curah Hujan (RR)'] = data['Curah Hujan (RR)'].replace(8888, 0)
            data

    # Tranformasi data
    with tab2:
        st.warning('Data Transformation')
        jumlah_8888 = (data['Curah Hujan (RR)'] == 8888).sum()
        st.write(f'Jumlah data dengan nilai 8888 : {jumlah_8888}')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Data Sebelum')
            data

        with col2:
            st.subheader('Data Sesudah')
            data['Curah Hujan (RR)'] = data['Curah Hujan (RR)'].fillna(0)
            data

    # Normalisasi data
    with tab3:
        st.warning('Normalisasi Data')
        st.info("pake minmaxscaler, Max nya segini, Min nya segini")
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[['Curah Hujan (RR)']])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Data Sebelum')
            data['Curah Hujan (RR)'] 

        with col2:
            st.subheader('Data Sesudah')
            data_scaled
        
    with tab4:
        st.warning('Lag Feature')
        
        X, y = create_dataset(data_scaled, 7)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Fitur Data')
            X 

        with col2:
            st.subheader('Target')
            y

    with tab5:
        st.warning('Dataset')
        # Convert X and y to DataFrames
        df_X = pd.DataFrame(X, columns=[f'Lag_{i+1}' for i in range(X.shape[1])])
        df_y = pd.DataFrame(y, columns=['Target'])

        # Save directly to an Excel file
        with pd.ExcelWriter("lagged data.xlsx") as writer:
            df_X.to_excel(writer, sheet_name='Fitur Data', index=False)
            df_y.to_excel(writer, sheet_name='Target', index=False)
        
        dataset

if (selected == 'Modelling'):
    # Fungsi untuk pengujian berbagai rasio latih dan uji serta hidden neuron
    def run_experiment(dataset, hidden_neurons_range, train_test_ratios):
        X = dataset.drop(columns=['Lag_7'])
        y = dataset['Lag_7']
        results = []
        for ratio in train_test_ratios:
            train_size = int(len(X) * ratio)
            X_train, X_test = X[:train_size].values, X[train_size:].values  # Convert to numpy arrays
            y_train, y_test = y[:train_size].values, y[train_size:].values 
            
            for hidden_neurons in hidden_neurons_range:
                # 4. Membuat Model ELM
                elm = ELM(X_train.shape[1], 1)
                elm.add_neurons(hidden_neurons, 'sigm')
                elm.train(X_train, y_train, 'r')

                # 5. Prediksi
                y_pred_test = elm.predict(X_test)
                
                # 6. Evaluasi dengan MAPE, MAE, dan MSE
                error_mape = safe_mape(y_test, y_pred_test)  # Menggunakan safe_mape
                error_mae = mean_absolute_error(y_test, y_pred_test)
                error_mse = mean_squared_error(y_test, y_pred_test)
                
                # Simpan hasilnya
                results.append({
                    'Train/Test Ratio': ratio,
                    'Hidden Neurons': hidden_neurons,
                    'MAPE': error_mape,  # Simpan sebagai desimal
                    'MAE': error_mae,
                    'MSE': error_mse
                })
        
        return pd.DataFrame(results)

    # Fungsi untuk menghitung MAPE dengan aman (menghindari pembagian dengan nol)
    def safe_mape(y_true, y_pred):
        mask = y_true != 0  # Hanya menghitung MAPE ketika nilai y_true tidak nol
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)  # Kembali ke format desimal
    
    tab1, tab2 = st.tabs(["Hasil Pengujian", "Uji Coba"])
    with tab1:
        # Skenario Pengujian
        hidden_neurons_range = range(1, 11)  # Hidden neuron dari 1 hingga 10
        train_test_ratios = [0.7, 0.8, 0.9]  # Rasio latih dan uji 70:30, 80:20, 90:10

        # Menjalankan eksperimen
        results = run_experiment(dataset, hidden_neurons_range, train_test_ratios)

        # Menyimpan hasil pengujian ke dalam file Excel
        output_excel = "hasil evaluasi.xlsx"
        results.to_excel(output_excel, index=False)

        st.write(f"Hasil pengujian telah disimpan ke {output_excel}")

        # 9. Mencari parameter terbaik (berdasarkan MAPE terkecil)
        best_result = results.loc[results['MAPE'].idxmin()]  # Ambil yang terendah

        # Menampilkan parameter terbaik
        st.write(f"Parameter terbaik ditemukan pada:")
        st.write(f"Train/Test Ratio: {best_result['Train/Test Ratio']}")
        st.write(f"Hidden Neurons: {best_result['Hidden Neurons']}")
        st.write(f"MAPE: {best_result['MAPE']}")  # Tampilkan MAPE sebagai desimal
        st.write(f"MAE: {best_result['MAE']}")
        st.write(f"MSE: {best_result['MSE']}")

        # 7. Visualisasi hasil MAPE, MAE, dan MSE
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))  # Create a single figure with 3 subplots

        # Plot MAPE
        for ratio in train_test_ratios:
            subset = results[results['Train/Test Ratio'] == ratio]
            axes[0].plot(subset['Hidden Neurons'], subset['MAPE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
        axes[0].set_xlabel('Hidden Neurons')
        axes[0].set_ylabel('MAPE')
        axes[0].set_title('MAPE vs Hidden Neurons')
        axes[0].legend()

        # Plot MAE
        for ratio in train_test_ratios:
            subset = results[results['Train/Test Ratio'] == ratio]
            axes[1].plot(subset['Hidden Neurons'], subset['MAE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
        axes[1].set_xlabel('Hidden Neurons')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('MAE vs Hidden Neurons')
        axes[1].legend()

        # Plot MSE
        for ratio in train_test_ratios:
            subset = results[results['Train/Test Ratio'] == ratio]
            axes[2].plot(subset['Hidden Neurons'], subset['MSE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
        axes[2].set_xlabel('Hidden Neurons')
        axes[2].set_ylabel('MSE')
        axes[2].set_title('MSE vs Hidden Neurons')
        axes[2].legend()

        plt.tight_layout()  # Adjust layout for better spacing

        # Display the figure in Streamlit
        st.pyplot(fig)

    with tab2:
        # Input for minimum and maximum hidden neurons
        hidden_neurons_min = st.slider('Minimum hidden neurons', min_value=1, max_value=20, value=1)
        hidden_neurons_max = st.slider('Maximum hidden neurons', min_value=1, max_value=20, value=10)

        # Input for train-test ratios
        train_test_ratios = st.multiselect(
            'Select train-test ratios',
            options=[0.7, 0.8, 0.9],
            default=[0.7, 0.8, 0.9]  # Default selected ratios
        )

        # When the "Run Experiment" button is clicked
        if st.button('Run Experiment'):
            # Generate hidden neurons range based on user input
            hidden_neurons_range = range(hidden_neurons_min, hidden_neurons_max + 1)

            # Run the experiment
            results = run_experiment(dataset, hidden_neurons_range, train_test_ratios)

            # Find the best result based on the smallest MAPE
            best_result = results.loc[results['MAPE'].idxmin()]  # Get the minimum MAPE row

            # Display the best result
            st.write("Best parameters found:")
            st.write(f"Train/Test Ratio: {best_result['Train/Test Ratio']}")
            st.write(f"Hidden Neurons: {best_result['Hidden Neurons']}")
            st.write(f"MAPE: {best_result['MAPE']}")  # Display MAPE as decimal
            st.write(f"MAE: {best_result['MAE']}")
            st.write(f"MSE: {best_result['MSE']}")

            # Visualize the results for MAPE, MAE, and MSE
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))  # Create a figure with 3 subplots

            # Plot MAPE
            for ratio in train_test_ratios:
                subset = results[results['Train/Test Ratio'] == ratio]
                axes[0].plot(subset['Hidden Neurons'], subset['MAPE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
            axes[0].set_xlabel('Hidden Neurons')
            axes[0].set_ylabel('MAPE')
            axes[0].set_title('MAPE vs Hidden Neurons')
            axes[0].legend()

            # Plot MAE
            for ratio in train_test_ratios:
                subset = results[results['Train/Test Ratio'] == ratio]
                axes[1].plot(subset['Hidden Neurons'], subset['MAE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
            axes[1].set_xlabel('Hidden Neurons')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('MAE vs Hidden Neurons')
            axes[1].legend()

            # Plot MSE
            for ratio in train_test_ratios:
                subset = results[results['Train/Test Ratio'] == ratio]
                axes[2].plot(subset['Hidden Neurons'], subset['MSE'], label=f"Ratio {int(ratio*100)}:{int((1-ratio)*100)}")
            axes[2].set_xlabel('Hidden Neurons')
            axes[2].set_ylabel('MSE')
            axes[2].set_title('MSE vs Hidden Neurons')
            axes[2].legend()

            plt.tight_layout()  # Adjust layout for better spacing

            # Display the figure in Streamlit
            st.pyplot(fig)
            
if (selected == 'Prediction'):
    # Mencari parameter terbaik (berdasarkan MAPE terkecil)
    best_result = results.loc[results['MAPE'].idxmin()]
    hidden_neurons_best = int(best_result['Hidden Neurons'])  # Pastikan menjadi integer

    # 4. Membuat Model ELM dengan parameter terbaik
    X = dataset.drop(columns=['Lag_7']).values
    y = dataset['Lag_7'].values
    elm = ELM(X.shape[1], 1)
    elm.add_neurons(hidden_neurons_best, 'sigm')
    elm.train(X, y)  # Latih model dengan seluruh dataset

    # 5. Prediksi 365 hari ke depan
    # Buat array untuk menyimpan prediksi
    predictions = []
    current_input = data_scaled[-6:].flatten()  # Mengambil data terakhir untuk prediksi
    for _ in range(365):
        # Prediksi
        pred = elm.predict(current_input.reshape(1, -1))
        predictions.append(pred[0][0])
        
        # Update input dengan memasukkan prediksi terbaru
        current_input = np.roll(current_input, -1)  # Geser input
        current_input[-1] = pred  # Masukkan prediksi ke input

    # 6. Balikkan normalisasi data
    predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predictions_rescaled

    # 7. Membuat DataFrame dari hasil prediksi
    future_dates = pd.date_range(start=data['Tanggal'].max() + pd.Timedelta(days=1), periods=365)
    predicted_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Curah Hujan (RR) Prediksi': predictions_rescaled.flatten()
    })

    # 8. Mengelompokkan berdasarkan bulan untuk menentukan jadwal tanam padi
    predicted_df['Bulan'] = predicted_df['Tanggal'].dt.month

    # Dapatkan bulan dari tanggal terakhir pada dataset asli
    last_month = data['Tanggal'].max().month

    # Menggeser bulan sehingga dimulai dari bulan terakhir
    predicted_df['Bulan'] = (predicted_df['Bulan'] + (last_month - 1)) % 12 + 1

    monthly_summary = predicted_df.groupby('Bulan')['Curah Hujan (RR) Prediksi'].sum().reset_index()

    # Tampilkan hasil ringkasan bulanan
    st.write("Ringkasan Prediksi Bulanan:")
    st.write(monthly_summary)

    fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure and axis
    ax.plot(predicted_df['Tanggal'], predicted_df['Curah Hujan (RR) Prediksi'], label="Prediksi Curah Hujan")
    ax.set_title('Prediksi Curah Hujan 365 Hari ke Depan')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Curah Hujan (mm)')
    ax.legend()

    # Display the figure in Streamlit
    st.pyplot(fig)

    