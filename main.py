import numpy as np
import streamlit as st
import pandas as pd
from vnstock import stock_historical_data
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


# Khởi tạo session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'size' not in st.session_state:
    st.session_state.size = None
if 'train' not in st.session_state:
    st.session_state.train = None
if 'test' not in st.session_state:
    st.session_state.test = None
if 'fourier_transform' not in st.session_state:
    st.session_state.fourier_transform = None
if 'stationarity_results' not in st.session_state:
    st.session_state.stationarity_results = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'data_ex' not in st.session_state:
    st.session_state.data_ex = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None


def ema(close, period=14):
    return close.ewm(span=period, adjust=False).mean()
def rsi(close, period=14):
    delta = close.diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(period).mean()
    avg_loss = abs(loss.rolling(period).mean())
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi
def macd(close, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line
def obv(close, volume):
    obv = np.where(close > close.shift(), volume, np.where(close < close.shift(), -volume, 0)).cumsum()
    return obv
st.title("Stock Price Prediction Model")

# Nhận input từ người dùng
symbol = st.text_input("Nhập mã cổ phiếu:", value="")
start_date = st.date_input("Chọn ngày bắt đầu:", value=pd.to_datetime("2013-01-02"))
end_date = st.date_input("Chọn ngày kết thúc:", value=pd.to_datetime("2023-06-30"))

# Tải dữ liệu
if st.button("Tải dữ liệu") or not st.session_state.data_loaded:
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    data = stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date)
    data.to_csv(f"{symbol}_DATA.csv", index=False)
    st.session_state.df = pd.read_csv(f"{symbol}_DATA.csv")
    st.session_state.df['time'] = pd.to_datetime(st.session_state.df['time'], format='%Y-%m-%d')
    st.session_state.data_loaded = True
    st.write(st.session_state.df.head())

# Chỉ hiển thị các phần tiếp theo nếu dữ liệu đã được tải
if st.session_state.data_loaded:
    # Vẽ biểu đồ giá cổ phiếu theo thời gian
    st.subheader(f"Biểu đồ giá cổ phiếu {symbol} theo thời gian")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(st.session_state.df['time'], st.session_state.df['close'], color='blue')
    ax.set_title(f'Giá {symbol} theo thời gian')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock price (VND)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Ma trận tương quan của dữ liệu
    st.session_state.numeric_df = st.session_state.df.select_dtypes(include=[np.number])
    correlation = st.session_state.numeric_df.corr()
    st.subheader("Biểu đồ ma trận tương quan giữa các số liệu")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='Blues', linewidths=.5, ax=ax_corr)
    st.pyplot(fig_corr)

    # Định dạng dữ liệu
    st.subheader("Thu gọn dữ liệu")
    data_ex = st.session_state.df.copy()
    data_ex = data_ex.reset_index()
    data_ex.set_index('time', inplace=True)
    data_ex = data_ex['close'].to_frame()
    st.write("Dữ liệu đã được xử lý xong")
    st.write(data_ex.head(10))


# Kiểm định tính dừng
    def test_stationarity(timeseries, title=""):
        result = adfuller(timeseries, autolag='AIC')
        output = []
        output.append(f"Kết quả kiểm định ADF cho {title}")
        output.append(f"Giá trị thống kê: {result[0]}")
        output.append(f"p-value: {result[1]}")
        output.append("Giá trị tới hạn:")
        for key, value in result[4].items():
            output.append(f"\t{key}: {value}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timeseries)
        ax.set_title(f'Biểu đồ chuỗi thời gian - {title} ')
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Giá trị')

        return output, fig


    if st.button("Kiểm định tính dừng") :
        st.session_state.stationarity_results = {}

        st.subheader("Kiểm định tính dừng cho chuỗi gốc")
        output_original, fig_original = test_stationarity(data_ex['close'], "Chuỗi gốc")
        st.session_state.stationarity_results['original'] = {'output': output_original, 'fig': fig_original}
        for line in output_original:
            st.write(line)
        st.pyplot(fig_original)

        diff1 = data_ex['close'].diff().dropna()
        st.subheader("Kiểm định tính dừng cho chuỗi sai phân bậc 1")
        output_diff, fig_diff = test_stationarity(diff1, "Sai phân bậc 1")
        st.session_state.stationarity_results['diff'] = {'output': output_diff, 'fig': fig_diff}
        for line in output_diff:
            st.write(line)
        st.pyplot(fig_diff)

        st.subheader("So sánh độ biến động")
        st.write(f"Độ lệch chuẩn chuỗi gốc: {data_ex['close'].std()}")
        st.write(f"Độ lệch chuẩn chuỗi sai phân bậc 1: {diff1.std()}")

    # Auto ARIMA
    if st.button("Kiểm tra tham số cho mô hình ARIMA"):
        # Hiển thị lại kết quả kiểm định tính dừng nếu có
        if st.session_state.stationarity_results:
            st.subheader("Kết quả kiểm định tính dừng (từ bước trước)")
            for test_type, results in st.session_state.stationarity_results.items():
                st.subheader(
                    f"Kiểm định tính dừng cho {'chuỗi gốc' if test_type == 'original' else 'chuỗi sai phân bậc 1'}")
                for line in results['output']:
                    st.write(line)
                st.pyplot(results['fig'])

        st.subheader("Kiểm tra tham số cho mô hình ARIMA")
        model = auto_arima(data_ex['close'], seasonal=False, trace=True)
        st.write(model.summary())

    # Chuẩn bị dữ liệu cho ARIMA và Fourier
    if st.session_state.size is None:
        st.session_state.size = int(len(data_ex) * 0.8)
        st.session_state.train, st.session_state.test = data_ex[:st.session_state.size], data_ex[st.session_state.size:]


    # Hàm dự đoán ARIMA
    def arima_forecast(history):
        model = ARIMA(history, order=(0, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        return output[0]


    # Dự đoán bằng ARIMA
    if st.button("Dự đoán bằng ARIMA") and st.session_state.predictions is None:
        X = st.session_state.df['close'].values
        size = int(len(X) * 0.8)
        train, test = X[:size], X[size:]
        st.session_state.history = [x for x in train]
        st.session_state.predictions = []
        for t in range(len(test)):
            yhat = arima_forecast(st.session_state.history)
            st.session_state.predictions.append(yhat)
            st.session_state.history.append(test[t])
        st.write("Dự đoán ARIMA đã hoàn thành.")
    # Vẽ biểu đồ dự đoán so với giá thực tế
    if st.session_state.predictions is not None:
        st.subheader("Biểu đồ dự đoán của ARIMA so với thực tế")
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        ax.plot(st.session_state.test.index, st.session_state.test['close'], label='Real')
        ax.plot(st.session_state.test.index, st.session_state.predictions, color='red', label='Predicted')
        ax.set_title('Dự đoán của ARIMA so với thực tế')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)

    # Biến đổi Fourier
    if st.button("Biến đổi Fourier"):
        close_fft = np.fft.fft(np.asarray(data_ex['close'].tolist()))
        fft_df = pd.DataFrame({'fft': close_fft})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

        st.session_state.fourier_transform = {
            'close_fft': close_fft,
            'fft_df': fft_df,
            'original_data': np.asarray(data_ex['close'].tolist())
        }

        st.write("Biến đổi Fourier đã được tính toán.")

    if st.session_state.fourier_transform is not None:
        fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
        ax.plot(st.session_state.fourier_transform['original_data'], label='Real')
        for num in [3, 6, 9]:
            fft_list = np.copy(st.session_state.fourier_transform['close_fft'])
            fft_list[num:-num] = 0
            ax.plot(np.fft.ifft(fft_list), label=f'Fourier transform với {num} thành phần')
        ax.set_xlabel('Ngày')
        ax.set_ylabel('VND')
        ax.set_title(f'Giá đóng cửa cổ phiếu {symbol} so với Fourier transform')
        ax.legend()
        st.pyplot(fig)
if st.button("Tinh toan cac chi bao ky thuat"):
    if st.session_state.data_ex is None:
        st.session_state.data_ex = data_ex.copy()

    st.session_state.data_ex['ema_20'] = ema(st.session_state.data_ex['close'], 20)
    st.session_state.data_ex['ema_50'] = ema(st.session_state.data_ex['close'], 50)
    st.session_state.data_ex['ema_100'] = ema(st.session_state.data_ex['close'], 100)
    st.session_state.data_ex['rsi'] = rsi(st.session_state.data_ex['close'])
    st.session_state.data_ex['macd'] = macd(st.session_state.data_ex['close'])
    volume_data = st.session_state.df.set_index('time')['volume']
    st.session_state.data_ex['obv'] = obv(st.session_state.data_ex['close'], volume_data)
    st.write(st.session_state.data_ex[['ema_20', 'ema_50', 'ema_100', 'rsi', 'macd', 'obv']].tail())
if st.button("Tổng hợp dữ liệu"):
    if (st.session_state.history is not None and
            st.session_state.fourier_transform is not None and
            st.session_state.data_ex is not None):

        arima_df = pd.DataFrame(st.session_state.history,
                                index=st.session_state.data_ex.index[-len(st.session_state.history):],
                                columns=['ARIMA'])

        # Biến đổi Fourier
        fft_df = st.session_state.fourier_transform['fft_df']
        fft_df.reset_index(inplace=True)
        fft_df['index'] = pd.to_datetime(st.session_state.data_ex.index)
        fft_df.set_index('index', inplace=True)
        fft_df_real = pd.DataFrame(np.real(fft_df['fft']), index=fft_df.index, columns=['Fourier_real'])
        fft_df_imag = pd.DataFrame(np.imag(fft_df['fft']), index=fft_df.index, columns=['Fourier_imag'])

        # Tổng hợp tất cả dữ liệu
        technical_indicators_df = st.session_state.data_ex[
            ['ema_20', 'ema_50', 'ema_100', 'rsi', 'macd', 'obv', 'close']]
        merged_df = pd.concat([arima_df, fft_df_real, fft_df_imag, technical_indicators_df], axis=1)
        merged_df = merged_df.dropna()

        st.write(merged_df.head())

        # Chuẩn bị dữ liệu cho mô hình
        train_size = int(len(merged_df) * 0.8)
        train_df, test_df = merged_df.iloc[:train_size], merged_df.iloc[train_size:]

        # Chuẩn hóa dữ liệu
        st.session_state.scaler = MinMaxScaler()
        train_scaled = st.session_state.scaler.fit_transform(train_df.drop('close', axis=1))
        test_scaled = st.session_state.scaler.transform(test_df.drop('close', axis=1))

        train_scaled_df = pd.DataFrame(train_scaled, columns=train_df.columns[:-1], index=train_df.index)
        test_scaled_df = pd.DataFrame(test_scaled, columns=test_df.columns[:-1], index=test_df.index)

        train_scaled_df['close'] = train_df['close']
        test_scaled_df['close'] = test_df['close']

        X_train = train_scaled_df.iloc[:, :-1].values
        y_train = train_scaled_df.iloc[:, -1].values
        st.session_state.X_test = test_scaled_df.iloc[:, :-1].values
        st.session_state.y_test = test_scaled_df.iloc[:, -1].values

        st.write("Dữ liệu đã chuẩn bị sẵn sàng cho mô hình")

        # Lưu dữ liệu đã chuẩn bị vào session state
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
    else:
        st.write(
            "Vui lòng chạy 'Dự đoán bằng ARIMA', 'Biến đổi Fourier', và 'Tinh toan cac chi bao ky thuat' trước khi tổng hợp dữ liệu.")
if st.button("Huấn luyện mô hình Deep Learning"):
    if 'X_train' in st.session_state and 'y_train' in st.session_state:
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=st.session_state.X_train.shape[1]))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

        history = model.fit(st.session_state.X_train, st.session_state.y_train, epochs=1000, batch_size=32, verbose=1,
                            validation_data=(st.session_state.X_test, st.session_state.y_test), callbacks=[early_stop],
                            shuffle=False)
        st.session_state.model = model
        st.write("Mô hình đã được huấn luyện xong")
    else:
        st.write("Vui lòng tổng hợp dữ liệu trước khi huấn luyện mô hình.")

if st.button("Đánh giá mô hình"):
    if st.session_state.model is not None:
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        mse = mean_squared_error(st.session_state.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(st.session_state.y_test, y_pred)
        r2 = r2_score(st.session_state.y_test, y_pred)
        evs = explained_variance_score(st.session_state.y_test, y_pred)
        mape = np.mean(np.abs((st.session_state.y_test - y_pred.flatten()) / st.session_state.y_test)) * 100
        mpe = np.mean((st.session_state.y_test - y_pred.flatten()) / st.session_state.y_test) * 100

        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"R2 Score: {r2}")
        st.write(f"Explained Variance Score: {evs}")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape}")
        st.write(f"Mean Percentage Error (MPE): {mpe}")

        # Lưu y_pred vào session state để sử dụng trong biểu đồ
        st.session_state.y_pred = y_pred
    else:
        st.write("Vui lòng huấn luyện mô hình trước khi đánh giá.")

if st.button("Biểu đồ dự đoán của mô hình Deep Learning"):
    if 'y_pred' in st.session_state:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        ax.plot(st.session_state.data_ex.index[-len(st.session_state.y_test):], st.session_state.y_test, label='Real')
        ax.plot(st.session_state.data_ex.index[-len(st.session_state.y_test):], st.session_state.y_pred, color='red',
                label='Predicted')
        ax.set_title('Dự đoán của mô hình so với thực tế')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("Vui lòng đánh giá mô hình trước khi vẽ biểu đồ dự đoán.")