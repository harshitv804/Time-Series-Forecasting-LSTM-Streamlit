import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from torch import nn
from sklearn.metrics import mean_squared_error
import time
from datetime import timedelta
import gc
import plotly.figure_factory as ff

st.set_page_config(layout="wide",page_title="Multi-Step Time Series Forecasting LSTM", page_icon="https://github.com/harshitv804/Time-Series-Forecasting-LSTM-Streamlit/assets/100853494/0d39dccf-42c7-4062-8000-3dc5646a0445")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
col3, col4 = st.columns([1,5])
col1, col2 = st.columns([3,2])

class LSTMForecasting(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, linear_hidden_size, lstm_num_layers, linear_num_layers, output_size):
        super(LSTMForecasting, self).__init__()
        self.linear_hidden_size = linear_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.linear_num_layers = linear_num_layers
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.linear_layers = nn.ModuleList()
        self.linear_num_layers-=1
        self.linear_layers.append(nn.Linear(self.lstm_hidden_size, self.linear_hidden_size))

        for _ in range(linear_num_layers): 
            self.linear_layers.append(nn.Linear(self.linear_hidden_size, int(self.linear_hidden_size/1.5)))
            self.linear_hidden_size = int(self.linear_hidden_size/1.5)
        
        self.fc = nn.Linear(self.linear_hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0)) 

        for linear_layer in self.linear_layers:
            out = linear_layer(out)
        
        out = self.fc(out[:, -1, :])
        return out
    
if 'sd_click' not in st.session_state:
    st.session_state.sd_click = False

if 'train_click' not in st.session_state:
    st.session_state.train_click = False

if 'disable_opt' not in st.session_state:
    st.session_state.disable_opt = False

if 'model_save' not in st.session_state:
    st.session_state.model_save = None

def split_sequences(sequences, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(sequences)):
      end_ix = i + n_steps_in
      out_end_ix = end_ix + n_steps_out

      if out_end_ix > len(sequences):
          break

      seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
      X.append(seq_x)
      y.append(seq_y)
  return torch.from_numpy(array(X)).float(), torch.from_numpy(array(y)).float()

def onClickSD():
    st.session_state.sd_click = True

def onClickTrain():
    st.session_state.train_click = True

def preProcessData(date_f,input_f,output_f):
    preProcessDataList=input_f
    preProcessDataList.insert(-1, output_f)
    preProcessDF = df[list(dict.fromkeys(preProcessDataList))]
    preProcessDF = preProcessDF.astype(float)

    preProcessDF = preProcessDF.replace(0, np.nan)
    preProcessDF = preProcessDF.interpolate(method='linear')
    preProcessDF = preProcessDF.bfill()
    
    preProcessDF.insert(0,date_f, df[date_f])
    if str(preProcessDF.at[0, date_f]).isdigit():
        preProcessDF[date_f] = pd.to_datetime(preProcessDF[date_f], format='%Y')
    else:
        preProcessDF[date_f] = pd.to_datetime(preProcessDF[date_f])
    return preProcessDF

def check_date_frequency(date_series):
    dates = pd.to_datetime(date_series)
    
    differences = (dates - dates.shift(1)).dropna()
    
    daily_count = (differences == timedelta(days=1)).sum()
    hourly_count = (differences == timedelta(hours=1)).sum()
    weekly_count = (differences == timedelta(weeks=1)).sum()
    monthly_count = (differences >= timedelta(days=28, hours=23, minutes=59)).sum()  # Approximate 28 days to a month
    
    if daily_count > max(monthly_count, hourly_count, weekly_count):
        return 365
    elif monthly_count > max(daily_count, hourly_count, weekly_count):
        return 12
    elif weekly_count > max(daily_count, hourly_count, monthly_count):
        return 52
    elif hourly_count > max(daily_count, weekly_count, monthly_count):
        return 24*365  # Assuming hourly data is daily data repeated every hour
    else:
        return 1

def sea_decomp(date_f,input_f,output_f):
    if date_f:
        sea_decomp_data = preProcessData(date_f,input_f,output_f)
        corr_df = sea_decomp_data.select_dtypes(include=['int', 'float'])
        correlation_matrix = np.round(corr_df.corr(), 1)
        result = seasonal_decompose(sea_decomp_data.set_index(date_f)[output_f], model='additive',period=check_date_frequency(sea_decomp_data[date_f]))
        
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=result.seasonal.index.values, y=result.seasonal.values, mode='lines',line=dict(color='orange')))
        fig_s.update_layout(title='Seasonal',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        height=300)
        
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=result.trend.index.values, y=result.trend.values, mode='lines',line=dict(color='orange')))
        fig_t.update_layout(title='Trend',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        height=300)
        
        fig_corr = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale='Viridis')
        
        with st.container(border=True):
            st.subheader("Correlation Matrix:")
            st.divider()
            st.plotly_chart(fig_corr,use_container_width=True)

        with st.container(border=True):
            st.subheader("Seasonal Decompose:")
            st.divider()
            st.plotly_chart(fig_t,use_container_width=True)
            st.divider()
            st.plotly_chart(fig_s,use_container_width=True)

        with st.container(border=True):
            st.subheader("Pre-Processed Data Preview:")
            st.divider()
            st.metric(label="Total Rows:", value=sea_decomp_data.shape[0])
            st.metric(label="Total Columns:", value=sea_decomp_data.shape[1])
            st.dataframe(sea_decomp_data,use_container_width=True,height=250)
        return sea_decomp_data

with col3:
    st.image("https://github.com/harshitv804/Time-Series-Forecasting-LSTM-Streamlit/assets/100853494/0d39dccf-42c7-4062-8000-3dc5646a0445")

with col4:
    st.title("Multi-Step Time Series Forecasting LSTM")
    st.subheader("Simple Streamlit GUI for LSTM Forecasting")

with col1:
    
    with st.container(border=True):
        st.subheader("CSV File Uploader:")
        st.divider()
        uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])

    if uploaded_file is not None:
        delimiters = (',', ';')
        df = pd.read_csv(uploaded_file, sep='[,;]',engine='python')

        
        with st.container(border=True):
            st.subheader("Feature Selection:")
            st.divider()
            date_f = st.selectbox(label="Select Date Feature:",options=df.columns)
            
            input_f = st.multiselect(label="Input Features (X):",options=[element for element in list(df.columns) if element != date_f])

            output_f = st.selectbox(label="Output Feature (Y):",options=[element for element in list(df.columns) if element != date_f])
            st.divider()
            st.button('Pre-Process Data',type="primary",on_click=onClickSD)


        with col2:
            
            with st.container(border=True):
                st.subheader("Dataset Preview:")
                st.divider()
                st.metric(label="Total Rows:", value=df.shape[0])
                st.metric(label="Total Columns:", value=df.shape[1])
                st.dataframe(df,use_container_width=True,height=250)

            if st.session_state.sd_click==True:
                data = sea_decomp(date_f,input_f,output_f)

        with st.container(border=True):
            st.subheader("Train LSTM Model:")
            st.divider()
            lag_steps = st.number_input('Lag Steps:',step=1,min_value=1)
            forecast_steps = st.number_input('Forecast Steps:',step=1,min_value=1)

            if (lag_steps+forecast_steps)>(df.shape[0]-forecast_steps):
                st.error(f'Lag Steps + Forecast Steps = {lag_steps+forecast_steps} should be <= {df.shape[0]-forecast_steps} (i.e Train set:({lag_steps+forecast_steps}) + Test set:({forecast_steps}) = {lag_steps+forecast_steps+forecast_steps} (>57))', icon="ℹ️")
                st.session_state.disable_opt = True
            else:
                st.session_state.disable_opt = False
            
            st.divider()

            lstm_layers = st.slider('LSTM Layers:', 1, 5)
            lstm_neurons = st.slider('LSTM Neurons:', 1, 500)
            st.divider()
            linear_hidden_layers = st.slider('Hidden Layers:', 1, 5)
            linear_hidden_neurons = st.slider('Hidden Neurons:', lstm_neurons,500)
            st.divider()
            n_epochs = st.number_input('No. of Epochs',step=1,min_value=1)
            batch_size = st.number_input('Batch Size',step=1,min_value=1,max_value=100)
            st.divider()
            
            sr = st.button('Train LSTM Model',type="primary",on_click=onClickTrain, disabled=st.session_state.disable_opt)
            st.error('Adjusting features or parameters after training will not maintain the session. Please ensure to retrain after making changes.', icon="ℹ️")
               
            if sr:
                with st.container(border=True):
                    df_train = data[:-forecast_steps]
                    df_test = data[-forecast_steps:]

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    datastack = np.empty((df_train.shape[0], 0))

                    for i in range(1,len(data.columns[1:])+1):
                        datastack = np.hstack((datastack, scaler.fit_transform(df_train.iloc[:,i].values.reshape((-1, 1)))))
                    datastack = np.hstack((datastack, scaler.fit_transform(df_train[output_f].values.reshape((-1, 1)))))

                    X, y = split_sequences(datastack, lag_steps, forecast_steps)
                    X = X.to(device)
                    y = y.to(device)
                    dataset = torch.utils.data.TensorDataset(X,y)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
                    model = LSTMForecasting(input_size=X.shape[2], lstm_hidden_size=lstm_neurons, lstm_num_layers=lstm_layers, linear_num_layers=linear_hidden_layers,linear_hidden_size=linear_hidden_neurons, output_size=forecast_steps).to(device)
                    st.write(model)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    
                    total_marks = n_epochs
                    progress_text = "Training LSTM. Please Wait..."
                    my_bar = st.progress(0, text=progress_text)

                    for epoch in range(1,n_epochs+1):
                        for inputs, labels in dataloader:
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                        my_bar.progress(int((epoch / total_marks) * 100), text=progress_text)
                    my_bar.empty()
                    st.success('Training Completed!', icon="✅")
                    st.balloons()

                    with torch.no_grad(): 
                        outputs = model(torch.Tensor(datastack[-forecast_steps:,:-1]).float().unsqueeze(0).to(device))
                        outputs_unscaled = scaler.inverse_transform(outputs[0].reshape(1, -1).cpu().numpy())
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                        st.session_state.model_save = [f'{loss.item():.4f}',outputs_unscaled[0], df_test[output_f],df_test[date_f]]
                        time.sleep(2)
                        
        if st.session_state.train_click==True:
            with st.status('Visualizing Predictions...',expanded=True):
                st.divider()
                st.metric(label="Training Loss:", value=st.session_state.model_save[0])
                st.metric(label="Testing RMSE:", value=int(np.sqrt(mean_squared_error(st.session_state.model_save[2], st.session_state.model_save[1]))))
                
                st.divider()
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=st.session_state.model_save[3], y=st.session_state.model_save[1], mode='lines+markers',name='Predicted',line=dict(color='orange', dash='dash'),marker=dict(size=10)))
                fig_pred.add_trace(go.Scatter(x=st.session_state.model_save[3], y=st.session_state.model_save[2], mode='lines+markers',name='Actual',marker=dict(size=10)))
                fig_pred.update_layout(
                                title='Testing: Actual vs Predicted',
                                xaxis_title='Date',
                                yaxis_title='Value',
                                height=410)
                st.plotly_chart(fig_pred,use_container_width=True)
                img_bytes = fig_pred.to_image(format="png",scale=2,engine="kaleido")
                st.download_button(
                    label="Download Forecast as IMG",
                    data=img_bytes,
                    file_name='forecast.png',
                    mime='image/jpeg')
    else:
        st.session_state.sd_click = False

ft = """
<style>
a:link , a:visited{
color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: #0283C3; /* theme's primary color*/
background-color: transparent;
text-decoration: underline;
}

#page-container {
  position: relative;
  min-height: 10vh;
}

footer{
    visibility:hidden;
}

.footer {
position: relative;
left: 0;
top:230px;
bottom: 0;
width: 100%;
background-color: transparent;
color: #808080; 
text-align: center;
padding: 0px 0px 15px 0px; 
}
</style>

<div id="page-container">

<div class="footer">
<p style='font-size: 0.875em;'>Made with <a style='display: inline; text-align: left;' href="https://streamlit.io/" target="_blank">Streamlit</a><br 'style= top:3px;'>
with <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><a style='display: inline; text-align: left;' href="https://github.com/harshitv804" target="_blank"> by Harshit V</a></p>
</div>

</div>
"""
st.write(ft, unsafe_allow_html=True)