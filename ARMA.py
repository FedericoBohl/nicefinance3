import arch
import numpy as np
import pandas as pd
import streamlit

from libraries import *

def arma():
    def plot_acf():
        with st.expander(r"$\text{See }$:blue[$\text{Autocorrelation Function (}ACF\text{)}$]$\text{ and }$:red[$\text{Partial Autocorrelation Function (}PACF\text{)}$]$\text{ graphs:}$"):
            acf_x,confint_acf=acf((S.data['Close']), alpha=0.05)
            pacf_x,confint_pacf = pacf(S.data['Close'],alpha=0.05)
            lag = np.array(range(len(acf_x))).astype(float)
            acf_df = pd.DataFrame({'Lag': lag, 'Autocorrelation': acf_x})
            pacf_df = pd.DataFrame({'Lag': lag, 'Partial Autocorrelation': pacf_x})

            # ACF plot
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=lag[:-1], y=acf_df['Autocorrelation'][:-1], name='Autocorrelation',mode='markers',showlegend=False))
            fig.update_traces(marker_color='royalblue',marker_size=9)
            for i in range(len((acf_x))-1):
                fig.add_shape(
                    type="line",
                    x0=i,  # x-coordinate where the line starts
                    x1=i,  # x-coordinate where the line ends (in this case, same as x0 for a vertical line)
                    y0=0,  # y-coordinate where the line starts
                    y1=acf_x[i],  # y-coordinate where the line ends
                    line=dict(color="royalblue", width=2)  # Customize the line color and width
                )
            lag[np.argmin(lag)] += 0.5
            lag[np.argmax(lag)] -= 0.5
            fig.add_trace(go.Scatter(
                x=lag[:-1],
                y=confint_acf[1:,1]-acf_x[1:],
                mode='lines',
                line=dict(color='royalblue', width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=lag[:-1],
                y=confint_acf[1:,0]-acf_x[1:],
                mode='lines',
                line=dict(color='red', width=0),
                fill='tonexty',  # Rellena el área entre las curvas
                fillcolor=f'rgba(65, 105, 225, {0.2})',  # Color verde con transparencia
                name='Confidence Interval',showlegend=False
            ))
            # Ajusta el ancho de la línea del intervalo de confianza
            fig.update_traces(selector=dict(name='Confidence Interval'), line=dict(width=0))
            fig.update_layout(title='Autocorrelation Function (ACF)',
                              xaxis_title='Lag', yaxis_title='Correlation')
            st.plotly_chart(fig,use_container_width=True)

            fig = go.Figure()
            lag = np.array(range(len(pacf_x))).astype(float)
            fig.add_trace(go.Scatter(x=lag[:-1], y=pacf_df['Partial Autocorrelation'][:-1], name='Autocorrelation', mode='markers',showlegend=False))
            fig.update_traces(marker_color='rgb(220, 20, 60)', marker_size=9)
            for i in range(len((pacf_x))-1):
                fig.add_shape(
                    type="line",
                    x0=i,  # x-coordinate where the line starts
                    x1=i,  # x-coordinate where the line ends (in this case, same as x0 for a vertical line)
                    y0=0,  # y-coordinate where the line starts
                    y1=pacf_x[i],  # y-coordinate where the line ends
                    line=dict(color="rgb(220, 20, 60)", width=2)  # Customize the line color and width
                )
            lag[np.argmin(lag)] += 0.5
            lag[np.argmax(lag)] -= 0.5
            fig.add_trace(go.Scatter(
                x=lag[:-1],
                y=confint_pacf[1:, 1] - pacf_x[1:],
                mode='lines',
                line=dict(color='rgb(220, 20, 60)', width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=lag[:-1],
                y=confint_pacf[1:, 0] - pacf_x[1:],
                mode='lines',
                line=dict(color='red', width=0),
                fill='tonexty',  # Rellena el área entre las curvas
                fillcolor=f'rgba(220, 20, 60, {0.2})',  # Color verde con transparencia
                name='Confidence Interval',showlegend=False
            ))
            # Ajusta el ancho de la línea del intervalo de confianza
            fig.update_traces(selector=dict(name='Confidence Interval'), line=dict(width=0))
            fig.update_layout(title='Partial Autocorrelation Function (PACF)',
                              xaxis_title='Lag', yaxis_title='Correlation')
            st.plotly_chart(fig, use_container_width=True)
    def set_arma_parameters():
        st.sidebar.latex(r'AR(p)')
        S.p=st.sidebar.slider(label='p',label_visibility='hidden',min_value=0,max_value=10,value=1)
        st.sidebar.latex(r'MA(q)')
        S.q = st.sidebar.slider(label='q',label_visibility='hidden',min_value= 0,max_value= 10, value=1)
        st.sidebar.latex(r'I(d)')
        S.d=st.sidebar.slider(label='d',label_visibility='hidden',min_value= 0,max_value= 5, value=0)
        if S.d!=0:
            S.data = (S.duplicate - S.duplicate.shift(S.d)).dropna()
        else:
            S.data = S.duplicate
    def arma_model():
        if st.button(r'$\text{Create model}$', use_container_width=True):
            split_index = int(len(S.data) * S.split)
            S.train_data = S.data.iloc[:split_index]
            S.test_data = S.data.iloc[split_index:]
            model = ARIMA(S.train_data['Close'], order=(S.p,S.d,S.q))
            S.results = model.fit()
            S.model=True
        if 'model' in S:
            results=S.results
            train_data=S.train_data
            test_data=S.test_data
            results.predict()
            train_predictions = results.predict(start=S.p,end=len(train_data)-1)
            test_predictions = results.predict(start=len(train_data), end=len(S.data)-1)
            train_rmse = np.sqrt(mean_squared_error(train_data['Close'][S.p:], train_predictions))
            test_rmse = np.sqrt(mean_squared_error(test_data['Close'], test_predictions))

            loglike, sumary = st.columns((0.3, 0.7))
            with loglike:
                st.subheader(r'$\text{Likelihood}$')
                st.dataframe({
                    '': {'BIC': round(results.bic, 4),
                         'AIC': round(results.aic, 4),
                         'HQIC': round(results.hqic, 4)
                         }
                })
            with sumary:
                st.subheader(r'$\text{Summary}$')
                pvals = np.array(results.pvalues)[:-1]
                z = np.array(results.zvalues)[:-1]
                coef = np.concatenate((np.array(results.arparams), np.array(results.maparams)), axis=0)

                names = []
                for i in range(S.p):
                    names.append(f'AR->{i + 1}')
                for i in range(S.q):
                    names.append(f'MA->{i + 1}')

                df = pd.DataFrame(zip(coef, z, pvals), columns=['Coefficient', 'Statistic', 'P-Value'], index=names)
                st.dataframe(df, use_container_width=True)

            col1,space,col2=st.columns((0.45,0.1,0.45))
            with col1:
                a=r':blue[$\text{RMSE Train:}\quad$'
                st.subheader(f'{a}${round(train_rmse,4)}$]')
                st.download_button(r'$\text{Download train predictions}$', data=convert_df(train_predictions), file_name=f'Train Predictions.csv', mime='text/csv')
            with col2:
                a=r':red[$\text{RMSE Test:}\quad$:'
                st.subheader(f'{a}${round(test_rmse,4)}$]')
                st.download_button(r'$\text{Download test predictions}$', data=convert_df(test_predictions), file_name=f'Test Predictions.csv',
                                   mime='text/csv')
            st.divider()
            date_train, date_test = S.data.index[len(S.data.index) - len(test_predictions) - len(train_predictions):len(S.data.index) - len(
                test_predictions)], S.data.index[len(S.data.index) - len(test_predictions):]
            date = np.concatenate((date_train, date_test))
            test_nan = np.empty((len(date_test)))
            test_nan[:] = np.nan
            train = np.concatenate((train_predictions, test_nan))

            train_nan = np.empty((len(date_train)))
            train_nan[:] = np.nan
            test = np.concatenate((train_nan, test_predictions))

            price = np.concatenate((train_data, test_data))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=date, y=S.data['Close'], line=dict(color='lime'), name='Y'))
            fig.add_trace(go.Scatter(x=date, y=train, line=dict(color='royalblue'), name='Training'))
            fig.add_trace(go.Scatter(x=date, y=test, line=dict(color='rgb(220, 20, 60)', dash='dash'), name='Testing'))
            if S.split != 1:
                fig.add_vline(S.data.index[int(round(len(S.data) * S.split - 1, 0))], line_dash="dash", line_color="white")
                #st.write(S.data.index)
                #st.write(len(S.data))
                #st.write(S.split)
                #st.write()
                #fig.add_vline(100, line_dash="dash", line_color="white")
            fig.update_layout(xaxis_title='Date', yaxis_title=r'Y')
            st.plotly_chart(fig, use_container_width=True)



    st.sidebar.divider()
    col1,space,col2 = streamlit.sidebar.columns((0.45,0.1,0.45))
    with col1:
        st.header('*ARIMA Model*')
    with col2: S.split=(st.slider(r'$\text{Split Train-Test}$',min_value=0,max_value=100,value=70))/100
    set_arma_parameters()
    info,tab1,tab2,tab3,tab4,tab5=st.tabs([r'$\text{Ticker Info}$',r'$\text{Model}$',r'$\text{Plots}$',r'$\text{Normality Analysis}$',r'$\text{Unit root}$',r'$\text{Heteroskedasticity}$'])
    with info:stock_info()
    with tab1:arma_model()
    with tab2:
        plot_acf()
        plot_data()
    with tab3:
        normality()
    with tab4:
        random_walk()
    with tab5:
        heteroskedasticity()
