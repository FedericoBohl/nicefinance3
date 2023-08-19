from libraries import *

def arch():
    def set_parameters():
       st.sidebar.latex(r'\text{Mean of the model}')
       S.mean=st.sidebar.selectbox('Mean of the model',label_visibility='hidden',options=['Constant','Zero','LS','AR','ARX','HAR','HARX'])
       st.sidebar.latex(r'\text{Volatility of the model}')
       S.vol = st.sidebar.selectbox('vol of the model', label_visibility='hidden',options=['GARCH', 'ARCH', 'EGARCH', 'APARCH', 'HARCH'])
    def make_model():
       if S.vol=='GARCH':
           model=arch_model(S.data['Close'],mean=S.mean, vol='GARCH',lags=S.lag, p=S.p, q=S.q, o=S.o, power=S.power)
       elif S.vol=='EGARCH':
           model=arch_model(S.data['Close'],mean=S.mean, vol='EGARCH',lags=S.lag, p=S.p, q=S.q, o=S.o)
       elif S.vol=='HARCH':
           model=arch_model(S.data['Close'],mean=S.mean, vol='HARCH', lags=S.l)
       elif S.vol=='ARCH':
           model=arch_model(S.data['Close'],mean=S.mean, vol='ARCH',lags=S.lag, p=S.p)
       else:
           model=arch_model(S.data['Close'],mean=S.mean,vol=S.vol, lags=S.lag, p=S.p, q=S.q, o=S.o)
       fit=model.fit()
       S.fit=fit

    st.sidebar.divider()
    col1,col2=st.sidebar.columns((0.45,0.55))
    with col1:st.header('*ARCH Models*')
    with col2:
       S.diff=st.checkbox(r'$\text{Differentiate}$', value=False)
       if S.diff:
           S.data = (S.duplicate - S.duplicate.shift(1)).dropna()
       else: S.data=S.duplicate
    set_parameters()
    info,tab1, tab2, tab3, tab4, tab5 = st.tabs([r'$\text{Ticker Info}$',r'$\text{Model}$',r'$\text{Plots}$',r'$\text{Normality Analysis}$',r'$\text{Unit root}$',r'$\text{Heteroskedasticity}$'])
    with info:stock_info()
    with tab1:
        equation = {
            'GARCH': r'\sigma_{t}^{\lambda}=\omega+ \sum_{i=1}^{p}\alpha_{i}\left|\epsilon_{t-i}\right|^{\lambda}+\sum_{j=1}^{o}\gamma_{j}\left|\epsilon_{t-j}\right|^{\lambda}I\left[\epsilon_{t-j}<0\right]+\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{\lambda}'
            , 'ARCH': r'\sigma_{t}^{2}=\omega+ \sum_{i=1}^{p}\alpha_{i}\left|\epsilon_{t-i}\right|'
            ,
            'EGARCH': r'\ln\sigma_{t}^{2}=\omega+\sum_{i=1}^{p}\alpha_{i}\left(\left|e_{t-i}\right|-\sqrt{2/\pi}\right)+\sum_{j=1}^{o}\gamma_{j} e_{t-j}+\sum_{k=1}^{q}\beta_{k}\ln\sigma_{t-k}^{2}'
            ,
            'HARCH': r'\sigma_{t}^{2}=\omega + \sum_{i=1}^{m}\alpha_{l_{i}}\left(l_{i}^{-1}\sum_{j=1}^{l_{i}}\epsilon_{t-j}^{2}\right)'
            ,
            'APARCH': r'\sigma_{t}^{\delta}=\omega+\sum_{i=1}^{p}\alpha_{i}\left(\left|\epsilon_{t-i}\right|-\gamma_{i}I_{[o\geq i]}\epsilon_{t-i}\right)^{\delta}+\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{\delta}'}
        st.header(r':green[$\text{Volatility model selected:}$]')
        st.latex(equation[S.vol])
        if S.mean == 'Zero': st.latex(r'\text{Where}\ \omega=0\text{.}')
        if S.vol == 'GARCH': st.latex(r'\text{Where}\ \lambda\ \text{is the power.}')
        with st.expander(r"$\text{See }$:blue[$\text{Autocorrelation Function (}ACF\text{)}$]$\text{ and }$:red[$\text{Partial Autocorrelation Function (}PACF\text{)}$]$\text{ graphs of the residuals of an }AR(1)\text{ process:}$"):
            model_res = ARIMA(S.data['Close'], order=(1, 0, 0))
            results_res = model_res.fit()
            acf_x,confint_acf=acf((results_res.resid), alpha=0.05)
            pacf_x,confint_pacf = pacf(results_res.resid,alpha=0.05)
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
        with st.expander(r":green[$\text{Set }p,q,o,\lambda,l\ \text{and}\ \delta\text{ values:}$]"):
            def p():
                st.latex(r'{p}')
                S.p = st.slider('p', label_visibility='hidden', min_value=0, max_value=10, value=1)
            def q():
                st.latex(r'{q}')
                S.q = st.slider('q', label_visibility='hidden', min_value=0, max_value=10, value=1)
            def o():
                st.latex(r'{o}')
                S.o = st.slider('o', label_visibility='hidden', min_value=0, max_value=10, value=1)
            def power():
                st.latex(r'\lambda')
                S.power = st.slider('power', label_visibility='hidden', min_value=1, max_value=10, value=2)
            if S.mean in ['AR', 'ARX', 'HAR', 'HARX'] and S.vol != 'HARCH':
                st.latex(r'\text{Lag of the autoregresive part of the mean}:')
                S.lag = st.slider('lag', label_visibility='hidden', min_value=1, max_value=10)
            else:
                S.lag = 0
            if S.vol == 'ARCH':
                p()
            elif S.vol == 'HARCH':
                st.latex(r'{l}\ \text{(Lag)}')
                S.l = st.slider('l', label_visibility='hidden', min_value=1, max_value=50)
            col1, space, col2 = st.columns((0.45, 0.1, 0.45))
            with col1:
                if S.vol in ['GARCH', 'EGARCH', 'APARCH']: p()
                if S.vol in ['GARCH', 'EGARCH', 'APARCH']: o()
            with col2:
                if S.vol in ['GARCH', 'EGARCH', 'APARCH']: q()
                if S.vol == 'GARCH':
                    power()
                elif S.vol == 'APARCH':
                    st.latex(r'\delta')
                    S.delta = st.slider('delta', label_visibility='hidden', min_value=1, max_value=10, value=1)
                    if S.delta == 0: S.delta = None
        if st.button(r'$\text{Create model}$',use_container_width=True):
           make_model()
        if 'fit' in S:st.write(S.fit.summary())
    with tab2: plot_data(hide_split=True)
    with tab3: normality()
    with tab4: random_walk()
    with tab5: heteroskedasticity()


'''   split_index = int(len(S.data) * S.split)
   train_data = S.data.iloc[:split_index]
   test_data = S.data.iloc[split_index:]

   model = arch_model(train_data['Close'], vol='GARCH', p=1, q=1)
   results = model.fit()

   # Predict volatility and calculate predictions
   volatility_predictions = results.conditional_volatility
   test_predictions = volatility_predictions[split_index:]
   st.write(len(volatility_predictions))
   # Calculate MSE for training and testing sets
   train_mse = mean_squared_error(train_data['Close'], volatility_predictions)
   test_mse = mean_squared_error(test_data['Close'], test_predictions)

   # Print MSE
   print("Training MSE:", train_mse)
   print("Testing MSE:", test_mse)

   # Plot predictions and actual returns
   plt.figure(figsize=(10, 6))
   plt.plot(train_data.index, train_data['Return'], label='Actual Returns (Train)')
   plt.plot(test_data.index, test_data['Return'], label='Actual Returns (Test)')
   plt.plot(train_data.index, volatility_predictions, label='Volatility Predictions (Train)')
   plt.plot(test_data.index, test_predictions, label='Volatility Predictions (Test)')
   plt.legend()
   plt.xlabel('Date')
   plt.ylabel('Returns / Volatility')
   plt.title('GARCH Volatility Predictions')
   plt.show()'''




