import streamlit as st
from streamlit import session_state as S
import yfinance as yf
from datetime import datetime, timedelta, date
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.stats.diagnostic import het_arch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import graph_objects as go
import numpy as np
from scipy import stats
import arch
from arch import unitroot
from arch.univariate import ARCH,APARCH,GARCH,EGARCH,HARCH
from sklearn.metrics import mean_squared_error
from arch import arch_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def plot_data(hide_split=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S.data.index, y=S.data['Close'], line=dict(color='lime')))
    if S.split != 1 and (hide_split is not True):
        fig.add_vline(S.data.index[round(len(S.data) * S.split, 0)], line_dash="dash", line_color="white")

    fig.update_layout(xaxis_title='Date', yaxis_title=r'Y')
    st.plotly_chart(fig, use_container_width=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
    fig.add_trace(
        go.Histogram(x=S.data['Close'], xaxis='x', yaxis='y', showlegend=False, marker=dict(color='lime')), row=1,
        col=1, secondary_y=False)
    fig.add_trace(go.Box(x=S.data['Close'], showlegend=False, name='', marker=dict(color='royalblue')), row=2,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Violin(x=S.data['Close'], showlegend=False, name='', marker=dict(color='coral')), row=2, col=1,
                  secondary_y=True)
    fig.update_layout(title='Distribution')
    st.plotly_chart(fig, use_container_width=True)
def normality():
    st.header(r'$\text{Nomality Analysys}$')
    shapiro_stat_normal, shapiro_p_normal = stats.shapiro(S.data['Close'])
    anderson_stat_normal, anderson_crit_vals_normal, anderson_sig_levels_normal = stats.anderson(S.data['Close'])

    fig = go.Figure()
    qq_normal = stats.probplot(S.data['Close'], dist="norm", plot=None)
    fig.add_trace(
        go.Scatter(x=qq_normal[0][0], y=qq_normal[0][1], mode='markers', name='Normal Data', showlegend=False,
                   line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=qq_normal[0][0], y=qq_normal[1][1] + qq_normal[1][0] * qq_normal[0][0],
                             line=dict(color='rgb(220, 20, 60)'), showlegend=False))
    fig.update_layout(title="QQ Plot for Normality Assessment",
                      xaxis_title="Theoretical Quantiles",
                      yaxis_title="Ordered Values")
    st.plotly_chart(fig,use_container_width=True)
    col1, space, col2 = st.columns((0.4, 0.1, 0.5))
    a1=r'\text{Test Statistic: }'
    a2 = r'\text{p-value: }'
    a3 = r'\text{Is normally distributed: }'
    with col1:
        st.subheader(r":violet[$\text{Shapiro-Wilk Test}$]")
        st.write(f"${a1}{round(shapiro_stat_normal, 2)}$")
        st.write(f"${a2}{round(shapiro_p_normal, 4)}$")
        st.write(f"${a3}{shapiro_p_normal > 0.05}$")
    with col2:
        st.subheader(r":violet[$\text{Anderson-Darling Test}$]")
        st.write(f"${a1}{round(anderson_stat_normal, 2)}$")
        st.write(f"${a2}{round(anderson_crit_vals_normal[2], 2)}$")
        st.write(f"${a3}{anderson_stat_normal < anderson_crit_vals_normal[2]}$")
    col1, space, col2 = st.columns((0.4, 0.1, 0.5))
    with col1:
        agostino_stat_normal, agostino_p_normal = stats.normaltest(S.data['Close'])
        st.subheader(r":violet[$\text{Agostino-Pearson Test}$]")
        st.write(f"${a1}{round(agostino_stat_normal, 2)}$")
        st.write(f"${a2}{round(agostino_p_normal, 4)}$")
        st.write(f"${a3}{agostino_p_normal > 0.05}$")
    with col2:
        skewness = stats.skew(S.data['Close'])
        kurtosis = stats.kurtosis(S.data['Close'])
        # Perform the Jarque-Bera test
        jarque_bera_statistic = (skewness ** 2 + (kurtosis - 3) ** 2 / 4) * len(S.data['Close']) / 6
        p_value = 1 - stats.chi2.cdf(jarque_bera_statistic,
                                     df=2)  # Chi-squared distribution with 2 degrees of freedom
        st.subheader(r":violet[$\text{Jarque-Bera Test}$]")
        st.write(f"${a1}{round(jarque_bera_statistic, 2)}$")
        st.write(f"${a2}{round(p_value, 4)}$")
        st.write(f"${a3}{p_value > 0.05}$")
def heteroskedasticity():
    st.header(r'$\text{Heteroskedasticity Analysys}$')
    np.random.seed(0)
    n = 100
    time = np.arange(n)
    y = time + np.random.normal(0, time, n)
    model = sm.OLS(S.data['Close'], sm.add_constant(np.arange(len(S.data['Close'])))).fit()
    residuals = model.resid

    # Gráfico de Residuos vs. Tiempo
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=S.data.index,y=residuals,mode='markers',marker=dict(color='royalblue',size=10)))
    fig.add_hline(y=0,line_dash="longdash",line=dict(color="red", width=3))
    fig.update_layout(xaxis_title='Date',yaxis_title='Residual')
    st.plotly_chart(fig,use_container_width=True)

    # Prueba ARCH para heterocedasticidad en series temporales
    arch_test = het_arch(residuals)
    st.subheader(r":violet[$\text{ARCH Test}$]")
    a=r'\text{Test\ Statistic}'
    st.write(f"${a}:\ {round(arch_test[0],2)}$")
    a = r'\text{p-value}'
    st.write(f"${a}:\ {round(arch_test[1],4)}$")
    a=r'\text{Presence\ of\ Heteroskedasticity}'
    st.write(f'${a}:\ {arch_test[1]<=0.05}$')
    del a
def random_walk():
    st.header(r'$\text{Unit Root Analysys}$')
    adf=unitroot.ADF(S.data['Close'])
    dfgls=unitroot.DFGLS(S.data['Close'])
    phillips=unitroot.PhillipsPerron(S.data['Close'])
    kpss=unitroot.KPSS(S.data['Close'])
    zivot=unitroot.ZivotAndrews(S.data['Close'])
    test_results = {
        'ADF': {
            'Test Statistic': round(adf.stat, 2),
            'p-value': round(adf.pvalue, 4),
            'lags': adf.lags,
            'trend': adf.trend
        },
        'DFGLS': {
            'Test Statistic': round(dfgls.stat, 2),
            'p-value': round(dfgls.pvalue, 4),
            'lags': dfgls.lags,
            'trend': dfgls.trend
        },
        'Phillips-Perron': {
            'Test Statistic': round(phillips.stat, 2),
            'p-value': round(phillips.pvalue, 4),
            'lags': phillips.lags,
            'trend': phillips.trend
        },
        'KPSS': {
            'Test Statistic': round(kpss.stat, 2),
            'p-value': round(kpss.pvalue, 4),
            'lags': kpss.lags,
            'trend': kpss.trend
        },
        'Zivot-Andrews': {
            'Test Statistic': round(zivot.stat, 2),
            'p-value': round(zivot.pvalue, 4),
            'lags': zivot.lags,
            'trend': zivot.trend
        }
    }
    df=pd.DataFrame(test_results)
    def test_result(p_value):
        if p_value <= 0.05:
            return "Not a random walk"
        else:
            return "Random walk"

    df.loc['Test result'] = df.loc['p-value'].apply(test_result)
    # Create the DataFrame from the dictionary
    st.dataframe(df)
    st.write('The "c" in trend means "Constant"')
def stock_info():
    tick=yf.Ticker(S.ticker)
    select,ticker_info=st.columns((0.25,0.75))
    with select:S.info=st.radio(r'$\text{Select the info:}$',['Basic Info','News','Dividens, and Splits','Income','Balance Sheet','Cashflow','Holders'])
    with ticker_info:
        if S.info=='Basic Info':
            basic_info={}
            for i in ['city','state','country','webstite','industry','sector','beta','open','bid','ask','dividendRate','dividendYield']:
                try:
                    basic_info[f'{i.capitalize()}']=tick.info[f'{i}']
                except:pass
            try:
                st.header('Company Officers')
                st.dataframe(pd.DataFrame(tick.info['companyOfficers']).drop(columns=['maxAge','exercisedValue','unexercisedValue','age','yearBorn'],axis=1))
            except:pass
            st.header('Information')
            st.dataframe(basic_info,use_container_width=True)
            del basic_info
        elif S.info=='News':
            news_data=pd.DataFrame(tick.news)
            if len(news_data)!=0:
                news={
                    'Title':news_data['title'],
                    'Publisher': news_data['publisher'],
                    'Related Tickers': news_data['relatedTickers'],
                    'Link': news_data['link']

                }
                st.dataframe(news,use_container_width=True)
                del news
            else:st.error('No news avaliable :(')
            del news_data
        elif S.info=='Dividens, and Splits':
            st.dataframe(tick.actions,use_container_width=True)
            if len (tick.actions)!=0:st.download_button(r'$\text{Download}$',data=convert_df(pd.DataFrame(tick.actions)),file_name=f'Dividends_Splits-{S.ticker}.csv',use_container_width=True)
        elif S.info=='Income':
            st.dataframe(tick.income_stmt,use_container_width=True)
            if len (tick.income_stmt)!=0:st.download_button(r'$\text{Download}$',data=convert_df(pd.DataFrame(tick.income_stmt)),file_name=f'Income-{S.ticker}.csv',use_container_width=True)
        elif S.info=='Balance Sheet':
            if st.radio('',label_visibility='hidden',options=['Anual','Quarterly'],horizontal=True)=='Anual':
                st.dataframe(tick.balance_sheet,use_container_width=True)
                if len(tick.balance_sheet) != 0: st.download_button(r'$\text{Download}$',
                                                              data=convert_df(pd.DataFrame(tick.balance_sheet)),
                                                              file_name=f'Balance_Sheet-{S.ticker}.csv',
                                                              use_container_width=True)
            else:
                st.dataframe(tick.quarterly_balance_sheet,use_container_width=True)
                if len(tick.quarterly_balance_sheet) != 0: st.download_button(r'$\text{Download}$',
                                                              data=convert_df(pd.DataFrame(tick.quarterly_balance_sheet)),
                                                              file_name=f'Quarterly_Balance_Sheet-{S.ticker}.csv',
                                                              use_container_width=True)
        elif S.info=='Cashflow':
            if st.radio('',label_visibility='hidden',options=['Anual','Quarterly'],horizontal=True)=='Anual':
                st.dataframe(tick.cashflow,use_container_width=True)
                if len(tick.cashflow) != 0: st.download_button(r'$\text{Download}$',
                                                              data=convert_df(pd.DataFrame(tick.cashflow)),
                                                              file_name=f'Cashflow-{S.ticker}.csv',
                                                              use_container_width=True)
            else:
                st.dataframe(tick.quarterly_cashflow,use_container_width=True)
                if len(tick.quarterly_cashflow) != 0: st.download_button(r'$\text{Download}$',
                                                              data=convert_df(pd.DataFrame(tick.quarterly_cashflow)),
                                                              file_name=f'Quarterly_Cashflow-{S.ticker}.csv',
                                                              use_container_width=True)
        else:
            st.dataframe(tick.major_holders,use_container_width=True)
            st.divider()
            if st.radio('',label_visibility='hidden',options=['Institutional Holders','Mutual Holders'],horizontal=True)=='Institutional Holders':
                st.dataframe(tick.institutional_holders,use_container_width=True)
                if len(tick.institutional_holders) != 0: st.download_button(r'$\text{Download}$',
                                                              data=convert_df(pd.DataFrame(tick.institutional_holders)),
                                                              file_name=f'Institutional_Holders-{S.ticker}.csv',
                                                              use_container_width=True)
            else:
                st.dataframe(tick.mutualfund_holders,use_container_width=True)
                try:
                    if len(tick.mutualfund_holders) != 0: st.download_button(r'$\text{Download}$',
                                                              data=convert_df(pd.DataFrame(tick.mutualfund_holders)),
                                                              file_name=f'Mutual_Holders-{S.ticker}.csv',
                                                              use_container_width=True)
                except:pass