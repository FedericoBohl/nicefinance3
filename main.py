import ssl

from libraries import *
from ARMA import *
from ARCH import *
if 'data' not in S:S.data=None
st.set_page_config(layout="wide")


def get_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None
if 'ticker' in S:
    S.ticker = (st.sidebar.text_input(r"$\text{Enter the ticker symbol (e.g., AAPL):}$", value=S.ticker)).replace(" ", "")
else:
    S.ticker = (st.sidebar.text_input(r"$\text{Enter the ticker symbol (e.g., AAPL):}$", value='')).replace(" ", "")
start_date = st.sidebar.date_input(r"$\text{Select the start date:}$", datetime(2023, 1, 1))
end_date = date.today()
if start_date > end_date:
    st.sidebar.error("Error: Start date cannot be after today's date.")
col1,space,col2=st.sidebar.columns((0.45,0.1,0.45))
with col1:
    if st.button(r"$\text{Search Data}$"):
        if S.ticker is not (None or '') :
            try:
                S.data = get_stock_data(S.ticker, start_date, end_date)
                S.duplicate=S.data
                S.make_title=True
            except: st.error('Error trying to get the data')
            if len(S.data)==0:st.error('Error trying to get the data');S.data=None
        else:st.error('No ticker selected')
with col2:
    if S.data is not None:
        if (len(S.data)!=0):
            st.download_button(r'$\text{Download}$',data=convert_df(S.data),file_name=f'{S.ticker.upper()}.csv',mime='text/csv')
if 'make_title' in S:
    try:st.title(f"{(yf.Ticker(S.ticker)).info['shortName']}")
    except:pass
if st.sidebar.checkbox(r'$\text{Show data}$') and S.data is not None:
    st.dataframe(S.data)

if S.data is not None:
    models={'ARIMA':arma,'ARCH':arch}
    page = st.sidebar.radio(r"$\text{Select a model}$", list(models.keys()))
    models[page]()

st.text('For recomendations, issues or questions, you can reach me at federicobohl@uca.edu.ar\nor in LinkedIn: Federico Ivan Bohl')