from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RetailPulse | Sales Intelligence", page_icon="📈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f5f7fb}.block-container{padding-top:1.6rem;max-width:1450px}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:#162033}
[data-testid="stSidebar"]{background:#111827}[data-testid="stSidebar"] *{color:#e5e7eb!important}
.hero{background:linear-gradient(120deg,#111827,#1e3a5f 60%,#155e75);padding:28px 32px;border-radius:22px;color:white;margin-bottom:22px}
.hero h1{color:white;margin:0;font-size:2.1rem}.hero p{color:#cbd5e1;margin:8px 0 0}
.eyebrow{color:#67e8f9;text-transform:uppercase;letter-spacing:.15em;font-size:.75rem;font-weight:700;margin-bottom:9px}
.kpi{background:white;border:1px solid #e6eaf1;border-radius:17px;padding:18px 20px;min-height:112px;box-shadow:0 5px 20px #0f172a08}
.kpi-label{color:#64748b;font-size:.82rem;font-weight:600}.kpi-value{color:#14213d;font-size:1.65rem;font-weight:700;margin-top:8px;font-family:'Space Grotesk',sans-serif}
.kpi-note{color:#0f766e;font-size:.78rem;margin-top:3px}
div[data-testid="stPlotlyChart"]{background:white;border:1px solid #e6eaf1;border-radius:16px;padding:8px}
.stButton button,.stDownloadButton button{border-radius:10px;font-weight:700}footer{visibility:hidden}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    d = pd.read_csv(ROOT/"data"/"retail_sales.csv", parse_dates=["date"])
    d["sales"] = pd.to_numeric(d.sales, errors="coerce")
    return d.dropna(subset=["date","store","category","sales"]).query("sales >= 0").sort_values("date")
data = load_data()
features = ["dow","month","dayofyear","dayofmonth","year","trend","lag_1","lag_7","lag_14","lag_28","roll7","roll28"]

def make_features(g):
    g=g.copy().sort_values("date")
    g["dow"]=g.date.dt.dayofweek; g["month"]=g.date.dt.month
    g["dayofyear"]=g.date.dt.dayofyear; g["dayofmonth"]=g.date.dt.day
    g["year"]=g.date.dt.year; g["trend"]=(g.date-g.date.min()).dt.days
    for lag in (1,7,14,28): g[f"lag_{lag}"]=g.sales.shift(lag)
    g["roll7"]=g.sales.shift(1).rolling(7).mean()
    g["roll28"]=g.sales.shift(1).rolling(28).mean()
    return g

@st.cache_data(show_spinner=False)
def run_forecast(frame, store, category, horizon):
    g=frame[(frame.store==store)&(frame.category==category)].sort_values("date").copy().reset_index(drop=True)
    f=make_features(g).dropna().reset_index(drop=True)
    split=int(len(f)*.8); train,test=f.iloc[:split],f.iloc[split:]
    model=RandomForestRegressor(n_estimators=140,min_samples_leaf=2,random_state=42,n_jobs=-1)
    model.fit(train[features],train.sales)
    pred=np.maximum(model.predict(test[features]),0)
    mae=mean_absolute_error(test.sales,pred); rmse=np.sqrt(mean_squared_error(test.sales,pred))
    wape=np.abs(test.sales.to_numpy()-pred).sum()/test.sales.sum()*100
    hold=test[["date","sales"]].copy(); hold["predicted_sales"]=pred
    values=g.sales.astype(float).tolist(); first,last=g.date.min(),g.date.max(); future=[]
    for step in range(1,int(horizon)+1):
        d=last+pd.Timedelta(days=step)
        v={"dow":d.dayofweek,"month":d.month,"dayofyear":d.dayofyear,"dayofmonth":d.day,"year":d.year,
           "trend":(d-first).days,"lag_1":values[-1],"lag_7":values[-7],"lag_14":values[-14],"lag_28":values[-28],
           "roll7":float(np.mean(values[-7:])),"roll28":float(np.mean(values[-28:]))}
        estimate=max(0.,float(model.predict(pd.DataFrame([[v[x] for x in features]],columns=features))[0]))
        values.append(estimate); future.append({"date":d,"forecast_sales":round(estimate,2)})
    return hold,pd.DataFrame(future),{"MAE":mae,"RMSE":rmse,"WAPE":wape}

with st.sidebar:
    st.markdown("## 📈 RetailPulse")
    st.caption("SALES INTELLIGENCE"); st.divider()
    store=st.selectbox("Store",sorted(data.store.unique()))
    category=st.selectbox("Product category",sorted(data.category.unique()))
    horizon=st.slider("Forecast horizon",7,90,30)
    page=st.radio("Workspace",["Overview","Forecast lab","Data explorer"])
    st.divider(); st.caption("Student project • Synthetic demo data")
st.markdown('<div class="hero"><div class="eyebrow">Retail analytics platform</div><h1>Know your sales. Plan your next move.</h1><p>Explore performance, forecast demand, and turn daily sales data into clearer decisions.</p></div>',unsafe_allow_html=True)

if page=="Overview":
    total=data.sales.sum(); avg=data.groupby("date").sales.sum().mean(); top=data.groupby("category").sales.sum().idxmax()
    st.subheader("Business overview")
    st.caption(f"All stores and categories · {data.date.min():%d %b %Y} — {data.date.max():%d %b %Y}")
    cols=st.columns(4)
    for col,label,value,note in zip(cols,["TOTAL SALES","AVG. DAILY SALES","TOP CATEGORY","HISTORY COVERED"],
        [f"{total:,.0f}",f"{avg:,.0f}",top,f"{data.date.nunique():,} days"],
        ["Across all series","Combined daily total","Highest total sales","Daily demonstration data"]):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)
    daily=data.groupby("date",as_index=False).sales.sum()
    monthly=data.assign(month=data.date.dt.to_period("M").dt.to_timestamp()).groupby("month",as_index=False).sales.sum()
    a,b=st.columns([1.5,1])
    with a:
        st.markdown("### Sales trend")
        fig=px.line(daily,x="date",y="sales",template="plotly_white",labels={"date":"Date","sales":"Sales"})
        fig.update_traces(line=dict(color="#0f766e",width=2)); fig.update_layout(height=330,margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with b:
        st.markdown("### Category mix")
        cat=data.groupby("category",as_index=False).sales.sum()
        fig=px.pie(cat,names="category",values="sales",hole=.62,template="plotly_white")
        fig.update_layout(height=330,margin=dict(l=5,r=5,t=10,b=10))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Monthly sales")
    fig=px.bar(monthly,x="month",y="sales",template="plotly_white",labels={"month":"Month","sales":"Sales"})
    fig.update_traces(marker_color="#2563eb"); fig.update_layout(height=300,margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.info("This website ships with synthetic demonstration data. Replace it with real historical sales before making business decisions.")

elif page=="Forecast lab":
    st.subheader("Forecast lab"); st.caption(f"{store} / {category} · Forecasting next {horizon} days")
    with st.spinner("Training model and preparing forecast..."):
        try:
            hold,future,scores=run_forecast(data,store,category,horizon)
            k=st.columns(3)
            for col,label,val,note in zip(k,["MAE","RMSE","WAPE"],[f'{scores["MAE"]:,.2f}',f'{scores["RMSE"]:,.2f}',f'{scores["WAPE"]:.2f}%'],["Mean absolute error","Root mean squared error","Weighted absolute percentage error"]):
                col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)
            st.markdown("### Future sales forecast")
            hist=data[(data.store==store)&(data.category==category)].tail(120)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=hist.date,y=hist.sales,name="Historical",mode="lines",line=dict(color="#0f766e",width=2)))
            fig.add_trace(go.Scatter(x=future.date,y=future.forecast_sales,name="Forecast",mode="lines+markers",line=dict(color="#2563eb",width=3,dash="dash")))
            fig.update_layout(template="plotly_white",height=390,xaxis_title="Date",yaxis_title="Sales",margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig,use_container_width=True)
            x,y=st.columns(2); x.metric("Forecast total",f'{future.forecast_sales.sum():,.2f}'); y.metric("Forecast daily average",f'{future.forecast_sales.mean():,.2f}')
            st.dataframe(future,use_container_width=True,hide_index=True)
            st.download_button("Download forecast CSV",future.to_csv(index=False).encode(),file_name=f"forecast_{store}_{category}.csv",mime="text/csv",type="primary")
            st.markdown("### Holdout test: actual vs predicted")
            test=hold.tail(120); fig=go.Figure()
            fig.add_trace(go.Scatter(x=test.date,y=test.sales,name="Actual",mode="lines",line=dict(color="#0f766e")))
            fig.add_trace(go.Scatter(x=test.date,y=test.predicted_sales,name="Predicted",mode="lines",line=dict(color="#f59e0b")))
            fig.update_layout(template="plotly_white",height=330,xaxis_title="Date",yaxis_title="Sales")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Evaluation uses an 80/20 chronological split. Forecast errors can accumulate over longer horizons.")
        except Exception as e: st.error(f"Forecast could not be generated: {e}")

else:
    st.subheader("Data explorer")
    a,b,c=st.columns(3)
    with a: fs=st.selectbox("Store filter",["All"]+sorted(data.store.unique()))
    with b: fc=st.selectbox("Category filter",["All"]+sorted(data.category.unique()))
    with c: start,end=st.date_input("Date range",value=(data.date.min().date(),data.date.max().date()),min_value=data.date.min().date(),max_value=data.date.max().date())
    view=data.copy()
    if fs!="All": view=view[view.store==fs]
    if fc!="All": view=view[view.category==fc]
    view=view[(view.date.dt.date>=start)&(view.date.dt.date<=end)]
    st.metric("Rows matching filters",f"{len(view):,}")
    st.dataframe(view,use_container_width=True,hide_index=True,height=430)
    st.download_button("Download filtered CSV",view.to_csv(index=False).encode(),file_name="retail_sales_filtered.csv",mime="text/csv",type="primary")
st.markdown("---")
st.caption("RETAILPULSE · Retail Sales Forecasting · Student project demonstration")
