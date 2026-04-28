# HOSPITAL FINANCIAL STRESS TESTING SIMULATOR

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import yfinance as yf
import streamlit as st   # ✅ ADDED

st.title("🏥 Hospital Financial Stress Testing Dashboard")  # ✅ ADDED

tickers = ["HCA", "THC", "UHS", "CYH", "ENSG",
           "LPNT", "SEM", "ACHC", "AMN", "AMED"]

rows = []

for ticker in tickers:
    stock = yf.Ticker(ticker)
    info = stock.info

    name     = info.get("longName", ticker)
    revenue  = info.get("totalRevenue", 0)
    expenses = info.get("totalExpenses", revenue * 0.85)
    cash     = info.get("totalCash", 0)
    assets   = info.get("totalAssets", 0)
    profit   = info.get("netIncomeToCommon", 0)

    rows.append({
        "FAC_NAME"      : name,
        "total_revenue" : revenue,
        "TOT_OP_EXP"    : expenses,
        "CASH"          : cash,
        "TOT_ASSETS"    : assets,
        "profit"        : profit
    })

df = pd.DataFrame(rows)
df = df[df["total_revenue"] > 0]

# Financial Calculations
df["monthly_expense"]   = df["TOT_OP_EXP"] / 12
df["survival_months"]   = df["CASH"] / df["monthly_expense"]
df["survival_months"]   = df["survival_months"].replace([np.inf, -np.inf], 0).fillna(0)

# Revenue Prediction
X = df[["TOT_ASSETS","TOT_OP_EXP"]].fillna(0)
Y = df["total_revenue"]
model = LinearRegression()
model.fit(X, Y)
df["predicted_revenue"] = model.predict(X)

# Stress Testing
# ================= SLIDERS ================= #

st.sidebar.header("⚙️ Stress Test Controls")

patient_drop = st.sidebar.slider(
    "Patient Drop (%)",
    min_value=0,
    max_value=80,
    value=30
)

cost_increase = st.sidebar.slider(
    "Operating Cost Increase (%)",
    min_value=0,
    max_value=50,
    value=15
)

# Convert % to decimal
patient_drop = patient_drop / 100
cost_increase = cost_increase / 100

# ================= STRESS TESTING ================= #

df["revenue_after_patient_drop"] = df["total_revenue"] * (1 - patient_drop)

df["expense_after_cost_increase"] = df["TOT_OP_EXP"] * (1 + cost_increase)

df["monthly_expense_crisis"] = df["expense_after_cost_increase"] / 12

df["survival_after_crisis"] = df["CASH"] / df["monthly_expense_crisis"]

df["survival_after_crisis"] = df["survival_after_crisis"].replace([np.inf, -np.inf], 0).fillna(0)
#DASHBOARDS VISUALIZATIONS

st.subheader("📊 Dataset Preview")  
st.dataframe(df)                   


col1, col2, col3 = st.columns(3)

col1.metric("Avg Revenue", f"{df['total_revenue'].mean():,.0f}")
col2.metric("Avg Survival (Months)", f"{df['survival_months'].mean():.1f}")
col3.metric("Avg Crisis Survival", f"{df['survival_after_crisis'].mean():.1f}")


# FIG 1
fig1 = px.bar(
    df,
    x="FAC_NAME",
    y=["total_revenue", "predicted_revenue"],
    title="Actual vs Predicted Revenue",
    barmode="group"
)
st.plotly_chart(fig1)   

# FIG 2
top10_profit = df.sort_values("profit", ascending=False).head(10)
fig2 = px.bar(
    top10_profit,
    x="FAC_NAME",
    y="profit",
    title="Top 10 Hospitals by Profit"
)
st.plotly_chart(fig2)   

# FIG 3
fig3 = px.scatter(
    df,
    x="CASH",
    y="survival_months",
    size="TOT_OP_EXP",
    color="survival_months",
    hover_name="FAC_NAME",
    title="Financial Risk Analysis",
    color_continuous_scale=["red", "yellow", "green"],
    trendline="ols"
)
st.plotly_chart(fig3)   

# FIG 4
fig4 = px.bar(
    df,
    x="FAC_NAME",
    y="total_revenue",
    title="Top Hospitals by Revenue"
)
st.plotly_chart(fig4)   

# FIG 5
fig5 = px.scatter(
    df,
    x="survival_months",
    y="survival_after_crisis",
    hover_name="FAC_NAME",
    title="Survival Before vs After Crisis",
    color="survival_months",
    color_continuous_scale=["red", "yellow", "green"]
)
st.plotly_chart(fig5)   