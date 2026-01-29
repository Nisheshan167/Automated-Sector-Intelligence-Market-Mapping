import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai

# ---------------- CONFIG ----------------
START_DATE = "2021-01-29"
END_DATE   = "2026-01-29"
TRADING_DAYS = 252

AVAILABLE_TICKERS = ["MAIN", "ARCC", "HTGC", "OBDC"]

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="BDC Analyzer", layout="wide")
st.title("📊 BDC Sector Analysis & Capital Preservation Tool")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Stock Selection")

num_stocks = st.sidebar.slider(
    "Number of stocks",
    min_value=1,
    max_value=len(AVAILABLE_TICKERS),
    value=4
)

tickers = st.sidebar.multiselect(
    "Select tickers",
    AVAILABLE_TICKERS,
    default=AVAILABLE_TICKERS[:num_stocks]
)

if len(tickers) == 0:
    st.stop()

# ---------------- DATA FUNCTIONS ----------------
@st.cache_data
def fetch_data(ticker):
    t = yf.Ticker(ticker)
    prices = t.history(start=START_DATE, end=END_DATE)[["Close"]]
    prices.columns = ["price"]
    div = t.dividends.rename("dividend_per_share")
    df = prices.join(div, how="left")
    df["dividend_per_share"] = df["dividend_per_share"].fillna(0.0)
    return df


def annualized_vol(df):
    rets = df["price"].pct_change().dropna()
    return rets.std() * np.sqrt(TRADING_DAYS)


def total_return_div_reinvest(df):
    shares = 1.0
    start_price = df["price"].iloc[0]

    for _, row in df.iterrows():
        if row["dividend_per_share"] > 0:
            shares += (shares * row["dividend_per_share"]) / row["price"]

    end_price = df["price"].iloc[-1]
    return (shares * end_price) / start_price - 1


# ---------------- CALCULATIONS ----------------
data = {t: fetch_data(t) for t in tickers}

metrics = []
returns_df = {}

for t in tickers:
    df = data[t]
    metrics.append({
        "Ticker": t,
        "Annualized Volatility": annualized_vol(df),
        "Total Return (Div Reinvested)": total_return_div_reinvest(df),
        "Start Price": df["price"].iloc[0],
        "End Price": df["price"].iloc[-1]
    })
    returns_df[t] = df["price"].pct_change()

metrics_df = pd.DataFrame(metrics).set_index("Ticker")

# ---------------- DISPLAY METRICS ----------------
st.subheader("📈 Performance Metrics")
st.dataframe(metrics_df.style.format("{:.2%}"))

# ---------------- CORRELATION ----------------
st.subheader("🔗 Correlation Matrix (Daily CLOSE Returns)")

returns_df = pd.DataFrame(returns_df).dropna()
corr = returns_df.corr()

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr.values)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.index)

for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

plt.title("Correlation Heatmap")
plt.colorbar(im)
st.pyplot(fig)

# ---------------- VOLATILITY REGIME ----------------
st.subheader("⚠️ Volatility Regime")

avg_vol = metrics_df["Annualized Volatility"].mean()

regime = "HIGH VOLATILITY" if avg_vol > 0.25 else "NORMAL VOLATILITY"
st.markdown(f"**Detected Regime:** `{regime}`")

# ---------------- AI INVESTMENT LOGIC ----------------
st.subheader("🤖 AI Investment Logic")

best_stock = metrics_df.sort_values(
    ["Annualized Volatility", "Total Return (Div Reinvested)"],
    ascending=[True, False]
).index[0]

prompt = f"""
You are an investment analyst.

Market regime: {regime}

Metrics:
{metrics_df.to_string()}

Question:
Which stock is best suited for capital preservation in a high-volatility regime and why?
Focus on volatility, drawdowns, and dividend stability.
Keep answer concise (6–8 bullets).
"""

if st.button("Generate AI Explanation"):
    with st.spinner("Thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        st.markdown(response.choices[0].message.content)
