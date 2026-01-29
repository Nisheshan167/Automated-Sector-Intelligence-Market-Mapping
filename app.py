import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

# ---------------- CONFIG ----------------
START_DATE = "2021-01-29"
END_DATE   = "2026-01-29"
TRADING_DAYS = 252

AVAILABLE_TICKERS = ["MAIN", "ARCC", "HTGC", "OBDC"]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="BDC Analyzer", layout="wide")
st.title("📊 BDC Sector Analysis & Capital Preservation Tool")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Stock Selection")

tickers = st.sidebar.multiselect(
    "Select BDC tickers",
    AVAILABLE_TICKERS,
    default=AVAILABLE_TICKERS
)

if len(tickers) == 0:
    st.warning("Please select at least one ticker.")
    st.stop()

# ---------------- DATA FUNCTIONS ----------------
@st.cache_data
def fetch_data(ticker):
    t = yf.Ticker(ticker)

    # RAW close prices (not adjusted)
    prices = t.history(
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False
    )[["Close"]]
    prices = prices.rename(columns={"Close": "price"})

    div = t.dividends.rename("dividend_per_share")

    df = prices.join(div, how="left")
    df["dividend_per_share"] = df["dividend_per_share"].fillna(0.0)

    return df


def annualized_vol(df):
    returns = df["price"].pct_change().dropna()
    return returns.std() * np.sqrt(TRADING_DAYS)


def total_return_div_reinvest(df):
    shares = 1.0
    start_price = df["price"].iloc[0]

    for _, row in df.iterrows():
        if row["dividend_per_share"] > 0 and row["price"] > 0:
            shares += (shares * row["dividend_per_share"]) / row["price"]

    end_price = df["price"].iloc[-1]
    return (shares * end_price) / start_price - 1


# ---------------- CALCULATIONS ----------------
data = {t: fetch_data(t) for t in tickers}

rows = []
returns = {}

for t in tickers:
    df = data[t]
    rows.append({
        "Ticker": t,
        "Annualized Volatility": annualized_vol(df),
        "Total Return (Div Reinvested)": total_return_div_reinvest(df),
        "Start Price": df["price"].iloc[0],
        "End Price": df["price"].iloc[-1],
    })
    returns[t] = df["price"].pct_change()

metrics_df = pd.DataFrame(rows).set_index("Ticker")

# ---------------- METRICS TABLE ----------------
st.subheader("📈 Performance Metrics")

st.dataframe(
    metrics_df.style.format({
        "Annualized Volatility": "{:.2%}",
        "Total Return (Div Reinvested)": "{:.2%}",
        "Start Price": "{:.2f}",
        "End Price": "{:.2f}",
    })
)

# ---------------- VOLATILITY BAR CHART ----------------
st.subheader("📉 Annualized Volatility Comparison")

fig, ax = plt.subplots()
metrics_df["Annualized Volatility"].sort_values().plot(
    kind="bar",
    ax=ax,
    ylabel="Annualized Volatility",
    title="Volatility (Lower = Better for Capital Preservation)"
)
ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
st.pyplot(fig)

lowest_vol = metrics_df["Annualized Volatility"].idxmin()
highest_vol = metrics_df["Annualized Volatility"].idxmax()

st.markdown(
    f"""
**Interpretation**
- 🟢 **Lowest volatility:** `{lowest_vol}` → strongest capital-preservation profile  
- 🔴 **Highest volatility:** `{highest_vol}` → higher downside risk in stressed markets  
"""
)

# ---------------- CORRELATION HEATMAP ----------------
st.subheader("🔗 Correlation Matrix (Daily CLOSE Returns)")

returns_df = pd.DataFrame(returns).dropna()
corr = returns_df.corr()

fig, ax = plt.subplots()
im = ax.imshow(corr.values)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.index)

for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

ax.set_title("Correlation Heatmap")
plt.colorbar(im)
st.pyplot(fig)

# ---------------- MARKET REGIME ----------------
st.subheader("⚠️ Market Regime")

avg_vol = metrics_df["Annualized Volatility"].mean()
regime = "HIGH VOLATILITY" if avg_vol > 0.25 else "NORMAL VOLATILITY"

st.markdown(f"**Detected Regime:** `{regime}`")

# ---------------- AI INVESTMENT LOGIC ----------------
st.subheader("🤖 AI Investment Logic")

prompt = f"""
You are an investment analyst.

Market regime: {regime}

Metrics:
{metrics_df.to_string()}

Question:
Which stock is best suited for capital preservation in a high-volatility regime and why?
Focus on volatility, downside risk, and dividend stability.
Answer in 6–8 concise bullet points.
"""

if st.button("Generate AI Explanation"):
    with st.spinner("Generating explanation..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        st.markdown(response.choices[0].message.content)
