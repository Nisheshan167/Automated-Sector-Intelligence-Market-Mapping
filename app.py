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

    prices = t.history(
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False
    )[["Close"]].rename(columns={"Close": "price"})

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
st.subheader(" Annualized Volatility Comparison")

fig, ax = plt.subplots()
metrics_df["Annualized Volatility"].sort_values().plot(
    kind="bar",
    ax=ax,
    ylabel="Annualized Volatility",
    title="Volatility Comparison (Lower = Better for Capital Preservation)"
)
ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
st.pyplot(fig)

lowest_vol = metrics_df["Annualized Volatility"].idxmin()
highest_vol = metrics_df["Annualized Volatility"].idxmax()

st.markdown(
    f"""
**Interpretation**
-  **Lowest volatility:** `{lowest_vol}` → strongest capital-preservation profile  
-  **Highest volatility:** `{highest_vol}` → higher downside risk  
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

# ---------------- CORRELATION INTERPRETATION ----------------
st.subheader(" Correlation Interpretation")

# Average correlation per stock (exclude self-correlation)
avg_corr_per_stock = corr.apply(
    lambda row: row.drop(row.name).mean(),
    axis=1
)

most_corr_pair = corr.where(~np.eye(len(corr), dtype=bool)).stack().idxmax()
least_corr_pair = corr.where(~np.eye(len(corr), dtype=bool)).stack().idxmin()

overall_avg_corr = (
    corr.where(~np.eye(len(corr), dtype=bool))
        .stack()
        .mean()
)

st.markdown("**Key observations based on computed correlations:**")

# Per-stock interpretation
for ticker, val in avg_corr_per_stock.items():
    st.markdown(
        f"- **{ticker}** average correlation with peers: **{val:.2f}** "
        f"→ {'moves closely with the sector' if val > 0.7 else 'shows some diversification benefit' if val < 0.5 else 'moderate co-movement'}"
    )

st.markdown(
    f"""
-  **Most correlated pair:** `{most_corr_pair[0]}` & `{most_corr_pair[1]}`  
  (correlation = **{corr.loc[most_corr_pair]:.2f}**) → tend to move together

-  **Least correlated pair:** `{least_corr_pair[0]}` & `{least_corr_pair[1]}`  
  (correlation = **{corr.loc[least_corr_pair]:.2f}**) → better diversification between them

-  **Overall average correlation across selected BDCs:** **{overall_avg_corr:.2f}**  
  → indicates {'a tightly coupled sector' if overall_avg_corr > 0.7 else 'partial diversification within the sector'}
"""
)

# ---------------- RULE-BASED RECOMMENDATION ----------------
st.subheader(" Capital Preservation Recommendation (Rule-Based)")

ranked = metrics_df.sort_values(
    ["Annualized Volatility", "Total Return (Div Reinvested)"],
    ascending=[True, False]
)

recommended = ranked.index[0]
rec_vol = ranked.loc[recommended, "Annualized Volatility"]
rec_tr  = ranked.loc[recommended, "Total Return (Div Reinvested)"]

st.markdown(
    f"""
**Recommended ticker:** `{recommended}`  
- Annualized volatility: **{rec_vol:.2%}**  
- 5Y total return (dividends reinvested): **{rec_tr:.2%}**

**Rule:** Lowest volatility → tie-break by higher total return.
"""
)

# ---------------- AI EXPLANATION ----------------
st.subheader("🤖 AI Explanation (Explanation-Only)")

prompt = f"""
You are an investment analyst.

A rule-based model has already selected **{recommended}** as the best BDC for capital preservation.

Rule used:
- Primary: lowest annualized volatility
- Secondary (tie-break): higher 5-year total return with dividends reinvested

Metrics:
{metrics_df.to_string()}

Task:
Explain why {recommended} is suitable for capital preservation based on these metrics.
Write 6–8 concise bullet points.
Do NOT contradict the rule-based result.
"""

if st.button("Generate AI Explanation"):
    with st.spinner("Generating explanation..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        st.markdown(response.choices[0].message.content)
