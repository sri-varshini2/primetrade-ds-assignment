import pandas as pd

# Load datasets
trader = pd.read_csv("historical_data.csv")
sentiment = pd.read_csv("fear_greed_index.csv")

# Convert trader timestamp
trader["Timestamp"] = pd.to_datetime(trader["Timestamp"], unit="ms")
trader["date"] = trader["Timestamp"].dt.date

# Convert sentiment date
sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date

# Convert numeric columns
trader["Closed PnL"] = pd.to_numeric(trader["Closed PnL"], errors="coerce")
trader["Size USD"] = pd.to_numeric(trader["Size USD"], errors="coerce")

# Merge
merged = pd.merge(
    trader,
    sentiment[["date", "classification", "value"]],
    on="date",
    how="left"
)

# Keep only rows with sentiment information
data = merged.dropna(subset=["classification"]).copy()

# --------------------------------
# PnL Analysis
# --------------------------------

summary = data.groupby("classification").agg(
    Trades=("Closed PnL", "count"),
    Average_PnL=("Closed PnL", "mean"),
    Total_PnL=("Closed PnL", "sum"),
    Winning_Trades=("Closed PnL", lambda x: (x > 0).sum()),
    Losing_Trades=("Closed PnL", lambda x: (x < 0).sum()),
    Average_Trade_Size=("Size USD", "mean")
)

# Win rate
summary["Win_Rate_%"] = (
    summary["Winning_Trades"] /
    (summary["Winning_Trades"] + summary["Losing_Trades"])
) * 100

print("\n==============================")
print("TRADING PERFORMANCE BY SENTIMENT")
print("==============================")

print(summary.round(2))

# --------------------------------
# Long vs Short
# --------------------------------

side_analysis = data.groupby(
    ["classification", "Side"]
)["Closed PnL"].agg(
    ["count", "mean", "sum"]
)

print("\n==============================")
print("LONG vs SHORT PERFORMANCE")
print("==============================")

print(side_analysis.round(2))
import matplotlib.pyplot as plt

# -----------------------------
# Graph 1: Average PnL by Sentiment
# -----------------------------

avg_pnl = data.groupby("classification")["Closed PnL"].mean()

plt.figure(figsize=(8, 5))
avg_pnl.sort_values().plot(kind="bar")
plt.title("Average PnL by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Closed PnL")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("average_pnl_by_sentiment.png")
plt.show()


# -----------------------------
# Graph 2: Win Rate by Sentiment
# -----------------------------

win_rate = data.groupby("classification")["Closed PnL"].apply(
    lambda x: (x > 0).mean() * 100
)

plt.figure(figsize=(8, 5))
win_rate.sort_values().plot(kind="bar")
plt.title("Win Rate by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Win Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("win_rate_by_sentiment.png")
plt.show()


# -----------------------------
# Graph 3: BUY vs SELL PnL
# -----------------------------

side_pnl = data.groupby(
    ["classification", "Side"]
)["Closed PnL"].mean().unstack()

side_pnl.plot(kind="bar", figsize=(9, 5))

plt.title("Average PnL by Sentiment and Trading Side")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Closed PnL")
plt.xticks(rotation=0)
plt.legend(title="Side")
plt.tight_layout()
plt.savefig("side_pnl_by_sentiment.png")
plt.show()


# -----------------------------
# Graph 4: Trading Volume
# -----------------------------

trade_count = data.groupby("classification").size()

plt.figure(figsize=(8, 5))
trade_count.sort_values().plot(kind="bar")
plt.title("Number of Trades by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Number of Trades")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("trades_by_sentiment.png")
plt.show()

print("\nGraphs created successfully!")
print("\n======================================")
print("FINAL KEY INSIGHTS")
print("======================================")

print("\n1. Greed has the highest average PnL:")
print("   Average PnL during Greed =", round(
    data[data["classification"] == "Greed"]["Closed PnL"].mean(), 2
))

print("\n2. Fear has the highest win rate:")
fear = data[data["classification"] == "Fear"]
fear_win_rate = (fear["Closed PnL"] > 0).mean() * 100
print("   Fear win rate =", round(fear_win_rate, 2), "%")

print("\n3. Trading performance differs significantly by sentiment.")

print("\n4. During Greed, SELL trades have higher average PnL than BUY trades.")

print("\n5. During Fear, BUY trades have higher average PnL than SELL trades.")

print("\n6. Extreme Greed does not necessarily produce the best performance.")