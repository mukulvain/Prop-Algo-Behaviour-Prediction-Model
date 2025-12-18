# Prop Algo Behaviour Prediction Model
*Given the market state at the start of the minute, what is the likelihood that the Prop Algo (Client Identity Flag 2) will enter the market at least once in the next 60 seconds?*

### Dataset
- 19th, 20th August 2019, Order and Trade Data for INFY Stock.
- LOB Aggregated at 10-second interval

### LOB State is defined as
- Best Bid ($P_b$): Highest price in the buy side.
- Best Ask ($P_a$): Lowest price in the sell side.
- Mid-Price: $(P_a + P_b) / 2$.
- Spread: $P_a - P_b$.
- Bid Depth ($Q_b$): Total volume available at the Best Bid.
- Ask Depth ($Q_a$): Total volume available at the Best Ask.
- Bid Deep Depth: Total volume available in the top 5 Bid levels.
- Bid Deep Depth: Total volume available in the top 5 Ask levels.
- Order Imbalance ($OI$): $\frac{Q_b - Q_a}{Q_b + Q_a}$. Indicates which side is "crowded".
- Skewness: Calculated using the prices of the top 5 levels, weighted by their volume. It tells the model if liquidity is "leaning" far away from the mid-price.
- Kurtosis: Measures if volume is concentrated at the top or spread out across many levels.
- Volatility: Standard deviation of the Last Traded Price over the last 50 trades.