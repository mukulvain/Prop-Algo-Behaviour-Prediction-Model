# Prop Algo Behaviour Prediction Model
*Given the market state at the start of the minute, what is the likelihood that the Prop Algo (Client Identity Flag 2) will enter the market at least once in the next 60 seconds? And if yes, what will be the price and quantity?*

### Dataset
- 19th, 20th August 2019, Order and Trade Data for INFY Stock.
- LOB Aggregated at 1-second interval

---

## Project Workflow & TODOs

### Goal
To model Prop Algo behavior by predicting their participation, price, and quantity, and verifying the model by reconstructing the LOB using "Rest of World" orders mixed with "Predicted Prop" orders.

### Methodology (Step-by-Step)
1. **State Extraction**: reliable extraction of LOB state vector at time $t$.
2. **Model Training**: Train a model on Day $N$ (e.g., Aug 19th) to predict Prop Algo behavior for the next period.
   - **Inputs**: Current LOB State.
   - **Target**: Participation (Yes/No), Side (Buy/Sell), Price (Limit Price), Quantity.
3. **Initial State**: Use Day $N+1$ (e.g., Aug 20th) pre-open state as the starting point.
4. **Simulation**: Reconstruct Day $N+1$ LOB using the Initial State and RoW Orders.
   - **Injection**: At each step, use the Model (trained on Day $N$) to predict if Prop Algo *would* act given the *current simulated* LOB state. 
5. **Reconstruction**: Run the simulation for the full day to generate `Predicted LOB`.The entire previous row is used for next step prediction. 4 fields are predicted. Prop algo participation, side (buy/sell), price & quantity.
6. **Comparison**: Compare `Predicted LOB` vs `Actual LOB` (Day $N+1$).
7. **Metrics**: Calculate error metrics (e.g., RMSE of Mid-Price, Spread deviation).
8. **Iterative Improvement**: Update model parameters based on errors and repeat for subsequent days.


### LOB State is defined as
- Best Bid ($P_b$): Highest price in the buy side.
- Best Ask ($P_a$): Lowest price in the sell side.
- Mid-Price: $(P_a + P_b) / 2$.
- Spread: $P_a - P_b$.
- Bid Depth ($Q_b$): Total volume available at the Best Bid.
- Ask Depth ($Q_a$): Total volume available at the Best Ask.
- Bid Deep Depth: Total volume available in the top 5 Bid levels.
- Ask Deep Depth: Total volume available in the top 5 Ask levels.
- Order Imbalance ($OI$): $\frac{Q_b - Q_a}{Q_b + Q_a}$. Indicates which side is "crowded".
- Skewness: Calculated using the prices of the top 5 levels, weighted by their volume. It tells the model if liquidity is "leaning" far away from the mid-price.
- Kurtosis: Measures if volume is concentrated at the top or spread out across many levels.
- Volatility: Standard deviation of the Last Traded Price over the last 50 trades.
- Order Flow Imbalance (OFI): $OFI_t = e_t \times q_t$. Indicates which side is "pressuring" prices short-term.
- Mid-Price Velocity: Rate of change of Mid-Price over the last 10 seconds.
- Mid-Price Volatility: Standard deviation of the Mid-Price over the last 10 seconds (Vol Surface).


