import csv
import numpy as np
from collections import deque

# Global state to track history per symbol across function calls
# Structure: { symbol: { 'mid_prices': deque, 'prev_bid': val, 'prev_ask': val, 'prev_bid_depth': val, 'prev_ask_depth': val } }
_context = {}

def write_header(filename):
    header_list = [
        "date",
        "period",
        "symbol",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "best_bid_depth",
        "best_ask_depth",
        "deep_bid_depth",
        "deep_ask_depth",
        "imbalance",
        "skewness",
        "kurtosis",
        "volatility",      
        "row_total_qty",
        "row_vwa_price",
        "prop_participated",
        "is_prop_buy",
        "prop_price",
        "prop_qty",
        # New Features
        "ofi",              # Order Flow Imbalance
        "mid_price_velocity", # 1st derivative of mid-price over 10 windows
        "mid_price_volatility" # Std dev of mid-price over 10 windows
    ]
    with open(filename, mode="w", newline="") as file:
        csv.DictWriter(file, delimiter=",", fieldnames=header_list).writeheader()


def write_line(stock, date, filename, period):
    if bool(stock.buy_book.queue) and bool(stock.sell_book.queue):
        # --- Basic LOB State (Existing) ---
        best_bid = stock.buy_book.fetch_price()
        best_ask = stock.sell_book.fetch_price()
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        bid_volumes, bid_prices = stock.buy_book.fetch_data()
        ask_volumes, ask_prices = stock.sell_book.fetch_data()

        best_bid_depth = bid_volumes[0]
        best_ask_depth = ask_volumes[0]
        deep_bid_depth = bid_volumes.sum()
        deep_ask_depth = ask_volumes.sum()
        
        # Avoid division by zero
        total_depth = best_bid_depth + best_ask_depth
        order_imbalance = (best_bid_depth - best_ask_depth) / total_depth if total_depth > 0 else 0
        
        prices = np.concatenate([bid_prices, ask_prices])
        volumes = np.concatenate([bid_volumes, ask_volumes])

        weighted_mean = np.average(prices, weights=volumes)
        variance = np.average((prices - weighted_mean) ** 2, weights=volumes)
        std_dev = np.sqrt(variance)
        if std_dev > 0:
            skewness = np.average((prices - weighted_mean) ** 3, weights=volumes) / (std_dev**3)
            kurtosis = np.average((prices - weighted_mean) ** 4, weights=volumes) / (std_dev**4)
        else:
            skewness, kurtosis = 0, 0
            
        # Existing Volatility (LTP based)
        ltp_prices = np.array(stock.ltp)
        if len(ltp_prices) > 1:
            returns = np.diff(np.log(ltp_prices))
            volatility = np.sqrt(np.sum(returns**2))  # Realized volatility
        else:
            volatility = 0

        # New Features ---
        symbol = stock.code
        
        # Initialize context
        if symbol not in _context:
            _context[symbol] = {
                'mid_prices': deque(maxlen=10), # Window of 10 for velocity/volatility
                'prev_bid': best_bid,
                'prev_ask': best_ask,
                'prev_bid_depth': best_bid_depth,
                'prev_ask_depth': best_ask_depth
            }
        
        ctx = _context[symbol]
        
        # 1. Order Flow Imbalance (OFI)
        # OFI = OFI_bid - OFI_ask 
        # OFI > 0 implies buying pressure
        
        # Bid Side 
        prev_bid = ctx['prev_bid']
        prev_bid_depth = ctx['prev_bid_depth']
        
        ofi_bid = 0
        if best_bid > prev_bid:
            ofi_bid = best_bid_depth
        elif best_bid < prev_bid:
            ofi_bid = -prev_bid_depth
        else: # Price unchanged
            ofi_bid = best_bid_depth - prev_bid_depth
            
        # Ask Side
        prev_ask = ctx['prev_ask']
        prev_ask_depth = ctx['prev_ask_depth']
        
        ofi_ask = 0
        if best_ask < prev_ask: # Price dropping -> Selling pressure
            ofi_ask = -best_ask_depth
        elif best_ask > prev_ask: # Price rising -> Less selling
            ofi_ask = prev_ask_depth
        else: # Price unchanged
            ofi_ask = -(best_ask_depth - prev_ask_depth) # More ask depth = More selling = Negative OFI
            
        ofi = ofi_bid + ofi_ask
        
        # Update One-Step History
        ctx['prev_bid'] = best_bid
        ctx['prev_ask'] = best_ask
        ctx['prev_bid_depth'] = best_bid_depth
        ctx['prev_ask_depth'] = best_ask_depth

        # 2. Mid-Price History (Velocity & Volatility)
        ctx['mid_prices'].append(mid_price)
        history = np.array(ctx['mid_prices'])
        
        # Velocity: (P_t - P_{t-N}) / N 
        # Using simple change over available window
        if len(history) > 1:
            mid_price_velocity = (history[-1] - history[0]) / len(history)
        else:
            mid_price_velocity = 0
            
        # Volatility Surface (Std Dev of Mid Price)
        # Standard deviation of the Mid-Price to capture quote volatility
        if len(history) > 1:
            mid_price_volatility = np.std(history)
        else:
            mid_price_volatility = 0


        row = [
            date,
            period,
            stock.code,
            best_bid,
            best_ask,
            mid_price,
            spread,
            best_bid_depth,
            best_ask_depth,
            deep_bid_depth,
            deep_ask_depth,
            order_imbalance,
            skewness,
            kurtosis,
            volatility,
            stock.row_total_qty,
            stock.row_vwa_price / stock.row_total_qty if stock.row_total_qty > 0 else 0,
            stock.prop_participated,
            stock.is_prop_buy,
            stock.prop_price,
            stock.prop_qty,
            # New
            ofi,
            mid_price_velocity,
            mid_price_volatility
        ]

        row = [
            0 if (isinstance(x, float) and (np.isinf(x) or np.isnan(x))) else x
            for x in row
        ]

        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
