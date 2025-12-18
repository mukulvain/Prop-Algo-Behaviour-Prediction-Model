import csv

import numpy as np


def write_header(filename):
    header_list = [
        "date",
        "period",
        "symbol",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "bid_depth",
        "ask_depth",
        "deep_bid_depth",
        "deep_ask_depth",
        "imbalance",
        "skewness",
        "kurtosis",
        "volatility"
    ]
    with open(filename, mode="w", newline="") as file:
        csv.DictWriter(file, delimiter=",", fieldnames=header_list).writeheader()


def write_line(stock, date, filename, period):
    if bool(stock.buy_book.queue) and bool(stock.sell_book.queue):
        best_bid = stock.buy_book.fetch_price()
        best_ask = stock.sell_book.fetch_price()
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        bid_volumes, bid_prices = stock.buy_book.fetch_data()
        ask_volumes, ask_prices = stock.sell_book.fetch_data()

        bid_depth = bid_volumes[0]
        ask_depth = ask_volumes[0]
        deep_bid_depth = bid_volumes.sum()
        deep_ask_depth = ask_volumes.sum()
        order_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        prices = np.concatenate([bid_prices, ask_prices])
        volumes = np.concatenate([bid_volumes, ask_volumes])

        weighted_mean = np.average(prices, weights=volumes)
        variance = np.average((prices - weighted_mean) ** 2, weights=volumes)
        std_dev = np.sqrt(variance)
        if std_dev > 0:
            skewness = np.average((prices - weighted_mean) ** 3, weights=volumes) / (
                std_dev**3
            )
            kurtosis = np.average((prices - weighted_mean) ** 4, weights=volumes) / (
                std_dev**4
            )
        else:
            skewness, kurtosis = 0, 0
        prices = np.array(stock.ltp)
        returns = np.diff(np.log(prices))
        volatility = np.sqrt(np.sum(returns**2)) # Realized volatility

        row = [
            date,
            period,
            stock.code,
            best_bid,
            best_ask,
            mid_price,
            spread,
            bid_depth,
            ask_depth,
            deep_bid_depth,
            deep_ask_depth,
            order_imbalance,
            skewness,
            kurtosis,
            volatility
        ]
        row = [
            0 if (isinstance(x, float) and (np.isinf(x) or np.isnan(x))) else x
            for x in row
        ]


        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
