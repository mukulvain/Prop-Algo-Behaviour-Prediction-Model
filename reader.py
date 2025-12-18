import csv
import gzip
from datetime import datetime, timedelta

from Order import Order
from Trade import Trade

order_repository = {}
symbols_file = "NSE500_2019.csv"


class AlphaNumeric:
    def __init__(self, length):
        self.value_type = str
        self.length = length


class Numeric:
    def __init__(self, length):
        self.value_type = int
        self.length = length


order = [
    AlphaNumeric(2),
    AlphaNumeric(4),
    Numeric(16),
    Numeric(14),
    AlphaNumeric(1),
    Numeric(1),
    AlphaNumeric(10),
    AlphaNumeric(2),
    Numeric(8),
    Numeric(8),
    Numeric(8),
    Numeric(8),
    AlphaNumeric(1),
    AlphaNumeric(1),
    AlphaNumeric(1),
    Numeric(1),
]

trade = [
    AlphaNumeric(2),
    AlphaNumeric(4),
    Numeric(16),
    Numeric(14),
    AlphaNumeric(10),
    AlphaNumeric(2),
    Numeric(8),
    Numeric(8),
    Numeric(16),
    Numeric(1),
    Numeric(16),
    Numeric(1),
]


def to_order(line):
    ptr = 0
    order_args = []
    for var in order:
        order_args.append(var.value_type(line[ptr : ptr + var.length]))
        ptr += var.length
    return Order(*order_args)


def to_trade(line):
    ptr = 0
    trade_args = []
    for var in trade:
        trade_args.append(var.value_type(line[ptr : ptr + var.length]))
        ptr += var.length
    return Trade(*trade_args)


def line_reader(file_path):
    with gzip.open(file_path, "rt") as file:
        for line in file:
            yield line.strip()


def get_symbols():
    symbols = []
    with open(symbols_file, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for line in file:
            symbols.append(line.strip())
    return symbols


def get_trade(trade_reader):
    try:
        return to_trade(next(trade_reader))
    except StopIteration:
        return None


def get_order(order_reader):
    try:
        return to_order(next(order_reader))
    except StopIteration:
        return None


def clock_time(jiffies):
    epoch = datetime(1980, 1, 1)
    seconds = jiffies / 65536
    timestamp = epoch + timedelta(seconds=seconds)
    return timestamp.time()


def add_time(timestamp, interval):
    datetimestamp = datetime.combine(datetime.today(), timestamp) + timedelta(
        seconds=interval
    )
    return datetimestamp.time()
