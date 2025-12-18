from Book import Book
from collections import deque

class Ticker:
    def __init__(self, code):
        self.code = code
        self.ltp = deque(maxlen=50)
        self.buy_book = Book(True)
        self.sell_book = Book(False)
        self.prop_participated = False
        self.is_prop_buy = None
        self.prop_price = None
        self.prop_qty = None
        self.row_vwa_price = 0
        self.row_total_qty = 0