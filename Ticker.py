from Book import Book
from collections import deque

class Ticker:
    def __init__(self, code):
        self.code = code
        self.ltp = deque(maxlen=50)
        self.buy_book = Book(True)
        self.sell_book = Book(False)