import numpy as np
from sortedcontainers import SortedDict

from reader import order_repository


class Book:
    def __init__(self, buy=True):
        self.buy = buy
        self.queue = SortedDict()
        self.orders = {}
        self.stop_loss_orders = {}

    # Fetches the best price of the queue
    def fetch_price(self):
        if self.buy:
            return self.queue.peekitem()[0]
        else:
            return self.queue.peekitem(0)[0]

    def add(self, order):
        # Adds order to stop loss queue
        if order.is_stop_loss:
            if order.order_number in self.stop_loss_orders:
                del self.stop_loss_orders[order.order_number]
                order.is_stop_loss = False
                if order.is_market_order:
                    del order_repository[order.order_number]
                else:
                    order_repository[order.order_number].is_stop_loss = False
                    self.add(order)
            else:
                self.stop_loss_orders[order.order_number] = order
        # Adds order to the queue
        else:
            self.orders[order.order_number] = order.limit_price
            if order.limit_price not in self.queue:
                self.queue[order.limit_price] = [order]
            else:
                self.queue[order.limit_price].append(order)

    def delete(self, order_number, is_stop_loss, volume=0):
        if is_stop_loss:
            del self.stop_loss_orders[order_number]
            del order_repository[order_number]

        # Removes order from queue
        elif order_number in self.orders:
            price = self.orders[order_number]
            for i in range(len(self.queue[price])):
                if self.queue[price][i].order_number == order_number:
                    if not volume or self.queue[price][i].volume_original == volume:
                        self.queue[price].pop(i)
                        del self.orders[order_number]
                        del order_repository[order_number]
                    else:
                        self.queue[price][i].volume_original -= volume
                    break
            if not len(self.queue[price]):
                del self.queue[price]

    def fetch_data(self):
        TOP = 5
        volumes = np.zeros(TOP)
        keys = list(self.queue.keys())

        if self.buy:
            prices = np.zeros(TOP)
            keys.reverse()
        else:
            prices = np.full(TOP, np.inf)

        i = 0
        for key in keys:
            if i >= TOP:
                break
            for order in self.queue[key]:
                volumes[i] += order.volume_original

                if self.buy:
                    prices[i] = max(prices[i], order.limit_price)
                else:
                    prices[i] = min(prices[i], order.limit_price)
            i += 1

        return volumes, prices
