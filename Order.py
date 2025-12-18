class Order:
    def __init__(
        self,
        record,
        segment,
        order_number,
        order_time,
        is_buy,
        activity_type,
        symbol,
        series,
        volume_disclosed,
        volume_original,
        limit_price,
        trigger_price,
        is_market_order,
        is_stop_loss,
        is_ioc,
        is_prop_algo,
    ):
        self.activities = {1: "ENTRY", 3: "CANCEL", 4: "MODIFY"}
        self.record = record
        self.segment = segment
        self.order_number = order_number
        self.order_time = order_time
        self.is_buy = is_buy == "B"
        self.activity_type = self.activities[activity_type]
        self.symbol = symbol
        self.series = series
        self.volume_disclosed = (
            volume_disclosed if volume_disclosed else volume_original
        )
        self.volume_original = volume_original
        self.limit_price = limit_price
        self.trigger_price = trigger_price
        self.is_market_order = is_market_order == "Y"
        self.is_stop_loss = is_stop_loss == "Y"
        self.is_ioc = is_ioc == "Y"
        self.is_prop_algo = bool(is_prop_algo)

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}({self.order_number})"
