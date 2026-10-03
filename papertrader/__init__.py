"""papertrader: a small, careful paper-trading research stack.

Layers, in the order data flows through them:

    data        -> collect, validate and clean bars          (papertrader.data)
    strategies  -> turn bars into target weights             (papertrader.strategies)
    engine      -> turn weights into proposed orders         (papertrader.engine)
    risk        -> approve / reject every order, kill switch (papertrader.risk)
    execution   -> simulated paper broker                    (papertrader.execution)
    journal     -> append-only log of all of the above       (papertrader.journal)

Backtests and paper trading run through the same TradingSession, so they
cannot quietly drift apart.
"""

__version__ = "0.1.0"
