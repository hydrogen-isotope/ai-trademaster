namespace Trademaster.Core;

public enum EventKind : byte { Quote = 1, Trade = 2, Status = 3 }

public enum TradingStatus : byte { Trading = 0, Halted = 1 }

/// <summary>
/// One market-data record. Field meaning depends on <see cref="Kind"/>:
/// Quote: Px1 = bid, Px2 = ask, Sz1 = bid size, Sz2 = ask size.
/// Trade: Px1 = price, Sz1 = size.
/// Status: <see cref="Status"/>.
/// </summary>
public readonly record struct MarketEvent(
    long Seq,
    long ExchangeTs,
    long RecvTs,
    int Instrument,
    EventKind Kind,
    long Px1,
    long Px2,
    long Sz1,
    long Sz2,
    TradingStatus Status)
{
    public Price Bid => new(Px1);
    public Price Ask => new(Px2);
    public long BidSize => Sz1;
    public long AskSize => Sz2;
    public Price TradePrice => new(Px1);
    public long TradeSize => Sz1;

    public static MarketEvent Quote(long seq, long exchTs, long recvTs, int instrument, Price bid, Price ask, long bidSize, long askSize) =>
        new(seq, exchTs, recvTs, instrument, EventKind.Quote, bid.Raw, ask.Raw, bidSize, askSize, TradingStatus.Trading);

    public static MarketEvent Trade(long seq, long exchTs, long recvTs, int instrument, Price price, long size) =>
        new(seq, exchTs, recvTs, instrument, EventKind.Trade, price.Raw, 0, size, 0, TradingStatus.Trading);

    public static MarketEvent StatusChange(long seq, long exchTs, long recvTs, int instrument, TradingStatus status) =>
        new(seq, exchTs, recvTs, instrument, EventKind.Status, 0, 0, 0, 0, status);
}
