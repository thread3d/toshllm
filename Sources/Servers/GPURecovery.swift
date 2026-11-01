import Foundation

/// A timed-out GPU job leaves the Metal backend failing until the engine restarts: one restart per failure,
/// at most one every ten minutes per server, so a card that keeps failing is not restarted in a loop.
final class GPURecovery: @unchecked Sendable {
    static let shared = GPURecovery()

    static let interval: TimeInterval = 600
    /// A restart this recent is the one for the failure the chat is telling about.
    static let current: TimeInterval = 20

    enum Outlook: Equatable { case restarting, withheld }

    private let lock = NSLock()
    private var last: [Int: Date] = [:]

    static func reports(_ line: Substring) -> Bool {
        line.contains("kIOAccelCommandBufferCallbackErrorTimeout") || line.contains("backend is in error state")
    }

    /// Whether a failure seen now on this port gets a restart; a granted one is recorded.
    func admit(port: Int, at now: Date = Date()) -> Bool {
        lock.lock(); defer { lock.unlock() }
        if let previous = last[port], now.timeIntervalSince(previous) <= Self.interval { return false }
        last[port] = now
        return true
    }

    /// What a failure seen now on this port leads to.
    func outlook(port: Int, at now: Date = Date()) -> Outlook {
        lock.lock(); defer { lock.unlock() }
        guard let previous = last[port] else { return .restarting }
        let age = now.timeIntervalSince(previous)
        return age <= Self.current || age > Self.interval ? .restarting : .withheld
    }

    /// Whether this port's server was restarted for a GPU failure after `date`, as under a request sent then.
    func restarted(port: Int, since date: Date) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return last[port].map { $0 >= date } ?? false
    }

    func reset() {
        lock.lock(); defer { lock.unlock() }
        last = [:]
    }
}
