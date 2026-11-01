// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Where a remote MCP server's requests may go: the scheme, host and effective port of the
/// address the user configured. An endpoint it announces, or a redirect, stays on it.
struct MCPOrigin: Equatable, CustomStringConvertible {
    let scheme: String
    let host: String
    let port: Int

    /// Nil for a URL with credentials in it, so it never matches.
    init?(_ url: URL) {
        guard url.user == nil, url.password == nil,
              let host = url.host?.lowercased(), !host.isEmpty else { return nil }
        // A WebSocket handshake is an HTTP request to the same origin.
        switch url.scheme?.lowercased() {
        case "http", "ws": scheme = "http"
        case "https", "wss": scheme = "https"
        default: return nil
        }
        self.host = host
        port = url.port ?? (scheme == "https" ? 443 : 80)
    }

    var description: String { "\(scheme)://\(host):\(port)" }

    /// The endpoint the older SSE transport posts to, when the announcement keeps to `base`'s origin.
    static func endpoint(_ announced: String, of base: URL) -> URL? {
        guard let url = URL(string: announced.trimmingCharacters(in: .whitespaces), relativeTo: base)?.absoluteURL,
              let origin = MCPOrigin(url), origin == MCPOrigin(base) else { return nil }
        return url
    }
}

/// Refuses a redirect that leaves the configured origin: URLSession would otherwise send the
/// server's custom headers, and a 307/308 body, to the new host.
final class MCPRedirectGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let origin: MCPOrigin?
    private let lock = NSLock()
    private var refused: String?

    init(_ url: URL) {
        origin = MCPOrigin(url)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        let target = request.url.flatMap(MCPOrigin.init)
        if let origin, target == origin { return completionHandler(request) }
        lock.withLock { refused = target?.description ?? "an address with credentials or no valid host" }
        completionHandler(nil)
    }

    /// Throws when a redirect was refused, so the caller reports that rather than the 3xx it got.
    func check() throws {
        guard let target = lock.withLock({ refused }) else { return }
        throw MCPError.remote("The MCP server redirected to another origin (\(target)); nothing was sent there.")
    }
}
