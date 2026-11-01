// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import Network

/// An HTTP/1.1 server on 127.0.0.1 for the remote MCP tests: one request per connection,
/// every request recorded, and a reply may keep its connection open as an event stream.
final class LoopbackHTTPServer: @unchecked Sendable {
    struct Request {
        let method: String
        let path: String
        let headers: [String: String]
        let body: Data

        var json: [String: Any]? { try? JSONSerialization.jsonObject(with: body) as? [String: Any] }
    }

    enum Reply {
        case respond(status: Int, headers: [String: String], body: Data)
        /// Leaves the connection open after `first`, for `push`.
        case stream(first: Data)
    }

    private let listener: NWListener
    private let queue = DispatchQueue(label: "LoopbackHTTPServer")
    private let lock = NSLock()
    private var recorded: [Request] = []
    private var streams: [NWConnection] = []
    private var handler: (Request) -> Reply = { _ in .respond(status: 404, headers: [:], body: Data()) }

    init() throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
    }

    var requests: [Request] { lock.withLock { recorded } }

    /// Starts serving and gives the port.
    func start(_ handler: @escaping (Request) -> Reply) async throws -> Int {
        self.handler = handler
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            connection.start(queue: self.queue)
            self.receive(connection, Data())
        }
        return try await withCheckedThrowingContinuation { continuation in
            listener.stateUpdateHandler = { [listener] state in
                switch state {
                case .ready:
                    listener.stateUpdateHandler = nil
                    continuation.resume(returning: Int(listener.port?.rawValue ?? 0))
                case let .failed(error):
                    listener.stateUpdateHandler = nil
                    continuation.resume(throwing: error)
                default: break
                }
            }
            listener.start(queue: queue)
        }
    }

    /// Writes to every connection a `.stream` reply left open.
    func push(_ data: Data) {
        for connection in lock.withLock({ streams }) {
            connection.send(content: data, completion: .idempotent)
        }
    }

    func stop() {
        listener.cancel()
        for connection in lock.withLock({ streams }) { connection.cancel() }
    }

    private func receive(_ connection: NWConnection, _ buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, done, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let request = Self.parse(buffer) {
                self.answer(connection, request)
            } else if done || error != nil {
                connection.cancel()
            } else {
                self.receive(connection, buffer)
            }
        }
    }

    private func answer(_ connection: NWConnection, _ request: Request) {
        lock.withLock { recorded.append(request) }
        switch handler(request) {
        case let .respond(status, headers, body):
            var head = "HTTP/1.1 \(status) Reply\r\nContent-Length: \(body.count)\r\nConnection: close\r\n"
            for (name, value) in headers { head += "\(name): \(value)\r\n" }
            connection.send(content: Data((head + "\r\n").utf8) + body, completion: .contentProcessed { _ in
                connection.cancel()
            })
        case let .stream(first):
            lock.withLock { streams.append(connection) }
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
            connection.send(content: Data(head.utf8) + first, completion: .idempotent)
        }
    }

    private static func parse(_ buffer: Data) -> Request? {
        guard let end = buffer.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let lines = String(decoding: buffer[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
        let start = lines.first?.split(separator: " ").map(String.init) ?? []
        guard start.count >= 2 else { return nil }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let length = Int(headers["content-length"] ?? "") ?? 0
        let body = buffer[end.upperBound...]
        guard body.count >= length else { return nil }
        return Request(method: start[0], path: start[1], headers: headers, body: Data(body.prefix(length)))
    }
}
