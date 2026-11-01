// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// Nothing a remote MCP server announces or redirects to may take a request to another origin.
final class MCPRemoteOriginTests: XCTestCase {
    private var previous: Data?
    private var listeners: [LoopbackHTTPServer] = []
    private var configured: [MCPServer] = []

    override func setUp() {
        previous = UserDefaults.standard.data(forKey: SettingsKeys.mcpServers)
    }

    override func tearDown() async throws {
        for server in configured { await ToshMCPService.shared.disconnect(server.id) }
        for listener in listeners { listener.stop() }
        UserDefaults.standard.set(previous, forKey: SettingsKeys.mcpServers)
    }

    private func serve(_ handler: @escaping (LoopbackHTTPServer.Request) -> LoopbackHTTPServer.Reply)
        async throws -> (LoopbackHTTPServer, String) {
        let listener = try LoopbackHTTPServer()
        listeners.append(listener)
        let port = try await listener.start(handler)
        return (listener, "127.0.0.1:\(port)")
    }

    /// Answers whatever reaches it as an MCP server would, so a request sent there shows.
    private func sink() async throws -> (LoopbackHTTPServer, String) {
        try await serve { request in
            .respond(status: 200, headers: ["Content-Type": "application/json"], body: Self.answer(to: request) ?? Data())
        }
    }

    private static func answer(to request: LoopbackHTTPServer.Request) -> Data? {
        guard let message = request.json, let id = message["id"] else { return nil }
        let result: [String: Any] = message["method"] as? String == "tools/list"
            ? ["tools": [["name": "echo", "inputSchema": ["type": "object"]]]]
            : ["protocolVersion": "2025-06-18", "capabilities": [String: Any]()]
        return try? JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "id": id, "result": result])
    }

    private static func event(_ name: String, _ data: String) -> Data {
        Data("event: \(name)\ndata: \(data)\n\n".utf8)
    }

    /// The tools found on `server`, failing instead of hanging when its connection never settles.
    private func discover(_ server: MCPServer) async -> [String] {
        configured.append(server)
        MCPServerStore.save([server])
        let found = Found()
        let done = expectation(description: "discovery of \(server.url) returned")
        Task {
            found.names = await ToshMCPService.shared.discoverTools().map(\.name)
            done.fulfill()
        }
        await fulfillment(of: [done], timeout: 10)
        return found.names
    }

    func testARedirectedRequestNeverReachesAnotherOrigin() async throws {
        for code in [301, 302, 303, 307, 308] {
            let (sink, sinkHost) = try await sink()
            let (_, origin) = try await serve { _ in
                .respond(status: code, headers: ["Location": "http://\(sinkHost)/mcp"], body: Data())
            }
            _ = await discover(MCPServer(name: "redirect", url: "http://\(origin)/mcp", transport: .streamableHTTP))
            XCTAssertEqual(sink.requests.map(\.method), [], "HTTP \(code)")
        }
    }

    func testAnAnnouncedEndpointOnAnotherOriginReceivesNothing() async throws {
        for form in ["http://localhost:%@/messages", "http://127.0.0.1:%@/messages",
                     "//localhost:%@/messages", "http://127.0.0.1:1@localhost:%@/messages"] {
            let (sink, sinkHost) = try await sink()
            let endpoint = String(format: form, String(sinkHost.split(separator: ":")[1]))
            let (_, origin) = try await serve { request in
                request.method == "GET" ? .stream(first: Self.event("endpoint", endpoint))
                    : .respond(status: 202, headers: [:], body: Data())
            }
            _ = await discover(MCPServer(name: "legacy", url: "http://\(origin)/sse", transport: .serverSentEvents))
            XCTAssertEqual(sink.requests.map(\.method), [], endpoint)
        }
    }

    func testARedirectedEventStreamNeverReachesAnotherOrigin() async throws {
        let (sink, sinkHost) = try await sink()
        let (_, origin) = try await serve { _ in
            .respond(status: 302, headers: ["Location": "http://\(sinkHost)/sse"], body: Data())
        }
        _ = await discover(MCPServer(name: "legacy", url: "http://\(origin)/sse", transport: .serverSentEvents))
        XCTAssertEqual(sink.requests.map(\.method), [])
    }

    func testARedirectedMessageOfTheOlderTransportNeverReachesAnotherOrigin() async throws {
        let (sink, sinkHost) = try await sink()
        let (_, origin) = try await serve { request in
            request.method == "GET" ? .stream(first: Self.event("endpoint", "/messages"))
                : .respond(status: 307, headers: ["Location": "http://\(sinkHost)/messages"], body: Data())
        }
        _ = await discover(MCPServer(name: "legacy", url: "http://\(origin)/sse", transport: .serverSentEvents))
        XCTAssertEqual(sink.requests.map(\.method), [])
    }

    func testARedirectedWebSocketHandshakeNeverReachesAnotherOrigin() async throws {
        let (sink, sinkHost) = try await sink()
        let (_, origin) = try await serve { _ in
            .respond(status: 307, headers: ["Location": "http://\(sinkHost)/ws"], body: Data())
        }
        _ = await discover(MCPServer(name: "socket", url: "ws://\(origin)/ws", transport: .webSocket))
        XCTAssertEqual(sink.requests.map(\.method), [])
    }

    func testAnOriginIsTheSchemeHostAndEffectivePort() throws {
        let base = try XCTUnwrap(URL(string: "https://Example.com/sse"))
        XCTAssertEqual(MCPOrigin(base), MCPOrigin(try XCTUnwrap(URL(string: "https://example.COM:443/x"))))
        XCTAssertEqual(MCPOrigin(base), MCPOrigin(try XCTUnwrap(URL(string: "wss://example.com/ws"))))
        XCTAssertEqual(MCPOrigin.endpoint("/messages?s=1", of: base)?.absoluteString, "https://Example.com/messages?s=1")
        for other in ["http://example.com/messages", "https://example.com:8443/messages", "https://example.org/messages",
                      "https://user:pw@example.com/messages", "//example.org/messages", "ftp://example.com/messages"] {
            XCTAssertNil(MCPOrigin.endpoint(other, of: base), other)
        }
    }

    func testARefusedRedirectCarriesNoCredentialsOrBody() async throws {
        let (sink, sinkHost) = try await sink()
        let (_, origin) = try await serve { _ in
            .respond(status: 307, headers: ["Location": "http://\(sinkHost)/mcp"], body: Data())
        }
        let url = try XCTUnwrap(URL(string: "http://\(origin)/mcp"))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        request.setValue("key", forHTTPHeaderField: "X-Api-Key")
        request.httpBody = Data(#"{"jsonrpc":"2.0","id":1,"method":"tools/call"}"#.utf8)
        let redirects = MCPRedirectGuard(url)
        _ = try await URLSession.shared.data(for: request, delegate: redirects)
        XCTAssertThrowsError(try redirects.check())
        XCTAssertEqual(sink.requests.map(\.method), [])
    }

    func testTheOlderTransportStillPostsToARelativeEndpointOnItsOrigin() async throws {
        let listener = try LoopbackHTTPServer()
        listeners.append(listener)
        let port = try await listener.start { [weak listener] request in
            if request.method == "GET" { return .stream(first: Self.event("endpoint", "/messages?session=1")) }
            if let answer = Self.answer(to: request) { listener?.push(Data("data: ".utf8) + answer + Data("\n\n".utf8)) }
            return .respond(status: 202, headers: [:], body: Data())
        }
        let names = await discover(MCPServer(name: "legacy", url: "http://127.0.0.1:\(port)/sse",
                                             transport: .serverSentEvents))
        XCTAssertEqual(names, ["mcp_legacy_echo"])
        XCTAssertEqual(listener.requests.filter { $0.path == "/messages?session=1" }.count, 3)
    }

    func testARedirectWithinTheOriginIsStillFollowed() async throws {
        let (_, origin) = try await serve { request in
            if request.path == "/mcp" { return .respond(status: 307, headers: ["Location": "/mcp/"], body: Data()) }
            guard let answer = Self.answer(to: request) else { return .respond(status: 202, headers: [:], body: Data()) }
            return .respond(status: 200, headers: ["Content-Type": "application/json"], body: answer)
        }
        let names = await discover(MCPServer(name: "moved", url: "http://\(origin)/mcp", transport: .streamableHTTP))
        XCTAssertEqual(names, ["mcp_moved_echo"])
    }
}

private final class Found: @unchecked Sendable {
    var names: [String] = []
}
