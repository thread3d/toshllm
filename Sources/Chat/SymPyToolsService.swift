// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Symbolic math tools. The engine runs the bundled helper as an MCP server and lists its
/// tools on /tools, so the app only decides whether to start it and how to show a call.
enum SymPyToolsService {
    static let serverName = "sympy"

    static var isEnabled: Bool {
        UserDefaults.standard.bool(forKey: SettingsKeys.sympyEnabled)
    }

    static func isTool(_ name: String) -> Bool {
        name.hasPrefix(serverName + "_")
    }

    /// The runtime built by scripts/build-sympy.sh, or nil when this build does not carry it.
    static func runtimeDirectory(resources: URL? = Bundle.main.resourceURL) -> URL? {
        guard let directory = resources?.appendingPathComponent("tosh-sympy"),
              FileManager.default.isExecutableFile(
                atPath: directory.appendingPathComponent("python/bin/python3").path)
        else { return nil }
        return directory
    }

    /// Lets the helpers tell the app's own provenance fields from ones a client or a model wrote.
    /// New on every launch, and only the app and the helpers it starts know it.
    static let trustKey = UUID().uuidString + UUID().uuidString

    static let agentName = "agent"

    /// One MCP server per tool set that is switched on. Both run the same helper from the
    /// same runtime, the scientific one with its name as an argument. A third is the engine's
    /// agent, which the chat asks for by header; with `agent` it also answers API clients that
    /// do not ask, which otherwise get the model as it is.
    static func serverArguments(enabled: Bool, scientific: Bool = false, agent: Bool = false,
                                resources: URL? = Bundle.main.resourceURL, shared: [MCPServer] = [],
                                extensions: [EngineExtensions.Extension] = []) -> [String] {
        // approved servers exist only for the agent, so they need its runtime too
        var servers = engineServers(shared, runtime: runtimeDirectory(resources: resources))
        let approved = servers.keys.sorted()
        guard enabled || scientific || !approved.isEmpty || !extensions.isEmpty,
              let runtime = runtimeDirectory(resources: resources)
        else { return [] }
        let python = runtime.appendingPathComponent("python/bin/python3").path
        let helper = runtime.appendingPathComponent("tosh_sympy/server.py").path
        let bundled = EngineExtensions.engineEntries(extensions, runtime: runtime, trustKey: trustKey)
        let enabled = enabled || bundled.mathTools
        let scientific = scientific || bundled.mathTools
        if enabled {
            servers[serverName] = ["command": python, "args": ["-I", "-B", helper], "timeout_ms": 30_000,
                                   "env": ["TOSH_TRUST_KEY": trustKey]]
        }
        if scientific {
            servers[ScientificToolsService.serverName] = [
                "command": python, "args": ["-I", "-B", helper, ScientificToolsService.serverName],
                "timeout_ms": 30_000, "env": ["TOSH_TRUST_KEY": trustKey],
            ]
        }
        // a turn can take minutes: it runs every round and tool call of the answer
        servers[agentName] = ["command": python, "args": ["-I", "-B", helper, agentName],
                              "timeout_ms": 900_000,
                              "env": ["TOSH_TRUST_KEY": trustKey,
                                      "TOSH_AGENT_SERVERS": (approved + bundled.servers.keys.sorted()).joined(separator: ",")]]
        servers.merge(bundled.servers) { current, _ in current }
        if let replacement = bundled.agent { servers[agentName] = replacement }
        let agentOnly = approved + bundled.agentOnly
        let arguments = mcpServersArguments(servers)
        guard !arguments.isEmpty else { return [] }
        return arguments + ["--mcp-agent", agentName] + (agent ? [] : ["--mcp-agent-explicit"])
            + (agentOnly.isEmpty ? [] : ["--mcp-agent-only", agentOnly.joined(separator: ",")])
            + (bundled.clientTools ? ["--mcp-agent-client-tools"] : [])
    }

    /// The MCP servers an administrator approved for the engine, by the name the engine gives them.
    /// A name the engine's own servers use, or one already taken, is left out, and so is a remote
    /// server when there is no runtime to run its relay.
    static func engineServers(_ shared: [MCPServer], runtime: URL? = nil) -> [String: Any] {
        let reserved: Set = [serverName, ScientificToolsService.serverName, agentName]
        var servers: [String: Any] = [:]
        for server in shared where server.sharedWithEngine {
            let name = server.engineName
            guard !reserved.contains(name), servers[name] == nil else { continue }
            if !server.transport.isLocal {
                if let runtime { servers[name] = remoteEntry(server, runtime: runtime) }
                continue
            }
            var entry: [String: Any] = ["command": server.command, "args": server.arguments,
                                        "env": server.environment, "timeout_ms": server.timeoutSeconds * 1000]
            if !server.workingDirectory.isEmpty { entry["cwd"] = server.workingDirectory }
            servers[name] = entry
        }
        return servers
    }

    /// The bundled relay, run as a local server that forwards to the remote one.
    static func relayEntry(runtime: URL, url: String, transport: String?, timeoutSeconds: Int,
                           headerArguments: [String]) -> [String: Any] {
        let python = runtime.appendingPathComponent("python/bin/python3").path
        let relay = runtime.appendingPathComponent("tosh_sympy/remote.py").path
        var args = ["-I", "-B", relay, "--url", url, "--timeout", String(timeoutSeconds)]
        if let transport { args += ["--transport", transport] }
        return ["command": python, "args": args + headerArguments, "timeout_ms": timeoutSeconds * 1000]
    }

    private static func remoteEntry(_ server: MCPServer, runtime: URL) -> [String: Any] {
        let transport: String? = server.transport == .streamableHTTP ? "http"
            : server.transport == .serverSentEvents ? "sse" : nil
        let headers = (Keychain.get(server.credentialAccount) ?? "").isEmpty ? []
            : ["--headers-env", headersVariable(server)]
        return relayEntry(runtime: runtime, url: server.url, transport: transport,
                          timeoutSeconds: server.timeoutSeconds, headerArguments: headers)
    }

    /// The relay reads a server's headers from this variable of the engine's environment, so a token
    /// never shows on a command line.
    static func headersVariable(_ server: MCPServer) -> String {
        "TOSH_MCP_HEADERS_" + server.engineName.uppercased().replacingOccurrences(of: "-", with: "_")
    }

    /// The headers of every approved remote server, for the engine's environment.
    static func engineEnvironment(_ shared: [MCPServer]) -> [String: String] {
        var env: [String: String] = [:]
        for server in shared where server.sharedWithEngine && !server.transport.isLocal {
            if let headers = Keychain.get(server.credentialAccount), !headers.isEmpty {
                env[headersVariable(server)] = headers
            }
        }
        return env
    }

    private static func mcpServersArguments(_ servers: [String: Any]) -> [String] {
        guard let data = try? JSONSerialization.data(withJSONObject: ["mcpServers": servers],
                                                     options: [.sortedKeys]) else { return [] }
        return ["--mcp-servers-json", String(decoding: data, as: UTF8.self)]
    }

    /// What the call was asked to work on, for the tool card.
    static func input(_ arguments: [String: Any]) -> String? {
        if let expression = arguments["expression"] as? String { return expression }
        if let equations = arguments["equations"] as? [Any] {
            return equations.map { String(describing: $0) }.joined(separator: "\n")
        }
        if let rows = arguments["matrix"] as? [[Any]] {
            return rows.map { $0.map { String(describing: $0) }.joined(separator: "  ") }
                .joined(separator: "\n")
        }
        if let left = arguments["left"] as? String, let right = arguments["right"] as? String {
            return left + "\n" + right
        }
        return nil
    }

    /// The JSON result as the few lines a person reads. The model still gets the JSON.
    static func readable(_ result: String) -> String {
        guard let data = result.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return result }
        var lines: [String] = []
        let read = MathTranscriptionService.interpreted(object)
        if let error = object["error"] as? [String: Any], let message = error["message"] as? String {
            // a timeout or an open integral still says what was asked and what is known
            guard object["timed_out"] != nil else { return ([message] + read).joined(separator: "\n") }
            lines.append(message)
            if let unevaluated = object["unevaluated"] as? String { lines.append(unevaluated) }
            if let partial = object["partial"] as? [String: Any], let exact = partial["exact"] as? String {
                lines.append(exact)
            }
            return lines.joined(separator: "\n")
        }
        if let equivalent = object["equivalent"] {
            lines.append("equivalent: \(text(equivalent))")
            if let difference = object["difference"] as? String, difference != "0" {
                lines.append("difference: \(difference)")
            }
        }
        if let satisfied = object["satisfied"] {
            lines.append("satisfied: \(text(satisfied))")
            for check in object["checks"] as? [[String: Any]] ?? [] {
                if let residual = check["residual"] as? String, residual != "0" {
                    lines.append("residual: \(residual)")
                }
            }
        }
        if let exact = object["exact"] as? String {
            lines.append(exact)
        } else if let unevaluated = object["unevaluated"] as? String {
            lines.append(unevaluated)
        }
        if let numeric = object["numeric"] as? String { lines.append("≈ \(numeric)") }
        for warning in object["warnings"] as? [String] ?? [] { lines.append("⚠︎ \(warning)") }
        return lines.isEmpty ? result : (lines + read).joined(separator: "\n")
    }

    private static func text(_ value: Any) -> String {
        if value is NSNull { return "undecided" }
        if let flag = value as? Bool { return flag ? "yes" : "no" }
        return String(describing: value)
    }
}
