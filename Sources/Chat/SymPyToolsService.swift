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
                                resources: URL? = Bundle.main.resourceURL) -> [String] {
        guard enabled || scientific, let runtime = runtimeDirectory(resources: resources) else { return [] }
        let python = runtime.appendingPathComponent("python/bin/python3").path
        let helper = runtime.appendingPathComponent("tosh_sympy/server.py").path
        var servers: [String: Any] = [:]
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
                              "timeout_ms": 900_000, "env": ["TOSH_TRUST_KEY": trustKey]]
        guard let data = try? JSONSerialization.data(withJSONObject: ["mcpServers": servers],
                                                     options: [.sortedKeys]) else { return [] }
        return ["--mcp-servers-json", String(decoding: data, as: UTF8.self), "--mcp-agent", agentName]
            + (agent ? [] : ["--mcp-agent-explicit"])
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
