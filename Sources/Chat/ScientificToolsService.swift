// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Numerical tools backed by NumPy and SciPy. They share the bundled Python runtime with
/// the SymPy tools but run as their own MCP server, so each can be on without the other.
enum ScientificToolsService {
    static let serverName = "scientific"

    static var isEnabled: Bool {
        UserDefaults.standard.bool(forKey: SettingsKeys.scientificEnabled)
    }

    static func isTool(_ name: String) -> Bool {
        name.hasPrefix(serverName + "_")
    }

    /// Sent as system text when both math tool sets are offered: the model otherwise takes
    /// the numerical tool for a request that wanted an exact answer.
    static let routingRule = "Math tools: sympy_* is the default and gives exact results. Use scientific_* only if the user says numerical, approximate, decimal or floating point, or the task is data: statistics, fitting, interpolation, optimization, signals, matrices with decimals. A plain integral, sum, derivative, equation, limit or matrix of integers or symbols goes to sympy_*. Never answer an exact or unspecified request with an approximation."

    static func system(_ system: String, sympy: Bool = SymPyToolsService.isEnabled,
                       scientific: Bool = isEnabled) -> String {
        guard sympy, scientific else { return system }
        return (system.isEmpty ? "" : system + "\n\n") + routingRule
    }

    /// What the call was asked to work on, for the tool card.
    static func input(_ arguments: [String: Any]) -> String? {
        if let expression = arguments["expression"] as? String { return expression }
        if let equations = arguments["equations"] as? [Any] {
            return equations.map { String(describing: $0) }.joined(separator: "\n")
        }
        if let rows = arguments["matrix"] as? [[Any]] {
            guard rows.count <= 8 else { return "\(rows.count) × \(rows.first?.count ?? 0)" }
            return rows.map { $0.map { String(describing: $0) }.joined(separator: "  ") }
                .joined(separator: "\n")
        }
        for key in ["values", "x", "y"] {
            if let samples = arguments[key] as? [Any] { return "\(key): \(samples.count) values" }
        }
        return nil
    }

    /// The fields a person looks for first; the rest follow in alphabetical order.
    private static let leading = ["value", "root", "solution", "objective", "parameters", "determinant",
                                  "eigenvalues", "dominant_frequencies", "final", "mean", "statistic", "p_value"]
    private static let hidden: Set<String> = ["success", "operation", "warnings", "error", "timed_out",
                                              "interpreted_input", "result_kind", "reasons"]

    /// The JSON result as the few lines a person reads. The model still gets the JSON.
    static func readable(_ result: String) -> String {
        guard let data = result.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return result }
        let read = MathTranscriptionService.interpreted(object)
        if let error = object["error"] as? [String: Any], let message = error["message"] as? String {
            return ([message] + read).joined(separator: "\n")
        }
        let keys = leading.filter { object[$0] != nil }
            + object.keys.filter { !leading.contains($0) && !hidden.contains($0) }.sorted()
        var lines = keys.prefix(12).map { "\($0): \(compact(object[$0]!))" }
        for warning in object["warnings"] as? [String] ?? [] { lines.append("⚠︎ \(warning)") }
        return lines.isEmpty ? result : (lines + read).joined(separator: "\n")
    }

    private static func compact(_ value: Any) -> String {
        if let text = value as? String { return text }
        if let number = value as? NSNumber { return number.stringValue }
        guard JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        else { return String(describing: value) }
        let text = String(decoding: data, as: UTF8.self)
        return text.count > 160 ? String(text.prefix(160)) + "…" : text
    }
}
