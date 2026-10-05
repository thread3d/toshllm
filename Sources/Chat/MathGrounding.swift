// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// A math call that ran and gave a result. With the user's own messages, these are the only
/// places the final answer of a math turn may take a number from.
struct ValidatedMathResult: Equatable {
    let tool: String
    let operation: String
    let input: [String]
    let exact: Bool
    let result: [String]
    let reply: String
}

enum MathLedger {
    static func turnStart(_ messages: [ChatMessage]) -> Int {
        (messages.lastIndex { $0.role == "user" } ?? -1) + 1
    }

    static func calls(_ messages: [ChatMessage], from start: Int = 0) -> [ChatToolCall] {
        messages[min(start, messages.count)...].flatMap { $0.toolCalls ?? [] }
            .filter { MathTranscriptionService.isMathTool($0.name) }
    }

    static func results(_ messages: [ChatMessage], from start: Int = 0) -> [ValidatedMathResult] {
        calls(messages, from: start).compactMap(result)
    }

    private static let shown = ["exact", "numeric", "value", "root", "roots", "solution", "solutions", "objective",
                                "parameters", "determinant", "eigenvalues", "dominant_frequencies", "final", "mean",
                                "statistic", "p_value", "equivalent", "satisfied", "error_estimate", "residual"]

    static func result(_ call: ChatToolCall) -> ValidatedMathResult? {
        guard MathTranscriptionService.succeeded(call), let text = call.result,
              let reply = MathTranscriptionService.reply(text) else { return nil }
        let fields = shown.compactMap { key -> String? in
            guard let value = reply[key], !(value is NSNull) else { return nil }
            return "\(key): \(compact(value))"
        }
        return ValidatedMathResult(tool: call.name, operation: reply["operation"] as? String ?? "",
                                   input: reply["interpreted_input"] as? [String] ?? [],
                                   exact: reply["result_kind"] as? String == "exact", result: fields, reply: text)
    }

    private static func compact(_ value: Any) -> String {
        if let text = value as? String { return text }
        if let number = value as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
            // the shortest form, as the helper wrote it: stringValue turns 8.29141347594e-15 into 8.291413475939999e-15
            return CFNumberIsFloatType(number) ? "\(number.doubleValue)" : number.stringValue
        }
        guard JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        else { return String(describing: value) }
        let text = String(decoding: data, as: UTF8.self)
        return text.count > 240 ? String(text.prefix(240)) + "…" : text
    }
}

/// What becomes of a round that ended without tool calls in a turn that used math tools.
enum MathFinalStep: Equatable {
    case keep
    case replace(String)
    /// Drop the round and run another, tools still on, with this note to the model.
    case again(String)
    /// Drop the round and run a last one without tools, with this note to the model.
    case finalize(String)
}

enum MathGrounding {
    /// The numbers and named constants of an answer that neither the user's messages nor a validated
    /// result state. A number may be rounded or cut to fewer digits; any other number is new.
    static func ungrounded(_ answer: String, sources: [String], results: [ValidatedMathResult]) -> [String] {
        let known = sources + results.map(\.reply)
        let values = known.flatMap { numbers(in: $0) }.map(\.value)
        var missing: [String] = []
        for literal in numbers(in: withoutListMarkers(answer)) where !grounded(literal, by: values) {
            if !missing.contains(literal.text) { missing.append(literal.text) }
        }
        let named = Set(known.flatMap(constants))
        for name in constants(in: answer) where !named.contains(name) && !missing.contains(name) {
            missing.append(name)
        }
        return missing
    }

    /// `required`: the user asked for a computed result, so the answer is checked even when the
    /// model called no math tool.
    static func step(messages: [ChatMessage], answer: String, closing: String?,
                     finalizing: Bool, regrounded: Bool, required: Bool = false) -> MathFinalStep {
        let start = MathLedger.turnStart(messages)
        let calls = MathLedger.calls(messages, from: start)
        guard !calls.isEmpty || required else { return .keep }
        let turn = MathLedger.results(messages, from: start)
        if closing != nil {
            return turn.isEmpty || finalizing ? .keep : .finalize(finalNote(turn))
        }
        if finalizing && answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .replace(safeAnswer(results: turn, calls: calls))
        }
        let all = MathLedger.results(messages)
        let sources = messages.filter { $0.role == "user" }.map(\.wireContent)
        let missing = ungrounded(answer, sources: sources, results: all)
        if missing.isEmpty { return .keep }
        if finalizing || regrounded { return .replace(safeAnswer(results: turn, calls: calls)) }
        return .again(retryNote(missing, all))
    }

    /// The turn ends after a refused call: with results in hand, a last round states them.
    static func stopStep(messages: [ChatMessage]) -> MathFinalStep {
        let turn = MathLedger.results(messages, from: MathLedger.turnStart(messages))
        return turn.isEmpty ? .keep : .finalize(finalNote(turn))
    }

    static func finalNote(_ results: [ValidatedMathResult]) -> String {
        "Some math calls of this turn were refused or gave no result, and no more tools can be called. Write the final "
            + "answer now from the validated results below: state them, then name in words, without any formula or "
            + "number, the parts of the request that could not be validated with the tools. Do not work those parts "
            + "out yourself, and do not state any number, constant or formula that is not in the request or in these "
            + "results.\n\nValidated results:\n"
            + listing(results)
    }

    /// Sent on the rounds after a math call, so the answer needs no second pass in the usual case.
    static let standingNote = "Answer in prose with the values the tools returned, written as they returned them; "
        + "do not paste the raw tool output. Do not add digits, constants, closed forms or intermediate steps that no "
        + "tool returned; say instead what was not computed."

    static func retryNote(_ missing: [String], _ results: [ValidatedMathResult]) -> String {
        "Your answer states values that are not in the request or in a validated tool result: "
            + missing.prefix(8).joined(separator: ", ")
            + ". Leave them out and say they could not be validated; only if the request itself gives everything a "
            + "calculation needs, compute it with a math tool instead. Never work them out yourself. Then give the "
            + "answer again.\n\nValidated results:\n" + listing(results)
    }

    /// The same call made again adds nothing to say.
    private static func distinct(_ results: [ValidatedMathResult]) -> [ValidatedMathResult] {
        results.enumerated().filter { index, result in
            !results[..<index].contains { $0.tool == result.tool && $0.operation == result.operation
                && $0.result == result.result && $0.input == result.input }
        }.map(\.element)
    }

    private static func listing(_ results: [ValidatedMathResult]) -> String {
        guard !results.isEmpty else { return "none" }
        return distinct(results).map { result in
            "- \(result.tool) \(result.operation) (\(result.exact ? "exact" : "approximate")): "
                + result.result.joined(separator: "; ")
                + (result.input.isEmpty ? "" : ". Input: " + result.input.joined(separator: "; "))
        }.joined(separator: "\n")
    }

    /// The answer Tosh writes itself when the model's own cannot be shown: the validated results and
    /// what is left open, nothing else.
    static func safeAnswer(results: [ValidatedMathResult], calls: [ChatToolCall]) -> String {
        guard !results.isEmpty else { return MathTranscriptionService.unresolvedMessage() }
        let loc = Localizer()
        var lines = [loc.t("Esto es lo que se pudo validar con las herramientas:", "This is what the tools validated:"), ""]
        for result in distinct(results) {
            let kind = result.exact ? loc.t("exacto", "exact") : loc.t("aproximado", "approximate")
            let title = result.operation.prefix(1).uppercased() + result.operation.dropFirst()
            lines.append("- **\(title)** (\(kind)): " + result.result.joined(separator: " · "))
            if !result.input.isEmpty { lines.append("  ▸ " + result.input.joined(separator: " · ")) }
        }
        lines.append("")
        if !results.contains(where: \.exact) {
            lines.append(loc.t("Ningún resultado exacto se pudo validar: los valores de arriba son aproximaciones numéricas.",
                               "No exact result could be validated: the values above are numerical approximations."))
        }
        let replies = calls.compactMap { $0.result.flatMap(MathTranscriptionService.reply) }
        var open: [String] = []
        for reply in replies where MathTranscriptionService.errorCode(reply) == "no_closed_form" {
            if let text = reply["unevaluated"] as? String, !open.contains(text) { open.append(text) }
        }
        for text in open {
            lines.append(loc.t("No se obtuvo una forma exacta: SymPy no encontró forma cerrada para `%@`.",
                               "No exact form was obtained: SymPy found no closed form for `%@`.", text))
        }
        if replies.contains(where: { MathTranscriptionService.errorCode($0).map(MathTranscriptionService.rejectionCodes.contains) == true }) {
            lines.append(loc.t("Algunos cálculos no se hicieron porque no correspondían a la petición.",
                               "Some calculations were not run because they did not match the request."))
        }
        lines.append(loc.t("El resto de lo pedido no se pudo validar con las herramientas disponibles, y no se completa de memoria.",
                           "The rest of the request could not be validated with the available tools, and is not filled in from memory."))
        return lines.joined(separator: "\n")
    }

    // reading numbers

    struct Literal: Equatable {
        let text: String
        let value: Double
        let digits: Int
        let power: Bool
    }

    private static let superscripts: [Character: Character] = [
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁻": "-", "⁺": "+",
    ]

    private static let numberPattern = try! NSRegularExpression(pattern:
        #"(\d+(?:[.,]\d+)?)\s*(?:\\times|\\cdot|×|·|\*)\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?"#
        + #"|10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?"#
        + #"|(\d+(?:\.\d+)?)[eE]([-+]?\d+)"#
        + #"|\d+(?:[.,]\d+)?"#)

    static func numbers(in text: String) -> [Literal] {
        var plain = ""
        var raised = false
        let spaced = text.replacingOccurrences(of: "−", with: "-")
            .replacingOccurrences(of: #"\\[,;:! ]|~"#, with: " ", options: .regularExpression)
        for character in spaced {
            if let digit = superscripts[character] {
                if !raised { plain.append("^") }
                plain.append(digit)
                raised = true
            } else {
                plain.append(character)
                raised = false
            }
        }
        let source = plain as NSString
        return numberPattern.matches(in: plain, range: NSRange(location: 0, length: source.length)).compactMap { match in
            let whole = source.substring(with: match.range)
            func group(_ index: Int) -> String? {
                let range = match.range(at: index)
                return range.location == NSNotFound ? nil : source.substring(with: range)
            }
            if let mantissa = group(1), let exponent = group(2).flatMap(Double.init) {
                guard let m = decimal(mantissa) else { return nil }
                return Literal(text: whole, value: m * pow(10, exponent), digits: significant(mantissa), power: false)
            }
            if let exponent = group(3).flatMap(Double.init) {
                return Literal(text: whole, value: pow(10, exponent), digits: 1, power: true)
            }
            if let mantissa = group(4), let exponent = group(5).flatMap(Double.init), let m = decimal(mantissa) {
                return Literal(text: whole, value: m * pow(10, exponent), digits: significant(mantissa), power: false)
            }
            return decimal(whole).map { Literal(text: whole, value: $0, digits: significant(whole), power: false) }
        }
    }

    private static func decimal(_ text: String) -> Double? {
        Double(text.replacingOccurrences(of: ",", with: "."))
    }

    private static func significant(_ text: String) -> Int {
        let digits = text.filter(\.isNumber).drop { $0 == "0" }
        return max(1, digits.count)
    }

    /// A number stated by a source, or that source's number rounded or cut to fewer digits. A short
    /// integer such as the 6 of 6ζ(4) is not taken for a rounding: it has to be stated as it is.
    static func grounded(_ literal: Literal, by values: [Double]) -> Bool {
        let x = abs(literal.value)
        let rounds = literal.digits >= 3 || literal.text.contains { ".,eE^×*".contains($0) }
        for value in values.map(abs) where value.isFinite {
            if x == value { return true }
            guard rounds, x > 0, value > 0 else { continue }
            let exponent = floor(log10(value))
            if literal.power {
                // "of the order of 10^-11" for 9.3e-11
                let k = log10(x).rounded()
                if k == exponent || k == exponent + 1 { return true }
                continue
            }
            let unit = pow(10, exponent - Double(literal.digits) + 1)
            let rounded = (value / unit).rounded() * unit
            let cut = (value / unit).rounded(.towardZero) * unit
            if abs(x - rounded) <= unit * 1e-3 || abs(x - cut) <= unit * 1e-3 { return true }
        }
        return false
    }

    private static let listMarker = try! NSRegularExpression(
        pattern: #"(?m)^[ \t]*(?:[-*+>][ \t]*)?(?:#{1,6}[ \t]*)?(?:\*\*|__)?\(?\d{1,2}[.)](?=\s|\*|_)"#)

    static func withoutListMarkers(_ text: String) -> String {
        listMarker.stringByReplacingMatches(in: text, range: NSRange(location: 0, length: (text as NSString).length),
                                            withTemplate: " ")
    }

    // named constants a model tends to bring from memory

    private static let constantPatterns: [(String, NSRegularExpression)] = [
        ("π", #"π|\\pi(?![A-Za-z])|(?<![A-Za-z])pi(?![A-Za-z])"#),
        ("ζ", #"ζ|\\zeta(?![A-Za-z])|(?<![A-Za-z])zeta\s*\("#),
        ("Γ", #"Γ|\\Gamma(?![A-Za-z])|(?<![A-Za-z])[Gg]amma\s*\("#),
        ("γ", #"γ|\\gamma(?![A-Za-z])|EulerGamma"#),
        ("Catalan", #"(?<![A-Za-z])Catalan(?![A-Za-z])"#),
        ("Li", #"\\operatorname\{Li\}|(?<![A-Za-z])Li_|polylog\s*\("#),
        ("erf", #"(?<![A-Za-z])erfc?\s*\(|\\operatorname\{erfc?\}"#),
    ].map { ($0.0, try! NSRegularExpression(pattern: $0.1)) }

    static func constants(in text: String) -> [String] {
        let range = NSRange(location: 0, length: (text as NSString).length)
        return constantPatterns.filter { $0.1.firstMatch(in: text, range: range) != nil }.map(\.0)
    }
}
