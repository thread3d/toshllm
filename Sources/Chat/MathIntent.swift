// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// What a user message asks of the math tools. A computed result has to come from a tool;
/// an explanation does not.
enum MathIntent: String, Equatable, CaseIterable {
    case noMath = "no_math"
    case conceptual
    case computational
    case ambiguous

    /// The first round may only call a math tool or ask the user.
    var requiresTools: Bool { self == .computational || self == .ambiguous }

    /// What the text shows about the request, before any model reads it.
    struct Evidence: Equatable {
        var structure = false      // an expression, equation, matrix, data set...
        var concrete = false       // something definite to compute on: numbers, an expression
        var compute = false        // a verb that asks for a result
        var object = false         // a mathematical or scientific object
        var concept = false        // asks to explain or define
        var precision = false      // asks for digits or a tolerance
        var code = false           // a code snippet, usually not a calculation
    }

    /// The deterministic part of the decision, or nil when the text alone cannot tell.
    static func classify(_ text: String, context: [String] = []) -> (intent: MathIntent?, evidence: Evidence) {
        let evidence = read(text)
        let earlier = context.map(read).contains { $0.structure }
        let e = evidence
        if e.code && !e.compute && !e.precision { return (.noMath, e) }
        if !e.structure && !e.object && !e.precision {
            return (e.compute ? nil : .noMath, e)
        }
        if e.precision && (e.structure || e.object) { return (.computational, e) }
        if e.structure && e.compute && !e.concept { return (.computational, e) }
        if e.structure && !e.compute && !e.concept { return (strongStructure(text) ? .computational : nil, e) }
        if e.structure { return (nil, e) }
        // no expression: an object named in words
        if e.concept && !e.compute { return (e.concrete ? nil : .conceptual, e) }
        if e.compute {
            if e.concrete || earlier { return (nil, e) }
            return (.ambiguous, e)
        }
        return (nil, e)
    }

    // reading the text

    private static func regex(_ pattern: String) -> NSRegularExpression {
        try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }

    private static func found(_ pattern: NSRegularExpression, _ text: String) -> Bool {
        pattern.firstMatch(in: text, range: NSRange(location: 0, length: (text as NSString).length)) != nil
    }

    private static let latex = regex(#"\\(?:frac|dfrac|int|iint|oint|sum|prod|sqrt|lim|partial|nabla|infty|cdot|times|begin\{[pbvB]?matrix\}|left|right|mathrm|alpha|beta|gamma|lambda|mu|sigma|theta|omega|pi)(?![A-Za-z])|\$[^$\n]+\$"#)
    private static let unicode = regex(#"[∫∬∮∑∏√∞≤≥≠≈±∂∇∆×÷²³⁴⁵⁶⁷⁸⁹ⁿ₀₁₂₃∈∉⊂∪∩]"#)
    private static let equation = regex(#"(?:[A-Za-z]\w*(?:\([^()]*\))?|\d)\s*(?:'+\s*)?(?:=|<=|>=|<|>)\s*[-+(]?\s*(?:\d|[A-Za-z]\b)"#)
    private static let operation = regex(#"(?:\b\d+(?:[.,]\d+)?|\b[A-Za-z]\b|\))\s*(?:\^|\*\*|[*/+−-])\s*(?:\d|\b[A-Za-z]\b|\()|\b\d+[A-Za-z]\b(?!\w)"#)
    private static let function = regex(#"\b(?:sin|cos|tan|exp|log|ln|sqrt|abs|sinh|cosh|tanh|arctan|atan|erf|gamma|zeta)\s*\("#)
    private static let derivative = regex(#"\bd\^?\d?[A-Za-z]\s*/\s*d[A-Za-z]|\b[A-Za-z]'{1,3}\s*(?:\(|[-+=])|∂"#)
    private static let matrix = regex(#"\[\s*\[|\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)\s*,|\\begin\{"#)
    private static let list = regex(#"-?\d+(?:[.,]\d+)?(?:\s*[,;]\s*-?\d+(?:[.,]\d+)?){3,}"#)
    private static let number = regex(#"\d"#)
    private static let code = regex(#"```|\bdef\s+\w+\s*\(|\bfunction\s+\w+\s*\(|\breturn\b|\bprint\s*\(|\bimport\s+\w+|console\.log|=>|\bclass\s+\w+"#)
    private static let precision = regex(#"\b\d+\s*(?:decimal|decimales|cifras|d[ií]gitos|digits|significant|significativas|places)\b|\b(?:tolerance|tolerancia|precisi[oó]n|precision|accuracy)\s*(?:of|de|=|:)?\s*\d|\bto\s+\d+\s+(?:dp|sf)\b"#)

    // the stems are short on purpose: they match the forms of a verb in both languages
    private static let computeWords = regex(#"\b(?:calcul|comput|solv|resuelv|resolv|evalu|eval[uú]|integr|deriv|differentiat|diferenci|simplif|factori|expand|desarroll|halla|hallar|find|determin[ae]|obt[eé]n|obtain|verif|comprueb|check|aproxim|approximat|ajust|fit|regres|interpol|minimi|maximi|optimi|estim|cu[aá]nto|cu[aá]nta|how much|how many|what is the value|cu[aá]l es el valor|value of|valor de|result|resultado|give me|dame|dime|tell me|ra[ií]ces de|roots? of|zeros? of|ceros de|probabilidad de que|probability that)"#)
    private static let objectWords = regex(#"\b(?:ecuaci[oó]n diferencial|differential equation|integral|derivad|derivative|diferencial|ecuaci[oó]n|equation|sistema de|system of|matri[zx]|matrices|determinant|autovalor|eigen|valor(?:es)? propio|inversa|inverse|l[ií]mite|limit|serie|series|sumatori|ra[ií]z|root|fft|fourier|laplace|se[nñ]al|signal|filtro|filter|regresi[oó]n|regression|media|mean|mediana|median|promedio|average|desviaci[oó]n|deviation|varianza|variance|percentil|percentile|probabilidad|probability|distribuci[oó]n|distribution|edo|ode|ecuaci[oó]n diferencial|differential equation|transformada|transform|polinomio|polynomial|vector|producto escalar|dot product|cross product|producto vectorial|svd|valores singulares|singular value|m[ií]nimos cuadrados|least squares|t-test|prueba t|chi|correlaci[oó]n|correlation|interpolaci[oó]n|interpolation|gradiente|gradient|hessian|jacobian|optimizaci[oó]n|optimization|m[ií]nimo|minimum|m[aá]ximo|maximum|logaritmo|logarithm|seno|coseno|tangente|sine|cosine|tangent|pi\b|π|infinit)"#)
    private static let conceptWords = regex(#"\b(?:qu[eé] es|qu[eé] son|what is an?\b|what are|what's an?\b|expl[ií]ca|explain|por qu[eé]|why|c[oó]mo funciona|how does|how do|qu[eé] significa|what does .{1,40}(?:mean|represent)|qu[eé] representa|represent|diferencia entre|difference between|intuici[oó]n|intuition|para qu[eé] sirve|what is .{1,30} used for|useful|[uú]til|defin|concept|concepto|meaning|significado|historia|history|ejemplo de|example of|describe|describ)"#)

    static func read(_ text: String) -> Evidence {
        var e = Evidence()
        e.code = found(code, text)
        e.structure = [latex, unicode, equation, operation, function, derivative, matrix, list].contains { found($0, text) }
        e.concrete = found(number, text) || e.structure
            || found(regex(#"\b(?:x|y|z|t|n)\b"#), text)
        // "integral" names an object and "diferencia entre" asks for an explanation: neither asks to compute
        let range = { (t: String) in NSRange(location: 0, length: (t as NSString).length) }
        let nouns = objectWords.stringByReplacingMatches(in: text, range: range(text), withTemplate: " ")
        let verbs = conceptWords.stringByReplacingMatches(in: nouns, range: range(nouns), withTemplate: " ")
        e.compute = found(computeWords, verbs)
        e.object = found(objectWords, text)
        e.concept = found(conceptWords, text)
        e.precision = found(precision, text)
        return e
    }

    /// An integral sign, a LaTeX formula or an equation in a variable stands for a calculation
    /// even with no verb around it.
    private static func strongStructure(_ text: String) -> Bool {
        found(latex, text) || found(regex(#"[∫∑∏√]"#), text) || found(matrix, text)
            || (found(equation, text) && found(regex(#"\b[a-z]\b"#), text))
    }

    // asking the model when the text alone cannot tell

    static let instructions = "Classify the user's message for a math assistant. Answer with one word. computational: it asks for a specific mathematical or scientific result, such as a value, a solution, a simplification, a fit or a statistic, even inside prose or an explanation. conceptual: it asks to explain, define or compare ideas and needs no specific result. ambiguous: it asks for a calculation but leaves out what the calculation needs. no_math: anything else, including numbers that are not to be computed."
    static let grammar = #"root ::= "computational" | "conceptual" | "ambiguous" | "no_math""#

    static func ask(_ text: String, port: Int) async -> MathIntent? {
        var body: [String: Any] = [
            "messages": [["role": "system", "content": instructions],
                         ["role": "user", "content": String(text.prefix(4000)) + " /no_think"]],
            "max_tokens": 6, "temperature": 0, "grammar": grammar, "cache_prompt": false,
            "chat_template_kwargs": ["enable_thinking": false],
        ]
        if let model = ServerSettings.activeRouterModel() { body["model"] = model }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = ServerSettings.activeAPIKey() { request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return nil }
        request.httpBody = data
        guard let (reply, _) = try? await URLSession.shared.data(for: request),
              let object = try? JSONSerialization.jsonObject(with: reply) as? [String: Any],
              let message = (object["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any],
              let word = (message["content"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        else { return nil }
        return MathIntent(rawValue: word)
    }

    /// The deterministic answer, the model's when the text cannot tell, and a safe default when
    /// the model cannot be asked: with an expression in it, the request is treated as a calculation.
    static func decide(_ text: String, context: [String], port: Int) async -> (intent: MathIntent, asked: Bool) {
        let (intent, evidence) = classify(text, context: context)
        if let intent { return (intent, false) }
        if let answer = await ask(text, port: port) { return (answer, true) }
        return (evidence.structure ? .computational : .conceptual, true)
    }
}
