// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import JavaScriptCore
import SwiftUI
import WebKit
import XCTest
@testable import ToshLLM

enum MarkdownMathFixture {
    static let inverse = #"""
    $A^{-1} = \begin{pmatrix} \frac{1}{84} & -\frac{4}{84} & \frac{14}{84} \\ \frac{12}{84} & \frac{36}{84} & -\frac{42}{84} \\ -\frac{7}{84} & \frac{7}{84} & \frac{7}{84} \end{pmatrix}$
    """#

    static let document = #"""
    1. First step

    2. Compute $A$

    3. The determinant is $84$.

    4. Continue with the calculations.

    El determinante es $84$.

    Tenemos $A$ y $B$.

    \#(inverse)

    Precio: $10

    Precio entre $10 y $25

    Valores: $123$, $3.14159$, $-4$ y $10^{12}$.

    Inline: $x$, $x^2 + y^2$ y $A^{-1}$.
    """#
}

final class MarkdownParserTests: XCTestCase {
    private func parse(_ text: String) -> [MDBlock] { RichText.parse(text) }

    func testFixtureBlocks() {
        let blocks = parse(MarkdownMathFixture.document)
        XCTAssertEqual(Array(blocks.prefix(4)), [
            .numbered(1, ["First step"]),
            .numbered(2, ["Compute $A$"]),
            .numbered(3, ["The determinant is $84$."]),
            .numbered(4, ["Continue with the calculations."]),
        ])
        XCTAssertEqual(blocks[4], .paragraph("El determinante es $84$."))
        guard case .math(let formula) = blocks[6] else { return XCTFail("\(blocks[6])") }
        XCTAssertTrue(formula.hasPrefix(#"A^{-1} = \begin{pmatrix}"#))
        XCTAssertEqual(blocks[7], .paragraph("Precio: $10"))
        XCTAssertEqual(blocks[8], .paragraph("Precio entre $10 y $25"))
    }

    func testNumberedListsKeepTheirSourceNumbers() {
        XCTAssertEqual(parse("1. one\n\nparagraph\n\n2. two\n\nparagraph\n\n3. three"), [
            .numbered(1, ["one"]), .paragraph("paragraph"),
            .numbered(2, ["two"]), .paragraph("paragraph"),
            .numbered(3, ["three"]),
        ])
        XCTAssertEqual(parse("5. five\n6. six"), [.numbered(5, ["five", "six"])])
        XCTAssertEqual(parse("1) a\n2) b"), [.numbered(1, ["a", "b"])])
    }

    func testListContinuationIndentIsDropped() {
        let text = "3. The inverse is:\n\n   \(MarkdownMathFixture.inverse)"
        let blocks = parse(text)
        XCTAssertEqual(blocks.count, 2)
        guard case .math = blocks[1] else { return XCTFail("\(blocks[1])") }
        XCTAssertEqual(parse("- item\n  more text"), [.bullet(["item"]), .paragraph("more text")])
        // Outside a list, indentation stays as written.
        XCTAssertEqual(parse("text\n\n    indented"), [.paragraph("text"), .paragraph("    indented")])
    }

    /// Shape of a real agent answer: indented continuation lines under each
    /// item, and the inverse on its own line under a sentence.
    func testAgentAnswerLayout() {
        let text = """
        1.  **Determinante de $A$**:
            El determinante de la matriz $A$ es $84$.

        2.  **Inversa de $A$**:
            La inversa exacta de la matriz es:
        \(MarkdownMathFixture.inverse.split(separator: "\n").map { "    " + $0 }.joined(separator: "\n"))

        3.  **Verificación**:
        """
        let blocks = parse(text)
        XCTAssertEqual(Array(blocks.prefix(4)), [
            .numbered(1, ["**Determinante de $A$**:"]),
            .paragraph("El determinante de la matriz $A$ es $84$."),
            .numbered(2, ["**Inversa de $A$**:"]),
            .paragraph("La inversa exacta de la matriz es:"),
        ])
        guard blocks.count == 6, case .math = blocks[4] else { return XCTFail("\(blocks)") }
        XCTAssertEqual(blocks[5], .numbered(3, ["**Verificación**:"]))
    }

    func testHeadingKeepsItsFormula() {
        XCTAssertEqual(parse("### 1) Determinante de $A$"), [.header(3, "1) Determinante de $A$")])
        XCTAssertEqual(RichText.inlineMathBodies("3) Verificación ($A \\cdot A^{-1} = I$)"), ["A \\cdot A^{-1} = I"])
        XCTAssertTrue(RichWebView.html(source: "$A$", kind: .inlineMath, fontSize: 15, bold: true)
            .contains("font:bold 15.0px"))
    }

    func testCodeIsNotTouched() {
        XCTAssertEqual(parse("1. step\n\n   ```swift\n   let x = $84$\n   ```"),
                       [.numbered(1, ["step"]), .code("swift", "   let x = $84$")])
        XCTAssertEqual(parse("```\n$A^{-1} = \\begin{pmatrix} 1 \\end{pmatrix}$\n```"),
                       [.code("", "$A^{-1} = \\begin{pmatrix} 1 \\end{pmatrix}$")])
    }

    func testStandaloneMath() {
        guard case .math(let f) = parse(MarkdownMathFixture.inverse).first else { return XCTFail() }
        XCTAssertTrue(f.hasSuffix(#"\end{pmatrix}"#))
        let multiline = "$M = \\begin{bmatrix}\n1 & 2 \\\\\n3 & 4\n\\end{bmatrix}$"
        XCTAssertEqual(parse(multiline), [.math("M = \\begin{bmatrix}\n1 & 2 \\\\\n3 & 4\n\\end{bmatrix}")])
        XCTAssertEqual(parse("\\(\\begin{cases} x & x > 0 \\\\ 0 & \\text{else} \\end{cases}\\)"),
                       [.math("\\begin{cases} x & x > 0 \\\\ 0 & \\text{else} \\end{cases}")])
        // A short standalone formula stays inline.
        XCTAssertEqual(parse("$x^2$"), [.paragraph("$x^2$")])
        XCTAssertEqual(parse("$10 y $25$"), [.paragraph("$10 y $25$")])
    }

    func testBlockMath() {
        XCTAssertEqual(parse("$$\\frac{1}{2}$$"), [.math("\\frac{1}{2}")])
        XCTAssertEqual(parse("$$\nx^2 + y^2\n$$"), [.math("x^2 + y^2")])
        XCTAssertEqual(parse("\\[\nE = mc^2\n\\]"), [.math("E = mc^2")])
        XCTAssertEqual(parse("\\[ E = mc^2 \\]"), [.math(" E = mc^2 ")])
        XCTAssertEqual(parse("$$\\begin{pmatrix}\n1 & 2 \\\\\n3 & 4\n\\end{pmatrix}$$\n\nafter"),
                       [.math("\\begin{pmatrix}\n1 & 2 \\\\\n3 & 4\n\\end{pmatrix}"), .paragraph("after")])
    }

    func testOrdinaryMarkdown() {
        let text = """
        # Title
        ## Sub

        A **bold** and *italic* [link](https://example.com) with `code`.

        - one
        - two
          - nested

        > quoted
        > text

        | a | b |
        |---|---|
        | 1 | 2 |

        ---
        """
        XCTAssertEqual(parse(text), [
            .header(1, "Title"), .header(2, "Sub"),
            .paragraph("A **bold** and *italic* [link](https://example.com) with `code`."),
            .bullet(["one", "two", "nested"]),
            .quote("quoted\ntext"),
            .table(["a", "b"], [["1", "2"]]),
            .rule,
        ])
    }
}

final class InlineMathRuleTests: XCTestCase {
    func testNumbersAreMath() {
        for (text, body) in [("El determinante es $84$.", "84"), ("$123$", "123"),
                             ("pi = $3.14159$", "3.14159"), ("$-4$", "-4"), ("$10^{12}$", "10^{12}"),
                             ("el factor $1/84$ aparte", "1/84"), ("$2 + 2 = 4$", "2 + 2 = 4")] {
            XCTAssertEqual(RichText.inlineMathBodies(text), [body], text)
        }
    }

    func testCurrencyIsText() {
        for text in ["Precio: $10", "Precio entre $10 y $25", "$25 USD", "from $10 to $25",
                     "Cuesta entre $10-$25 al mes.", "US$10 y US$20$", "$10, $20 y $30", "$10 - $25", "$10/$20 al mes"] {
            XCTAssertEqual(RichText.inlineMathBodies(text), [], text)
        }
    }

    func testInlineFormulas() {
        XCTAssertEqual(RichText.inlineMathBodies("$x$, $x^2 + y^2$ y $A^{-1}$ con $\\frac{1}{2}$ y \\(x + y\\)"),
                       ["x", "x^2 + y^2", "A^{-1}", "\\frac{1}{2}", "x + y"])
        XCTAssertEqual(RichText.inlineMathBodies("**Sea $A$** y *$84$*"), ["A", "84"])
    }

    /// The KaTeX page runs the same rule in JavaScript; both must find the
    /// same formulas.
    func testJavaScriptRuleMatchesSwift() throws {
        let context = try XCTUnwrap(JSContext())
        context.evaluateScript(RichText.inlineMathScript)
        XCTAssertNil(context.exception)
        let corpus = MarkdownMathFixture.document.components(separatedBy: "\n") + [
            "Set $HOME and then export $PATH.", "Cuesta entre $10-$25 al mes.", "$25 USD",
            "Escrito como \\(e^{i\\pi} + 1 = 0\\)", "Usa $\\frac{a}{b}$ y $x_1$", "US$10 y US$20$", "el factor $1/84$ y $10 - $25",
        ]
        let tokenize = try XCTUnwrap(context.objectForKeyedSubscript("toshTokenizeMath"))
        for line in corpus {
            let result = tokenize.call(withArguments: [line])
            let formulas = result?.objectForKeyedSubscript("formulas")?.toArray() as? [String]
            XCTAssertEqual(formulas, RichText.inlineMathBodies(line), line)
        }
    }
}

/// Loads the real page off screen and runs its measure. Off screen WebKit
/// skips animation frames, so the test calls the measure directly.
@MainActor
final class RichContentHeightTests: XCTestCase {
    private final class Box { var values: [CGFloat] = [] }

    private func makeCoordinator() -> (RichWebView.Coordinator, Box) {
        let box = Box()
        let coordinator = RichWebView.Coordinator(height: Binding(
            get: { box.values.last ?? 0 }, set: { box.values.append($0) }))
        return (coordinator, box)
    }

    /// The page and a copy of the bundled KaTeX and marked, in one folder the
    /// web process may read.
    private func pageURL(_ source: String, kind: RichContentKind) throws -> URL {
        let assets = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Assets/rich-content")
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-rich-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        for name in ["katex", "marked"] {
            try FileManager.default.copyItem(at: assets.appendingPathComponent(name),
                                             to: folder.appendingPathComponent(name))
        }
        let page = folder.appendingPathComponent("page.html")
        try RichWebView.html(source: source, kind: kind).write(to: page, atomically: true, encoding: .utf8)
        return page
    }

    private func load(_ source: String, kind: RichContentKind, size: NSSize) throws -> WKWebView {
        let view = WKWebView(frame: NSRect(origin: .zero, size: size))
        let page = try pageURL(source, kind: kind)
        view.loadFileURL(page, allowingReadAccessTo: page.deletingLastPathComponent())
        let deadline = Date().addingTimeInterval(20)
        while view.isLoading, Date() < deadline { RunLoop.main.run(until: Date().addingTimeInterval(0.05)) }
        XCTAssertFalse(view.isLoading)
        var rendered: Any?
        view.evaluateJavaScript("document.querySelectorAll('.katex').length") { value, _ in rendered = value ?? 0 }
        while rendered == nil, Date() < deadline { RunLoop.main.run(until: Date().addingTimeInterval(0.02)) }
        XCTAssertGreaterThan((rendered as? Int) ?? 0, 0, "KaTeX did not render")
        return view
    }

    private func measure(_ view: WKWebView, size: NSSize) throws -> CGFloat {
        view.setFrameSize(size)
        var result: Result<Any, Error>?
        view.callAsyncJavaScript("await document.fonts.ready; return toshMeasure();",
                                 arguments: [:], in: nil, in: .page) { result = $0 }
        let deadline = Date().addingTimeInterval(20)
        while result == nil, Date() < deadline { RunLoop.main.run(until: Date().addingTimeInterval(0.02)) }
        let value = try XCTUnwrap(try XCTUnwrap(result).get() as? NSNumber)
        return CGFloat(truncating: value)
    }

    func testReportedHeightIsTheContentAlone() {
        XCTAssertFalse(RichWebView.heightScript.contains("scrollHeight"))
        let (coordinator, box) = makeCoordinator()
        let send = { (height: Double) in
            coordinator.userContentController(WKUserContentController(),
                                              didReceive: FakeMessage(body: ["height": height, "baseline": 14]))
        }
        send(37)
        XCTAssertEqual(box.values, [37])
        send(37.2)
        XCTAssertEqual(box.values, [37], "sub-pixel repeats must not trigger a layout")
        send(21)
        XCTAssertEqual(box.values, [37, 21])
    }

    func testInlineHeightDoesNotFollowTheFrame() throws {
        let source = "El determinante es $84$ y la inversa usa $A^{-1}$ con $\\frac{1}{2}$."
        let view = try load(source, kind: .inlineMath, size: NSSize(width: 420, height: 1000))
        let fitted = try measure(view, size: NSSize(width: 420, height: 1000))
        XCTAssertGreaterThan(fitted, 10)
        XCTAssertLessThan(fitted, 60, "a 1000 pt frame must not leak into the measure")

        // Fit the frame to the report, as the chat does, and measure again.
        for _ in 0..<3 {
            XCTAssertEqual(try measure(view, size: NSSize(width: 420, height: fitted)), fitted)
        }
        var probe: Any?
        view.evaluateJavaScript("document.getElementById('toshBaseline').getBoundingClientRect().bottom") { v, _ in probe = v ?? 0 }
        while probe == nil { RunLoop.main.run(until: Date().addingTimeInterval(0.02)) }
        let baseline = CGFloat(truncating: (probe as? NSNumber) ?? 0)
        XCTAssertGreaterThan(baseline, 0)
        XCTAssertLessThan(baseline, fitted, "the first baseline sits inside the first line")

        let narrow = try measure(view, size: NSSize(width: 90, height: fitted))
        XCTAssertGreaterThan(narrow, fitted)
        XCTAssertEqual(try measure(view, size: NSSize(width: 420, height: narrow)), fitted,
                       "height must shrink back when the width grows")
    }

    func testBlockHeightIsStable() throws {
        let formula = String(MarkdownMathFixture.inverse.dropFirst().dropLast())
        let view = try load(formula, kind: .math, size: NSSize(width: 600, height: 900))
        let first = try measure(view, size: NSSize(width: 600, height: 900))
        XCTAssertLessThan(first, 300)
        for _ in 0..<3 {
            XCTAssertEqual(try measure(view, size: NSSize(width: 600, height: first)), first)
        }
        XCTAssertEqual(try measure(view, size: NSSize(width: 600, height: 40)), first)
    }
}

private final class FakeMessage: WKScriptMessage {
    private let value: Any
    init(body: Any) { value = body }
    override var body: Any { value }
}
