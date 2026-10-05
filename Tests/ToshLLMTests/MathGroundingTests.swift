// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class MathGroundingTests: XCTestCase {
    private let request = #"""
        Calcula exactamente
        \[
        I=\int_{0}^{\infty}\frac{x^3}{e^x-1}\,dx
        \]
        Después:
        1. expresa el resultado en forma exacta;
        2. dame su valor decimal con 10 cifras decimales;
        3. verifica el resultado mediante integración numérica independiente;
        4. indica el error absoluto entre el valor exacto evaluado numéricamente y la integración numérica.
        """#
    private let integral = #"{"success": true, "operation": "integrate", "value": 6.49393940227, "error_estimate": 9.32511651849e-11, "method": "adaptive quadrature (QUADPACK)", "tolerance": 1e-10, "evaluations": 255, "warnings": [], "interpreted_input": ["expression: x**3/(exp(x) - 1)", "variable: x", "limits: [0, +oo)"], "result_kind": "approximate"}"#
    private let factored = #"{"success": true, "operation": "factor", "exact": "(x - 3)*(x - 2)", "warnings": [], "interpreted_input": ["expression: x**2 - 5*x + 6"], "result_kind": "exact"}"#
    private let memory = #"{"success": false, "operation": "evaluate", "error": {"code": "needs_review", "message": "Not computed."}, "interpreted_input": ["expression: pi**4/15"], "reasons": ["'expression' is `pi**4/15`; the request does not spell that formula out"]}"#
    private let open = #"{"success": false, "operation": "integrate", "exact": null, "unevaluated": "Integral(x**3/(exp(x) - 1), (x, 0, oo))", "timed_out": false, "error": {"code": "no_closed_form", "message": "SymPy found no closed form for this integral."}, "warnings": [], "interpreted_input": ["expression: x**3/(exp(x) - 1)", "limits: [0, +oo)"]}"#

    private func round(_ id: String, _ name: String, _ result: String) -> [ChatMessage] {
        let state: ChatToolCallState = MathTranscriptionService.reply(result)?["success"] as? Bool == true ? .completed : .failed
        return [ChatMessage(role: "assistant", content: "", toolCalls: [
                    ChatToolCall(serverID: id, name: name, arguments: "{}", result: result, state: state)]),
                ChatMessage(role: "tool", content: result, toolCallID: id)]
    }

    private func turn(_ rounds: [ChatMessage]...) -> [ChatMessage] {
        [ChatMessage(role: "user", content: request)] + rounds.flatMap { $0 }
    }

    private func step(_ messages: [ChatMessage], _ answer: String, closing: String? = nil,
                      finalizing: Bool = false, regrounded: Bool = false) -> MathFinalStep {
        MathGrounding.step(messages: messages, answer: answer, closing: closing,
                           finalizing: finalizing, regrounded: regrounded)
    }

    func testAValidResultSurvivesALaterRefusal() throws {
        let messages = turn(round("c1", "scientific_compute", integral), round("c2", "sympy_expression", open),
                            round("c3", "sympy_expression", memory), round("c4", "sympy_expression", memory))
        guard case .finalize(let note) = MathGrounding.stopStep(messages: messages) else {
            return XCTFail("the turn should end on its validated result")
        }
        XCTAssertTrue(note.contains("value: 6.49393940227") && note.contains("error_estimate: 9.32511651849e-11"), note)
        XCTAssertTrue(note.contains("limits: [0, +oo)") && !note.contains("pi**4"), note)
        XCTAssertEqual(step(messages, "", closing: MathTranscriptionService.unresolvedMessage()), .finalize(note))
        // the last round can only state it; with nothing usable Tosh states it itself
        guard case .replace(let safe) = step(messages, " ", finalizing: true) else { return XCTFail() }
        XCTAssertTrue(safe.contains("6.49393940227") && safe.contains("9.32511651849e-11"), safe)
        XCTAssertTrue(safe.contains("Integral(x**3/(exp(x) - 1), (x, 0, oo))") && !safe.contains("pi"), safe)
        XCTAssertEqual(step(messages, "La forma exacta es $\\pi^4/15$.", finalizing: true), .replace(safe))
        XCTAssertEqual(step(messages, "La integral numérica da 6.49393940227 (error estimado 9.3e-11); la forma exacta no se pudo validar.",
                            finalizing: true), .keep)
    }

    func testNoValidResultKeepsTheFixedMessage() {
        let messages = turn(round("c1", "sympy_expression", memory), round("c2", "sympy_expression", memory))
        XCTAssertEqual(MathGrounding.stopStep(messages: messages), .keep)
        XCTAssertEqual(step(messages, "", closing: MathTranscriptionService.unresolvedMessage()), .keep)
        XCTAssertEqual(MathGrounding.safeAnswer(results: [], calls: []), MathTranscriptionService.unresolvedMessage())
    }

    func testTwoValidResultsAreBothKept() throws {
        let messages = turn(round("c1", "scientific_compute", integral), round("c2", "sympy_expression", factored),
                            round("c3", "sympy_expression", memory))
        guard case .finalize(let note) = MathGrounding.stopStep(messages: messages) else { return XCTFail() }
        XCTAssertTrue(note.contains("6.49393940227") && note.contains("(x - 3)*(x - 2)"), note)
        let results = MathLedger.results(messages, from: MathLedger.turnStart(messages))
        XCTAssertEqual(results.map(\.operation), ["integrate", "factor"])
        let safe = MathGrounding.safeAnswer(results: results, calls: MathLedger.calls(messages))
        XCTAssertTrue(safe.contains("6.49393940227") && safe.contains("(x - 3)*(x - 2)"), safe)
        XCTAssertFalse(safe.contains("numerical approximations"), "one of them is exact")
        let again = turn(round("c1", "scientific_compute", integral), round("c2", "scientific_compute", integral))
        let once = MathGrounding.safeAnswer(results: MathLedger.results(again), calls: MathLedger.calls(again))
        XCTAssertEqual(once.components(separatedBy: "6.49393940227").count, 2, once)
        XCTAssertTrue(once.contains("No exact result could be validated"), once)
    }

    func testANewNumberInTheAnswerIsSentBack() throws {
        let messages = turn(round("c1", "scientific_compute", integral))
        let answer = "La integral vale 6.49393940227 y el error absoluto es $6\\times 10^{-12}$."
        guard case .again(let note) = step(messages, answer) else { return XCTFail() }
        XCTAssertTrue(note.contains("6\\times 10^{-12}") && note.contains("value: 6.49393940227"), note)
        guard case .replace(let safe) = step(messages, answer, regrounded: true) else { return XCTFail() }
        XCTAssertFalse(safe.contains("10^{-12}"), safe)
    }

    func testAnExactFormFromMemoryIsSentBack() {
        let messages = turn(round("c1", "scientific_compute", integral))
        let missing = MathGrounding.ungrounded("El resultado exacto es $\\frac{\\pi^4}{15} \\approx 6.4939$, porque es $6\\zeta(4)$.",
                                               sources: [request], results: MathLedger.results(messages))
        XCTAssertEqual(Set(missing), ["15", "6", "π", "ζ"])
        guard case .again = step(messages, "El resultado exacto es $\\pi^4/15$.") else { return XCTFail() }
    }

    func testADerivedValueNeedsAToolFirst() {
        let ask = "Find the determinant of [[1, 2], [3, 4]] and multiply it by 7."
        let determinant = #"{"success": true, "operation": "determinant", "exact": "-2", "interpreted_input": ["matrix: 2 x 2 [[1, 2], [3, 4]]"], "result_kind": "exact"}"#
        let times = #"{"success": true, "operation": "evaluate", "value": -14.0, "interpreted_input": ["expression: -2*7"], "result_kind": "approximate"}"#
        let answer = "The determinant is -2, and multiplied by 7 it is -14."
        var messages = [ChatMessage(role: "user", content: ask)] + round("c1", "sympy_matrix", determinant)
        guard case .again(let note) = step(messages, answer) else { return XCTFail() }
        XCTAssertTrue(note.hasPrefix("Your answer states values that are not in the request or in a validated tool result: 14."), note)
        messages += round("c2", "scientific_compute", times)
        XCTAssertEqual(step(messages, answer), .keep)
    }

    func testExplanationAroundValidatedResultsStays() {
        let messages = turn(round("c1", "sympy_expression", open), round("c2", "scientific_compute", integral))
        let answer = """
            ### 1. Forma exacta
            Las herramientas no obtuvieron una forma cerrada para \\(\\int_{0}^{\\infty} \\frac{x^3}{e^x-1}\\,dx\\).

            ### 2. Valor decimal con 10 cifras decimales
            **6,4939394023** según la integración numérica.

            ### 3. Verificación
            La cuadratura adaptativa da \\(I \\approx 6.49393940227\\) con un error estimado de \\(9.3\\,\\times\\,10^{-11}\\), del orden de 10⁻¹⁰.

            4. El error absoluto no se puede dar sin la forma exacta.
            """
        XCTAssertEqual(MathGrounding.ungrounded(answer, sources: [request], results: MathLedger.results(messages)), [])
        XCTAssertEqual(step(messages, answer), .keep)
    }

    func testOtherToolsAndPlainTurnsAreLeftAlone() {
        let file = [ChatMessage(role: "user", content: "Lee notas.txt")]
            + round("c1", "read_file", "x = 42")
        XCTAssertEqual(step(file, "El archivo dice 42, que es 6 por 7."), .keep)
        XCTAssertEqual(step([ChatMessage(role: "user", content: "¿Cuánto es 6 por 7?")], "42"), .keep)
        // an earlier turn with math does not bind a later one without it
        let later = turn(round("c1", "scientific_compute", integral)) + [ChatMessage(role: "assistant", content: "6.49"),
                                                                       ChatMessage(role: "user", content: "Gracias")]
        XCTAssertEqual(step(later, "De nada, son 3 pasos."), .keep)
    }

    func testNumbersAreReadInTheirUsualNotations() {
        let read = MathGrounding.numbers(in: "9.3\\,\\times\\,10^{-11}, 6×10⁻¹², 1e-10, 6,49 y x^3")
        XCTAssertEqual(read.map(\.text), ["9.3\\,\\times\\,10^{-11}", "6×10^-12", "1e-10", "6,49", "3"].map {
            $0.replacingOccurrences(of: "\\,", with: " ")
        })
        XCTAssertEqual(read[0].value, 9.3e-11, accuracy: 1e-24)
        XCTAssertEqual(read[1].value, 6e-12, accuracy: 1e-24)
        let values = [6.49393940227]
        XCTAssertTrue(MathGrounding.grounded(MathGrounding.numbers(in: "6.4939394023")[0], by: values))
        XCTAssertTrue(MathGrounding.grounded(MathGrounding.numbers(in: "6.4939")[0], by: values))
        XCTAssertFalse(MathGrounding.grounded(MathGrounding.numbers(in: "6.4938")[0], by: values))
        XCTAssertFalse(MathGrounding.grounded(MathGrounding.numbers(in: "6.4939394024")[0], by: values))
    }
}
