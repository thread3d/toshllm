// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class SymPyToolsTests: XCTestCase {
    private func makeSettings() -> ServerSettings {
        ServerSettings(serverBinary: "/usr/bin/true", modelPath: "/tmp/m.gguf", port: 8080,
                       ngl: 99, ncmoe: 0, ctx: 8192, threads: 6, flashAttn: "auto",
                       noMmap: true, jinja: true,
                       vramReserveMB: 1024, gpuIndex: -1, extraArgs: "",
                       cacheTypeK: "f16", cacheTypeV: "f16", mlock: false)
    }

    private func makeRuntime() throws -> URL {
        let resources = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-sympy-test-\(UUID().uuidString)")
        let bin = resources.appendingPathComponent("tosh-sympy/python/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.copyItem(atPath: "/usr/bin/true",
                                         toPath: bin.appendingPathComponent("python3").path)
        addTeardownBlock { try? FileManager.default.removeItem(at: resources) }
        return resources
    }

    func testOffByDefault() {
        XCTAssertFalse(makeSettings().sympyEnabled)
        XCTAssertFalse(makeSettings().arguments.contains("--mcp-servers-json"))
        XCTAssertTrue(SettingsKeys.resettableOptionKeys.contains(SettingsKeys.sympyEnabled))
    }

    func testDisabledAddsNothingEvenWithTheRuntimePresent() throws {
        XCTAssertEqual(SymPyToolsService.serverArguments(enabled: false, resources: try makeRuntime()), [])
    }

    func testEnabledWithoutTheRuntimeAddsNothing() {
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        XCTAssertEqual(SymPyToolsService.serverArguments(enabled: true, resources: empty), [])
        XCTAssertEqual(SymPyToolsService.serverArguments(enabled: true, resources: nil), [])
    }

    func testEnabledStartsTheHelperAsAnMCPServer() throws {
        let resources = try makeRuntime()
        let arguments = SymPyToolsService.serverArguments(enabled: true, resources: resources)
        XCTAssertEqual(arguments.first, "--mcp-servers-json")
        let config = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(try XCTUnwrap(arguments.dropFirst().first).utf8)) as? [String: Any])
        let server = try XCTUnwrap((config["mcpServers"] as? [String: Any])?["sympy"] as? [String: Any])
        let runtime = resources.appendingPathComponent("tosh-sympy").path
        XCTAssertEqual(server["command"] as? String, runtime + "/python/bin/python3")
        XCTAssertEqual(server["args"] as? [String], ["-I", "-B", runtime + "/tosh_sympy/server.py"])
    }

    func testScientificServerRunsBesideSymPy() throws {
        let resources = try makeRuntime()
        XCTAssertFalse(makeSettings().scientificEnabled)
        XCTAssertTrue(SettingsKeys.resettableOptionKeys.contains(SettingsKeys.scientificEnabled))
        func servers(_ sympy: Bool, _ scientific: Bool) throws -> [String: Any] {
            let arguments = SymPyToolsService.serverArguments(enabled: sympy, scientific: scientific,
                                                              resources: resources)
            guard let json = arguments.dropFirst().first else { return [:] }
            let config = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
            return try XCTUnwrap(config["mcpServers"] as? [String: Any])
        }
        XCTAssertEqual(try servers(false, false).count, 0)
        XCTAssertEqual(Set(try servers(true, false).keys), ["sympy", "agent"])
        XCTAssertEqual(Set(try servers(false, true).keys), ["scientific", "agent"])
        let both = try servers(true, true)
        XCTAssertEqual(Set(both.keys), ["sympy", "scientific", "agent"])
        let scientific = try XCTUnwrap(both["scientific"] as? [String: Any])
        XCTAssertEqual((scientific["args"] as? [String])?.last, "scientific")
        XCTAssertEqual((both["sympy"] as? [String: Any])?["command"] as? String, scientific["command"] as? String)
    }

    /// The chat asks for the agent by header, so it always runs with the math tools; only the API
    /// toggle lets clients that do not ask get it too.
    func testAgentRunsWithTheMathToolsAndAnswersUnaskedClientsOnlyWhenSwitchedOn() throws {
        let resources = try makeRuntime()
        let explicit = SymPyToolsService.serverArguments(enabled: true, resources: resources)
        XCTAssertEqual(Array(explicit.suffix(3)), ["--mcp-agent", "agent", "--mcp-agent-explicit"])
        let open = SymPyToolsService.serverArguments(enabled: true, agent: true, resources: resources)
        XCTAssertEqual(Array(open.suffix(2)), ["--mcp-agent", "agent"])
        XCTAssertEqual(SymPyToolsService.serverArguments(enabled: false, agent: true, resources: resources), [])
    }

    func testRoutingRuleIsSentOnlyWithBothToolSets() {
        let rule = ScientificToolsService.routingRule
        XCTAssertEqual(ScientificToolsService.system("", sympy: true, scientific: true), rule)
        XCTAssertEqual(ScientificToolsService.system("Be brief.", sympy: true, scientific: true),
                       "Be brief.\n\n" + rule)
        XCTAssertEqual(ScientificToolsService.system("Be brief.", sympy: true, scientific: false), "Be brief.")
        XCTAssertEqual(ScientificToolsService.system("", sympy: false, scientific: true), "")
        XCTAssertEqual(ScientificToolsService.system("", sympy: false, scientific: false), "")
    }

    func testScientificCallIsPresentedAsMath() {
        XCTAssertTrue(ScientificToolsService.isTool("scientific_linalg"))
        XCTAssertFalse(ScientificToolsService.isTool("sympy_matrix"))
        let call = ChatToolCall(
            name: "scientific_compute",
            arguments: #"{"operation":"integrate","expression":"sin(x**2)","lower":0,"upper":10}"#,
            result: #"{"success": true, "operation": "integrate", "value": 0.58367089993, "error_estimate": 2.7e-11, "method": "adaptive quadrature (QUADPACK)", "warnings": []}"#)
        let presentation = ToolCallPresentation.make(call)
        XCTAssertEqual(presentation.kind, .math)
        XCTAssertEqual(presentation.title, "Integrate")
        XCTAssertEqual(presentation.code, "sin(x**2)")
        XCTAssertEqual(presentation.result?.components(separatedBy: "\n").first, "value: 0.58367089993")
        XCTAssertEqual(
            ScientificToolsService.readable(#"{"success":false,"operation":"solve","error":{"code":"singular_matrix","message":"singular"}}"#),
            "singular")
        XCTAssertEqual(ScientificToolsService.input(["values": [1, 2, 3]]), "values: 3 values")
    }

    func testMathCallsCarryTheUsersOwnWords() throws {
        let messages = [
            ChatMessage(role: "user", content: "Integrate x^2 from 0 to 3."),
            ChatMessage(role: "assistant", content: ""),
            ChatMessage(role: "tool", content: #"{"success": true, "exact": "9"}"#),
            ChatMessage(role: "user", content: "Now from 0 to infinity of exp(-x)."),
        ]
        let source = try XCTUnwrap(MathTranscriptionService.source(messages: messages))
        XCTAssertEqual(source["request"] as? String, "Now from 0 to infinity of exp(-x).")
        XCTAssertEqual(source["context"] as? String, "Integrate x^2 from 0 to 3.\n" + #"{"success": true, "exact": "9"}"#)
        XCTAssertNil(MathTranscriptionService.source(messages: [ChatMessage(role: "assistant", content: "hi")]))
        XCTAssertTrue(MathTranscriptionService.isMathTool("scientific_ode"))
        XCTAssertFalse(MathTranscriptionService.isMathTool("read_file"))
        let refused = #"{"success": false, "operation": "limit", "error": {"code": "transcription_mismatch", "message": "Not computed."}, "interpreted_input": ["expression: -x + sin(x)", "point: 0"]}"#
        XCTAssertEqual(SymPyToolsService.readable(refused), "Not computed.\n▸ expression: -x + sin(x)\n▸ point: 0")
        XCTAssertEqual(MathTranscriptionService.errorCode(try XCTUnwrap(MathTranscriptionService.reply("error: " + refused))),
                       "transcription_mismatch")
        let computed = #"{"success": true, "value": 1.5, "interpreted_input": ["limits: [0, 3]"], "result_kind": "approximate", "warnings": []}"#
        XCTAssertEqual(ScientificToolsService.readable(computed), "value: 1.5\n▸ limits: [0, 3]")
    }

    private let refusedBound = #"{"success": false, "operation": "integrate", "error": {"code": "transcription_mismatch", "message": "Not computed: 'upper' is 100."}, "interpreted_input": ["limits: [0, 100]"]}"#
    private let openIntegral = #"{"success": false, "operation": "integrate", "exact": null, "error": {"code": "no_closed_form", "message": "SymPy found no closed form for this integral."}}"#
    private let integral = #"{"success": true, "operation": "integrate", "value": 6.49393940227}"#
    private let memory = "El resultado exacto es π^4/15."

    private func call(_ id: String, _ name: String = "scientific_compute", result: String? = nil,
                      state: ChatToolCallState = .completed) -> ChatToolCall {
        ChatToolCall(serverID: id, name: name, arguments: "{}", result: result, state: state)
    }

    func testRefusedAndFailedCallsAreNotContext() throws {
        let messages = [
            ChatMessage(role: "user", content: "Integrate x^3/(e^x - 1) from 0 to infinity."),
            ChatMessage(role: "assistant", content: "", toolCalls: [
                call("c1", result: refusedBound, state: .failed),
                call("c2", "sympy_expression", result: openIntegral),
                call("c3", "read_file", result: "no such file", state: .failed),
                call("c4", "read_file", result: "x = 4"),
            ]),
            ChatMessage(role: "tool", content: refusedBound, toolCallID: "c1"),
            ChatMessage(role: "tool", content: openIntegral, toolCallID: "c2"),
            ChatMessage(role: "tool", content: "no such file", toolCallID: "c3"),
            ChatMessage(role: "tool", content: "x = 4", toolCallID: "c4"),
            ChatMessage(role: "assistant", content: "", toolCalls: [call("c5", result: integral)]),
            ChatMessage(role: "tool", content: integral, toolCallID: "c5"),
        ]
        let source = try XCTUnwrap(MathTranscriptionService.source(messages: messages))
        XCTAssertEqual(source["context"] as? String, "x = 4\n" + integral)
    }

    func testTextBeforeACallIsNeverTheAnswer() {
        let round = ChatMessage.toolRound(reasoning: "", visible: "  " + memory + "\n")
        XCTAssertEqual(round.content, "")
        XCTAssertEqual(round.interim, memory)
        XCTAssertEqual(ChatMessage.toolRound(reasoning: "plan", visible: memory).content, "<think>plan</think>")
        XCTAssertNil(ChatMessage.toolRound(reasoning: "plan", visible: " ").interim)
    }

    func testHeldTextIsDroppedWhenTheMathCallDoesNotSucceed() {
        let user = ChatMessage(role: "user", content: "Integrate x^3/(e^x - 1) from 0 to infinity.")
        for (result, state) in [(refusedBound, ChatToolCallState.failed), (openIntegral, .completed), ("denied", .denied)] {
            var message = ChatMessage(role: "assistant", content: "", toolCalls: [call("c1", state: .running)], interim: memory)
            XCTAssertNil(message.settledInterim, "shown while the call runs")
            message.toolCalls?[0].state = state
            message.toolCalls?[0].result = result
            message.settleInterim()
            XCTAssertNil(message.interim, result)
            XCTAssertTrue(message.parts.body.isEmpty)
            let history = ChatStore.requestHistory(system: "", summary: nil, messages: [user, message], from: 0)
            XCTAssertEqual(history.last?["content"] as? String, "", "sent back to the model")
        }
    }

    func testHeldTextOfASuccessfulRoundIsReasoningNotTheAnswer() {
        var message = ChatMessage(role: "assistant", content: "",
                                  toolCalls: [call("c1", result: integral), call("c2", "read_file", result: "x = 4")],
                                  interim: memory)
        message.settleInterim()
        XCTAssertEqual(message.settledInterim, memory)
        XCTAssertTrue(message.parts.body.isEmpty)
        let history = ChatStore.requestHistory(system: "", summary: nil, messages: [message], from: 0)
        XCTAssertEqual(history.last?["content"] as? String, memory)
        // without a math call the text shows at once, so a permission prompt keeps its explanation
        let other = ChatMessage(role: "assistant", content: "", toolCalls: [call("c3", "write_file", state: .awaitingPermission)],
                                interim: "Guardo la nota.")
        XCTAssertEqual(other.settledInterim, "Guardo la nota.")
    }

    func testConsecutiveToolRoundsAreOneRow() {
        let messages = [
            ChatMessage(role: "user", content: "q"),
            ChatMessage(role: "assistant", content: "", toolCalls: [call("c1", result: refusedBound, state: .failed)]),
            ChatMessage(role: "tool", content: refusedBound, toolCallID: "c1"),
            ChatMessage(role: "assistant", content: "", toolCalls: [call("c2", result: integral)]),
            ChatMessage(role: "tool", content: integral, toolCallID: "c2"),
            ChatMessage(role: "assistant", content: "answer"),
            ChatMessage(role: "tool", content: "stray"),
            ChatMessage(role: "user", content: "next"),
        ]
        let shape = TranscriptRow.rows(messages).map { row -> String in
            switch row {
            case .message(let message): message.role
            case .tools(let rounds): "tools \(rounds.count)"
            }
        }
        XCTAssertEqual(shape, ["user", "tools 4", "assistant", "tool", "user"])
    }

    private let refused = #"{"success": false, "operation": "limit", "error": {"code": "transcription_mismatch", "message": "Not computed."}}"#
    private let computed = #"{"success": true, "operation": "limit", "exact": "-1/6", "warnings": []}"#

    func testRefusedMathCallThenCorrectedCallFreesTheTurn() {
        var guardState = MathTurnGuard()
        guardState.record(tool: "sympy_expression", result: refused)
        XCTAssertEqual(guardState.next, .mathOnly)
        XCTAssertNil(guardState.closing(toolNames: ["sympy_expression"], arguments: ["{}"]))
        guardState.record(tool: "sympy_expression", result: computed)
        XCTAssertEqual(guardState.next, .free)
        XCTAssertNil(guardState.closing(toolNames: [], arguments: []))
    }

    func testTwoRefusalsEndTheTurn() {
        var guardState = MathTurnGuard()
        guardState.record(tool: "scientific_compute", result: refused)
        guardState.record(tool: "scientific_compute", result: refused)
        XCTAssertEqual(guardState.next, .stop)
    }

    func testAnswerWithoutToolAfterRefusalIsReplaced() {
        var guardState = MathTurnGuard()
        guardState.record(tool: "sympy_solve", result: refused)
        XCTAssertEqual(guardState.closing(toolNames: [], arguments: []), MathTranscriptionService.unresolvedMessage())
        XCTAssertEqual(guardState.closing(toolNames: ["read_file"], arguments: ["{}"]), MathTranscriptionService.unresolvedMessage())
    }

    func testClarificationAfterRefusalIsAllowed() {
        var guardState = MathTurnGuard()
        let review = #"{"success": false, "operation": "solve", "error": {"code": "needs_review", "message": "Not computed."}}"#
        guardState.record(tool: "sympy_solve", result: review)
        let text = guardState.closing(toolNames: [MathTranscriptionService.clarifyToolName], arguments: [#"{"missing": "data"}"#])
        XCTAssertEqual(text, MathTranscriptionService.unresolvedMessage(missing: "data"))
        XCTAssertNotEqual(text, MathTranscriptionService.unresolvedMessage())
    }

    func testSuccessfulMathAndOtherToolsLeaveTheTurnAlone() {
        var guardState = MathTurnGuard()
        guardState.record(tool: "sympy_expression", result: computed)
        XCTAssertEqual(guardState.next, .free)
        XCTAssertNil(guardState.closing(toolNames: [], arguments: []))
        // a non-math tool that fails, even with the same code, changes nothing
        guardState.record(tool: "read_file", result: refused)
        XCTAssertEqual(guardState.next, .free)
        // an ordinary failure of a math tool is not a refused transcription
        guardState.record(tool: "sympy_expression", result: #"{"success": false, "error": {"code": "timeout", "message": "x"}}"#)
        XCTAssertEqual(guardState.next, .free)
    }

    func testClarifyToolOffersOnlyFixedReasons() throws {
        let function = try XCTUnwrap(MathTranscriptionService.clarifyTool["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, MathTranscriptionService.clarifyToolName)
        let parameters = try XCTUnwrap(function["parameters"] as? [String: Any])
        let missing = try XCTUnwrap((parameters["properties"] as? [String: Any])?["missing"] as? [String: Any])
        XCTAssertEqual(missing["type"] as? String, "string")
        XCTAssertNotNil(missing["enum"] as? [String])
    }

    func testToolNames() {
        XCTAssertTrue(SymPyToolsService.isTool("sympy_expression"))
        XCTAssertTrue(SymPyToolsService.isTool("sympy_verify"))
        XCTAssertFalse(SymPyToolsService.isTool("read_file"))
        XCTAssertFalse(SymPyToolsService.isTool("sympy"))
    }

    func testCallIsPresentedAsMath() {
        let call = ChatToolCall(
            name: "sympy_expression",
            arguments: #"{"operation":"laplace_transform","expression":"exp(-a*t)"}"#,
            result: #"{"success": true, "operation": "laplace_transform", "exact": "1/(a + s)", "latex": "x", "warnings": []}"#)
        let presentation = ToolCallPresentation.make(call)
        XCTAssertEqual(presentation.kind, .math)
        XCTAssertEqual(presentation.title, "Laplace transform")
        XCTAssertEqual(presentation.code, "exp(-a*t)")
        XCTAssertEqual(presentation.result, "1/(a + s)")
    }

    func testReadableResults() {
        XCTAssertEqual(
            SymPyToolsService.readable(#"{"success":true,"exact":"sqrt(pi)","numeric":"1.77","warnings":["w"]}"#),
            "sqrt(pi)\n≈ 1.77\n⚠︎ w")
        XCTAssertEqual(
            SymPyToolsService.readable(#"{"success":true,"equivalent":false,"difference":"1","warnings":[]}"#),
            "equivalent: no\ndifference: 1")
        XCTAssertEqual(
            SymPyToolsService.readable(#"{"success":false,"operation":"factor","error":{"code":"timeout","message":"stopped"}}"#),
            "stopped")
        XCTAssertEqual(SymPyToolsService.readable("not json"), "not json")
        XCTAssertEqual(
            SymPyToolsService.readable(#"{"success":true,"exact":null,"unevaluated":"Integral(x**x, (x, 0, 1))","numeric":"0.78","method":"numerical_integration","warnings":[]}"#),
            "Integral(x**x, (x, 0, 1))\n≈ 0.78")
        XCTAssertEqual(
            SymPyToolsService.readable(#"{"success":false,"timed_out":true,"exact":null,"unevaluated":"Integral(f(x), x)","error":{"code":"timeout","message":"No closed form."},"warnings":[]}"#),
            "No closed form.\nIntegral(f(x), x)")
        XCTAssertEqual(SymPyToolsService.input(["equations": ["x = 1", "y = 2"]]), "x = 1\ny = 2")
        XCTAssertEqual(SymPyToolsService.input(["matrix": [["1", 2], [3, "4"]]]), "1  2\n3  4")
    }
}
