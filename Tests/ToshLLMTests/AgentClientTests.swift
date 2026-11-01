// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// The chat as a client of the engine's agent: what it sends, and how the agent's events show.
@MainActor
final class AgentClientTests: XCTestCase {
    private let integral = #"{"success": true, "operation": "integrate", "value": 6.49393940227, "interpreted_input": ["limits: [0, +oo)"], "result_kind": "approximate"}"#
    private let refused = #"{"success": false, "operation": "evaluate", "error": {"code": "transcription_mismatch", "message": "No."}}"#

    private func store() -> (ChatStore, UUID) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tosh-agent-client-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let store = ChatStore(storageDirectory: directory)
        store.newConversation()
        let id = store.currentID!
        let index = store.currentIndex!
        store.conversations[index].messages = [ChatMessage(role: "user", content: "Integrate it."),
                                               ChatMessage(role: "assistant", content: "")]
        return (store, id)
    }

    private func event(pass: Int, id: String, tool: String, state: String, result: String) -> [String: Any] {
        ["type": "tool_call", "pass": pass,
         "call": ["id": id, "tool": tool, "arguments": ["operation": "integrate"], "status": state == "completed" ? "ok" : "transcription_mismatch",
                  "state": state, "result": result]]
    }

    func testCallsOfOnePassShareARoundAheadOfTheAnswer() throws {
        let (store, id) = store()
        store.agentEvent(["type": "intent", "intent": "computational", "asked_model": false], conversation: id)
        store.agentEvent(event(pass: 1, id: "a", tool: "sympy_expression", state: "failed", result: refused), conversation: id)
        store.agentEvent(event(pass: 1, id: "b", tool: "scientific_compute", state: "completed", result: integral), conversation: id)
        store.agentEvent(event(pass: 2, id: "c", tool: "scientific_compute", state: "completed", result: integral), conversation: id)
        let messages = try XCTUnwrap(store.current?.messages)
        XCTAssertEqual(messages.map(\.role), ["user", "assistant", "tool", "tool", "assistant", "tool", "assistant"])
        XCTAssertEqual(messages[1].toolCalls?.map(\.serverID), ["a", "b"])
        XCTAssertEqual(messages[1].toolCalls?.map(\.state), [.failed, .completed])
        XCTAssertEqual(messages[2].toolCallID, "a")
        XCTAssertEqual(messages[4].toolCalls?.first?.arguments, #"{"operation":"integrate"}"#)
        XCTAssertTrue(messages.last?.content.isEmpty == true, "the answer stays last, still streaming")
        XCTAssertEqual(store.lastMathIntent, .computational)
        XCTAssertTrue(MathTranscriptionService.succeeded(try XCTUnwrap(messages[4].toolCalls?.first)))
        // the rows group as the chat's own rounds do
        let rows = TranscriptRow.rows(Array(messages.dropLast()))
        XCTAssertEqual(rows.count, 2, "the user message, then one block with both rounds")
    }

    func testAnExtensionAgentHandsBackOnlyWhenTheChatHasToolsOfItsOwn() throws {
        let defaults = UserDefaults.standard
        let keys = ["toolPermission.always.builtin.sympy_solve", SettingsKeys.mathToolsAllowed]
        let saved = keys.map { defaults.object(forKey: $0) }
        defer { for (key, value) in zip(keys, saved) { defaults.set(value, forKey: key) } }
        keys.forEach(defaults.removeObject(forKey:))

        let math = try BuiltinToolInfo(displayName: "Solve", name: "sympy_solve", writesData: false, definition: [:])
        let file = try BuiltinToolInfo(displayName: "Read", name: "read_file", writesData: false, definition: [:])
        let foreign = try BuiltinToolInfo(displayName: "Solve", name: "sympy_solve", definition: [:],
                                          mcpServerID: UUID(), remoteName: "solve")
        XCTAssertFalse(ChatStore.hasOwnTools([]))
        XCTAssertFalse(ChatStore.hasOwnTools([math]), "the agent covers the bundled math tools")
        XCTAssertTrue(ChatStore.hasOwnTools([math, file]))
        XCTAssertTrue(ChatStore.hasOwnTools([foreign]), "a user's MCP server is not the helper")

        XCTAssertTrue(ChatStore.agentDeclined(["version": 1, "outcome": "declined", "passes": 0]))
        XCTAssertFalse(ChatStore.agentDeclined(["version": 1, "outcome": "answered"]))
        XCTAssertFalse(ChatStore.agentDeclined(["version": 2, "outcome": "declined"]))
        XCTAssertFalse(ChatStore.agentDeclined(nil))
    }

    func testAnyOtherToolShowsItsOperationAndLaidOutJSON() {
        let call = ChatToolCall(name: "gis_query", arguments: #"{"operation":"count","target":"https:\/\/x\/MapServer"}"#,
                                result: #"{"success": true, "data": {"count": 7}}"#, state: .completed)
        let shown = ToolCallPresentation.make(call)
        XCTAssertEqual(shown.title, "gis_query (count)")
        XCTAssertEqual(shown.code?.contains("\"https://x/MapServer\""), true)
        XCTAssertEqual(shown.result?.components(separatedBy: "\n").count, 6)
        XCTAssertNil(ToolCallPresentation.indented("plain text"))
        XCTAssertNil(ToolCallPresentation.indented("[" + String(repeating: "1,", count: 9000) + "1]"))
    }

    func testACallReportedOnlyByItsStatusShowsAsItWent() throws {
        let (store, id) = store()
        for (callID, status) in [("a", "ok"), ("b", "not_found")] {
            store.agentEvent(["type": "tool_call", "pass": 1,
                              "call": ["id": callID, "tool": "gis_query", "status": status, "result": "{}"]], conversation: id)
        }
        let messages = try XCTUnwrap(store.current?.messages)
        XCTAssertEqual(messages[1].toolCalls?.map(\.state), [.completed, .failed])
    }

    /// The ten math tools run without asking by default, so math turns go to the engine's agent; nothing
    /// else changes, and the user can turn it off.
    func testOnlyTheBundledMathToolsAreAllowedByDefault() {
        let defaults = UserDefaults.standard
        let names = ["sympy_expression", "sympy_solve", "sympy_matrix", "sympy_verify", "scientific_compute",
                     "scientific_linalg", "scientific_optimize", "scientific_signal", "scientific_ode", "scientific_stats"]
        let others = ChatToolsService.builtInNames + ["run_javascript", "memory_archive", "github_search"]
        let keys = (names + others).map { "toolPermission.always.builtin.\($0)" } + [SettingsKeys.mathToolsAllowed]
        let saved = keys.map { defaults.object(forKey: $0) }
        defer { for (key, value) in zip(keys, saved) { defaults.set(value, forKey: key) } }
        keys.forEach(defaults.removeObject(forKey:))

        XCTAssertTrue(names.allSatisfy { ChatToolsService.isAlwaysAllowed($0) })
        XCTAssertFalse(others.contains { ChatToolsService.isAlwaysAllowed($0) })
        XCTAssertFalse(ChatToolsService.isAlwaysAllowed("sympy_expression", bundled: false), "a user's MCP server is not the helper")
        XCTAssertTrue(SettingsKeys.resettableOptionKeys.contains(SettingsKeys.mathToolsAllowed))

        defaults.set(false, forKey: SettingsKeys.mathToolsAllowed)
        XCTAssertFalse(names.contains { ChatToolsService.isAlwaysAllowed($0) })
        ChatToolsService.allowAlways("sympy_solve")
        XCTAssertTrue(ChatToolsService.isAlwaysAllowed("sympy_solve"), "an explicit choice still wins")
        ChatToolsService.revokeAllPermissions()
        XCTAssertFalse(ChatToolsService.isAlwaysAllowed("sympy_solve"))
    }

    func testTheAgentGetsWhatWasSaidButNoToolTraffic() {
        var round = ChatMessage(role: "assistant", content: "")
        round.toolCalls = [ChatToolCall(serverID: "a", name: "sympy_expression", arguments: "{}", result: integral, state: .completed)]
        let messages = [ChatMessage(role: "user", content: "First."), round,
                        ChatMessage(role: "tool", content: integral, toolCallID: "a"),
                        ChatMessage(role: "assistant", content: "It is 6.49393940227."),
                        ChatMessage(role: "user", content: "Second.")]
        let history = ChatStore.agentHistory(system: "Be brief.", summary: nil, messages: messages, from: 0)
        XCTAssertEqual(history.map { $0["role"] as? String }, ["system", "user", "assistant", "user"])
        XCTAssertEqual(history[2]["content"] as? String, "It is 6.49393940227.")
        XCTAssertFalse(history.contains { $0["tool_calls"] != nil })
    }

    func testOnlyTheLastRequestTakesItsImagesToTheAgent() {
        var earlier = ChatMessage(role: "user", content: "What is this?")
        earlier.imageURIs = ["data:image/png;base64,AAAA"]
        var last = ChatMessage(role: "user", content: "")
        last.imageURIs = ["data:image/jpeg;base64,BBBB"]
        let messages = [earlier, ChatMessage(role: "assistant", content: "A cat."), last]
        let history = ChatStore.agentHistory(system: "", summary: nil, messages: messages, from: 0)
        XCTAssertEqual(history[0]["content"] as? String, "What is this?")
        let parts = history[2]["content"] as? [[String: Any]]
        XCTAssertEqual(parts?.count, 1)
        XCTAssertEqual((parts?.first?["image_url"] as? [String: Any])?["url"] as? String, "data:image/jpeg;base64,BBBB")
        let blind = ChatStore.agentHistory(system: "", summary: nil, messages: messages, from: 0,
                                           modalities: ModelModalities(vision: false, audio: false, video: false))
        XCTAssertEqual(blind[2]["content"] as? String, "")
    }
}
