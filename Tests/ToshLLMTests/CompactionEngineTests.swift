// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// The chat's own summary against a running engine: a conversation that keeps growing past 70% of the context is
/// summarized with the model, the meter follows it, and every turn after it still gets an answer.
///
///     TOSH_COMPACT_E2E_PORT=8099 TOSH_COMPACT_E2E_CTX=8192 ./scripts/test.sh --filter CompactionEngineTests
final class CompactionEngineTests: XCTestCase {
    private static let words = ("the crew walked the north fence line and noted posts leaning after the storm while the "
        + "creek ran high along the lower pasture and the gate by the county road needed a new hinge before the next "
        + "inspection").split(separator: " ")

    private static func section(_ n: Int) -> String {
        let text = (0..<1100).map { String(words[($0 * 7 + n) % words.count]) }.joined(separator: " ")
        return "Section \(n) of the field report:\n\(text)\n\nSummarize this section in one sentence."
    }

    @MainActor
    func testALongConversationIsSummarizedAndGoesOn() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let port = environment["TOSH_COMPACT_E2E_PORT"].flatMap(Int.init) else {
            throw XCTSkip("Set TOSH_COMPACT_E2E_PORT to a running engine to run this test")
        }
        let context = environment["TOSH_COMPACT_E2E_CTX"].flatMap(Int.init) ?? 8192
        let defaults = UserDefaults.standard
        let saved = [SettingsKeys.ctx, SettingsKeys.chatAutoCompact, SettingsKeys.agentToolsEnabled,
                     SettingsKeys.memoryToolsEnabled].map { ($0, defaults.object(forKey: $0)) }
        defer { for (key, value) in saved { defaults.set(value, forKey: key) } }
        defaults.set(context, forKey: SettingsKeys.ctx)
        defaults.set(true, forKey: SettingsKeys.chatAutoCompact)
        defaults.set(false, forKey: SettingsKeys.agentToolsEnabled)
        defaults.set(false, forKey: SettingsKeys.memoryToolsEnabled)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tosh-compact-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatStore(storageDirectory: directory)
        var sampling = ChatSamplingSettings()
        sampling.reasoningEffort = "off"
        var rows: [[String: Any]] = []
        var summarizedAt: Int?
        for turn in 1...Int(environment["TOSH_COMPACT_E2E_TURNS"] ?? "12")! {
            store.send(text: Self.section(turn), port: port, temperature: 0, maxTokens: 256, system: "",
                       thinking: false, sampling: sampling)
            let deadline = Date().addingTimeInterval(600)
            while (store.agentFlowActive || store.compacting), Date() < deadline {
                try await Task.sleep(for: .milliseconds(200))
            }
            let conversation = try XCTUnwrap(store.current)
            let reading = try XCTUnwrap(store.contextReading)
            let answer = conversation.messages.last { $0.role == "assistant" }?.parts.body ?? ""
            let row: [String: Any] = ["turn": turn, "tokens": reading.tokens, "estimated": reading.estimated,
                                      "summarized": conversation.summarizedCount ?? 0,
                                      "summary_chars": conversation.summary?.count ?? 0,
                                      "error": store.lastError ?? "", "answer": String(answer.prefix(160))]
            rows.append(row)
            print("COMPACT", row)
            if summarizedAt == nil, (conversation.summarizedCount ?? 0) > 0 { summarizedAt = turn }
            XCTAssertNil(store.lastError, "turn \(turn): \(store.lastError ?? "")")
            XCTAssertFalse(answer.isEmpty, "turn \(turn) has no answer")
        }
        if let path = environment["TOSH_COMPACT_E2E_TRACE"],
           let data = try? JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys]) {
            try data.write(to: URL(fileURLWithPath: path))
        }
        let at = try XCTUnwrap(summarizedAt, "the conversation was never summarized")
        // the turns after the summary went on, and the meter fell back from where it had been
        XCTAssertLessThan(at, rows.count)
        XCTAssertLessThan(rows.last?["tokens"] as? Int ?? .max, context)
    }
}
