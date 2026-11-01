// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// The context meter shows the server's count while it is current and an estimate, said to be one, when it is not;
/// summaries are decided on the larger of the two; an answer cut by a full context is told apart from one cut at its
/// limit; and what an agent marked to keep of an answer survives the summary.
final class ContextBudgetTests: XCTestCase {
    private func conversation(_ messages: [ChatMessage], used: Int? = nil, at count: Int? = nil) -> Conversation {
        var c = Conversation(title: "t", messages: messages)
        c.contextUsed = used
        c.contextUsedAt = count ?? (used == nil ? nil : messages.count)
        return c
    }

    private func turns(_ n: Int, size: Int) -> [ChatMessage] {
        (0..<n).flatMap { i in
            [ChatMessage(role: "user", content: "Question \(i) " + String(repeating: "q", count: size)),
             ChatMessage(role: "assistant", content: "Answer \(i) " + String(repeating: "a", count: size))]
        }
    }

    func testTheMeterShowsTenthsUnderTenThousandTokens() {
        XCTAssertEqual(ChatStore.kiloTokens(0), "0.0k")
        XCTAssertEqual(ChatStore.kiloTokens(840), "0.8k")
        XCTAssertEqual(ChatStore.kiloTokens(9_950), "9.9k")
        XCTAssertEqual(ChatStore.kiloTokens(12_345), "12k")
    }

    func testTheServersCountIsShownWhileNoMessageCameAfterIt() {
        let messages = turns(2, size: 400)
        let exact = ChatStore.contextReading(conversation(messages, used: 900), toolTokens: 0)
        XCTAssertEqual(exact.tokens, 900)
        XCTAssertFalse(exact.estimated)
        // a turn the server did not count, such as an agent's answer by rules, leaves the count behind
        let stale = ChatStore.contextReading(conversation(messages + turns(1, size: 400), used: 900, at: 4),
                                             toolTokens: 0)
        XCTAssertTrue(stale.estimated)
        XCTAssertEqual(stale.tokens, ChatStore.contextEstimate(conversation(messages + turns(1, size: 400))))
    }

    func testANoughtFromTheServerNeverHidesAFullConversation() {
        // an agent turn answered by rules reported 0 tokens for a conversation of some 30 thousand
        let c = conversation(turns(30, size: 2000), used: 0)
        XCTAssertGreaterThan(ChatStore.contextForDecisions(c), 29_000)
        // a current count of the engine's decides, larger or smaller than a guess from the text
        XCTAssertEqual(ChatStore.contextForDecisions(conversation(turns(1, size: 40), used: 5000)), 5000)
        let long = turns(6, size: 4000)
        XCTAssertGreaterThan(ChatStore.contextEstimate(conversation(long)), 6000)
        XCTAssertEqual(ChatStore.contextForDecisions(conversation(long, used: 3500)), 3500)
        // stale, it is a guess again
        XCTAssertEqual(ChatStore.contextForDecisions(conversation(long, used: 3500, at: 2)),
                       ChatStore.contextEstimate(conversation(long)))
    }

    func testImagesCountInTheEstimate() {
        var message = ChatMessage(role: "user", content: "What is this?")
        message.imageURIs = ["data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB"]
        XCTAssertGreaterThanOrEqual(message.estimatedTokens, 2 * ChatMessage.tokensPerImageMegapixel)
    }

    func testAnAnswerCutByAFullContextIsToldFromOneCutAtItsLimit() {
        // the engine stops one token before the end of the context
        XCTAssertTrue(ChatStore.cutByContext(usage: (prompt: 8000, completion: 191), maxTokens: 4096, context: 8192))
        // fewer tokens than allowed: the context, not the limit, stopped it
        XCTAssertTrue(ChatStore.cutByContext(usage: (prompt: 3000, completion: 1500), maxTokens: 4096, context: 0))
        XCTAssertFalse(ChatStore.cutByContext(usage: (prompt: 300, completion: 4096), maxTokens: 4096, context: 16384))
        XCTAssertFalse(ChatStore.cutByContext(usage: nil, maxTokens: 4096, context: 16384))
        XCTAssertNotEqual(ChatStore.contextCutMessage(), ChatStore.cutAnswerMessage())
    }

    func testARequestThatDidNotFitSaysHowManyTokensItNeeded() {
        let error: [String: Any] = ["type": "exceed_context_size_error", "n_prompt_tokens": 8722, "n_ctx": 8192,
                                    "message": "request (8722 tokens) exceeds the available context size (8192 tokens)"]
        XCTAssertEqual(ChatStore.contextTokens(error), 8722)
        XCTAssertNil(ChatStore.contextTokens(["type": "server_error", "message": "Compute error."]))
        XCTAssertNil(ChatStore.contextTokens(nil))
    }

    func testWhatAnAgentMarkedToKeepGoesAfterTheSummaryAsWritten() {
        let kept = ["Figures used:\n1. 10, 100\n\nReference: abcd1234.0123456789abcdef01234567",
                    "Figures used:\n1. 5, 20"]
        let history = ChatStore.requestHistory(system: "Be brief.", summary: "They worked on two documents.",
                                               messages: turns(1, size: 10), from: 0, kept: kept)
        let system = history[0]["content"] as? String ?? ""
        XCTAssertTrue(system.hasPrefix("Be brief.\n\nSummary of the earlier part of this conversation:\nThey worked on two documents."))
        XCTAssertTrue(system.contains(ChatStore.keptHeader + "\n" + kept[0] + "\n---\n" + kept[1]), system)
        let agent = ChatStore.agentHistory(system: "", summary: "s", messages: turns(1, size: 10), from: 0, kept: kept)
        XCTAssertTrue((agent[0]["content"] as? String ?? "").contains("Reference: abcd1234.0123456789abcdef01234567"))
        // nothing to keep and no summary: no system part for them
        XCTAssertNil(ChatStore.summarySystem(nil, kept: []))
    }

    func testASummaryPromptFitsHalfTheContext() {
        let messages = ArraySlice(turns(20, size: 20_000))
        let prompt = ChatStore.summaryPrompt(prior: "Before.", messages: messages, limit: 8192)
        XCTAssertLessThan(prompt.count, 8192 * 2 + 4000)
        XCTAssertTrue(prompt.contains("Question 0") && prompt.contains("Answer 19") && prompt.contains(" […] "))
        // one that fits goes whole
        let small = ArraySlice(turns(2, size: 100))
        XCTAssertFalse(ChatStore.summaryPrompt(prior: nil, messages: small, limit: 8192).contains(" […] "))
    }

    func testNearTheEndOfTheContextOlderTurnsGoWithoutMediaAndWithShortResults() {
        var old = ChatMessage(role: "user", content: "Look at this")
        old.imageURIs = ["data:image/jpeg;base64,AAAA"]
        var tool = ChatMessage(role: "tool", content: String(repeating: "r", count: 30_000))
        tool.toolCallID = "call-1"
        var recent = ChatMessage(role: "user", content: "And this one")
        recent.imageURIs = ["data:image/jpeg;base64,BBBB"]
        let messages = [old, ChatMessage(role: "assistant", content: "A tree."), tool,
                        ChatMessage(role: "user", content: "Thanks"), ChatMessage(role: "assistant", content: "Sure."),
                        recent]
        let c = conversation(messages)
        XCTAssertEqual(ChatStore.trimStart(c, limit: 1_000_000), 0)
        let start = ChatStore.trimStart(c, limit: 1000)
        XCTAssertEqual(start, 3)
        let history = ChatStore.requestHistory(system: "", summary: nil, messages: messages, from: 0, trimBefore: start)
        XCTAssertEqual(history[0]["content"] as? String, "Look at this\n\n" + ChatStore.mediaLeftOut)
        XCTAssertLessThan((history[2]["content"] as? String ?? "").count, 2200)
        XCTAssertNotNil(history.last?["content"] as? [[String: Any]], "the newest message keeps its image")
    }
}
