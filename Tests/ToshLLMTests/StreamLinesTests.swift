// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// A server-sent event line breaks only at LF: text the engine sends unescaped inside a JSON string, such as U+2028,
/// stays in its chunk, which still parses.
final class StreamLinesTests: XCTestCase {
    private struct Bytes: AsyncSequence {
        typealias Element = UInt8
        let data: [UInt8]
        struct AsyncIterator: AsyncIteratorProtocol {
            var index = 0
            let data: [UInt8]
            mutating func next() async -> UInt8? {
                guard index < data.count else { return nil }
                defer { index += 1 }
                return data[index]
            }
        }
        func makeAsyncIterator() -> AsyncIterator { AsyncIterator(data: data) }
    }

    private func lines(_ text: String) async throws -> [String] {
        var out: [String] = []
        for try await line in Bytes(data: Array(text.utf8)).streamLines { out.append(line) }
        return out
    }

    func testALineBreaksOnlyAtLineFeed() async throws {
        for separator in ["\u{2028}", "\u{2029}", "\u{0085}"] {
            let chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"a\(separator)b\"}}]}"
            let found = try await lines(chunk + "\n\n" + "data: [DONE]\n\n")
            XCTAssertEqual(found, [chunk, "data: [DONE]"], separator.unicodeScalars.first!.debugDescription)
            var accumulator = ChatStreamAccumulator()
            _ = try accumulator.consume(found[0])
            XCTAssertEqual(accumulator.visible, "a\(separator)b")
        }
    }

    func testCRLFEmptyLinesAndAnUnterminatedLastLine() async throws {
        let crlf = try await lines("data: 1\r\n\r\n: ping\r\ndata: 2")
        XCTAssertEqual(crlf, ["data: 1", ": ping", "data: 2"])
        let empty = try await lines("")
        let blank = try await lines("\n\n\n")
        XCTAssertEqual(empty + blank, [])
        // a character of several bytes is decoded whole
        let wide = try await lines("data: ñ😀°\n")
        XCTAssertEqual(wide, ["data: ñ😀°"])
    }
}
