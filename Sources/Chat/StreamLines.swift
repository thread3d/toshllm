// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// The lines of a server-sent event stream, split at LF with a CR before it dropped, and empty lines left out as
/// `AsyncBytes.lines` leaves them. That one also breaks at U+2028, U+2029 and U+0085, which a JSON string may carry
/// unescaped, and a chunk split there no longer parses.
struct StreamLines<Base: AsyncSequence>: AsyncSequence where Base.Element == UInt8 {
    typealias Element = String
    let base: Base

    struct AsyncIterator: AsyncIteratorProtocol {
        var bytes: Base.AsyncIterator
        var ended = false

        mutating func next() async throws -> String? {
            guard !ended else { return nil }
            var line: [UInt8] = []
            while true {
                guard let byte = try await bytes.next() else {
                    ended = true
                    return line.isEmpty ? nil : String(decoding: line, as: UTF8.self)
                }
                guard byte == 0x0A else {
                    line.append(byte)
                    continue
                }
                if line.last == 0x0D { line.removeLast() }
                if !line.isEmpty { return String(decoding: line, as: UTF8.self) }
            }
        }
    }

    func makeAsyncIterator() -> AsyncIterator { AsyncIterator(bytes: base.makeAsyncIterator()) }
}

extension AsyncSequence where Element == UInt8 {
    var streamLines: StreamLines<Self> { StreamLines(base: self) }
}
