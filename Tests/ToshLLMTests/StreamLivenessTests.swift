// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class StreamLivenessTests: XCTestCase {
    func testALongPromptWithProgressIsAlive() {
        let start = Date()
        var liveness = StreamLiveness(now: start)
        // a slow GPU reports a batch of prompt every minute for half an hour
        for minute in 1...30 {
            let now = start.addingTimeInterval(Double(minute) * 60)
            XCTAssertFalse(liveness.stalled(now: now))
            liveness.note(text: false, now: now)
        }
        XCTAssertFalse(liveness.sawText)
    }

    func testSilenceDuringThePromptStalls() {
        let start = Date()
        let liveness = StreamLiveness(now: start)
        XCTAssertFalse(liveness.stalled(now: start.addingTimeInterval(170)))
        XCTAssertTrue(liveness.stalled(now: start.addingTimeInterval(190)))
    }

    func testOnceTextStartsItHasToKeepComing() {
        let start = Date()
        var liveness = StreamLiveness(now: start)
        liveness.note(text: false, now: start.addingTimeInterval(100))
        liveness.note(text: true, now: start.addingTimeInterval(120))
        XCTAssertFalse(liveness.stalled(now: start.addingTimeInterval(145)))
        XCTAssertTrue(liveness.stalled(now: start.addingTimeInterval(155)))
    }
}
