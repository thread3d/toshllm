// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// A bundle without the identity keys of make-app.sh is plain ToshLLM, so its data, logs,
/// Keychain entries and updates stay where they always were.
final class AppIdentityTests: XCTestCase {
    func testABundleWithoutIdentityKeysIsToshLLM() {
        XCTAssertEqual(AppInfo.name, "ToshLLM")
        XCTAssertEqual(AppInfo.identifier, "dev.engel.toshllm")
        XCTAssertEqual(AppInfo.dataFolder, "ToshLLM")
        XCTAssertEqual(AppInfo.productVersion, AppInfo.version)
        XCTAssertTrue(AppInfo.updatesEnabled)
        XCTAssertFalse(AppInfo.isRebranded)
        XCTAssertEqual(AppInfo.branded("Welcome to ToshLLM"), "Welcome to ToshLLM")
        XCTAssertEqual(AppSupport.directory.lastPathComponent, "ToshLLM")
    }
}
