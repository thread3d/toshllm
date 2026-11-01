// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class ChatMapBaseTests: XCTestCase {
    private var folder: URL!
    private var previous: String?

    override func setUpWithError() throws {
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("map-base-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("assets/fonts"),
                                                withIntermediateDirectories: true)
        try Data((0..<100).map { UInt8($0) }).write(to: folder.appendingPathComponent("b.pmtiles"))
        try Data("x".utf8).write(to: folder.appendingPathComponent("a.PMTiles"))
        try Data("notes".utf8).write(to: folder.appendingPathComponent("notes.txt"))
        previous = UserDefaults.standard.string(forKey: SettingsKeys.mapFolder)
        UserDefaults.standard.set(folder.path, forKey: SettingsKeys.mapFolder)
    }

    override func tearDownWithError() throws {
        UserDefaults.standard.set(previous, forKey: SettingsKeys.mapFolder)
        try? FileManager.default.removeItem(at: folder)
    }

    func testTheFolderGivesItsRegionsAndKnowsWhetherLabelsCanBeDrawn() throws {
        XCTAssertEqual(MapBase.regions, ["a.PMTiles", "b.pmtiles"])
        XCTAssertTrue(MapBase.labels)
        try FileManager.default.removeItem(at: folder.appendingPathComponent("assets"))
        XCTAssertFalse(MapBase.labels)
    }

    func testAPathCannotLeaveTheFolder() {
        XCTAssertNotNil(MapBase.file("b.pmtiles"))
        XCTAssertNil(MapBase.file("../b.pmtiles"))
        XCTAssertNil(MapBase.file("assets/../../etc/passwd"))
        XCTAssertNil(MapBase.file(""))
    }

    func testRegionsAreServedInByteRanges() throws {
        let file = try XCTUnwrap(MapBase.file("b.pmtiles"))
        let part = MapPageScheme.reply(file, range: "bytes=10-19")
        XCTAssertEqual(part.status, 206)
        XCTAssertEqual(part.data, Data((10..<20).map { UInt8($0) }))
        XCTAssertEqual(part.headers["Content-Range"], "bytes 10-19/100")
        XCTAssertEqual(part.headers["Content-Length"], "10")
        // a range past the end is cut at the end
        let tail = MapPageScheme.reply(file, range: "bytes=95-500")
        XCTAssertEqual(tail.data.count, 5)
        XCTAssertEqual(tail.headers["Content-Range"], "bytes 95-99/100")
        let whole = MapPageScheme.reply(file, range: nil)
        XCTAssertEqual(whole.status, 200)
        XCTAssertEqual(whole.data.count, 100)
        XCTAssertEqual(MapPageScheme.reply(nil, range: nil).status, 404)
    }

    func testOnlyAppleDataStaysOnTheAppleMap() {
        let plain = ChatMap.Layer(title: "Tract", geojson: Data(), complete: true)
        var route = plain
        route.apple = true
        XCTAssertFalse(plain.isApple)
        XCTAssertTrue(route.isApple)
    }
}
