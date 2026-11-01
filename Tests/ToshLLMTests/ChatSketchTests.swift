// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class ChatSketchTests: XCTestCase {
    func testAnAgentsFigureIsReadAndMeasured() throws {
        let sketch = try XCTUnwrap(ChatSketch(["title": "Tract", "unit": "ft", "closed": true,
                                               "points": [[0, 0], [737.85, -426], [613.56, -649.46], [-123.25, -224.06], [0.06, -0.06]]]))
        XCTAssertEqual(sketch.unit, "ft")
        XCTAssertEqual(sketch.gap, hypot(0.06, 0.06), accuracy: 1e-9)
        XCTAssertEqual(sketch.scaleLength, 200)
        XCTAssertNil(sketch.corners)
    }

    func testWhatCannotBeDrawnIsRefused() {
        XCTAssertNil(ChatSketch(["points": [[0, 0]]]), "one point is not a figure")
        XCTAssertNil(ChatSketch(["points": [[0, 0], ["a", 1]]]))
        XCTAssertNil(ChatSketch(["points": [[0, 0], [1, 1]], "corners": [[0]]]))
        XCTAssertNil(ChatSketch(["points": Array(repeating: [0, 0], count: ChatSketch.maximumPoints + 1)]))
    }

    func testAFigureExportsForCADAndAsCorners() {
        let open = ChatSketch(title: nil, unit: "ft", points: [[0, 0], [100, 0], [100, 100], [1, 2]], closed: true)
        let dxf = open.dxf().components(separatedBy: "\n")
        XCTAssertEqual(dxf.filter { $0 == "VERTEX" }.count, 4)
        XCTAssertEqual(dxf.filter { $0 == "TEXT" }.count, 3, "the end that does not close is not a corner")
        XCTAssertTrue(dxf.contains("MISCLOSURE"))
        XCTAssertEqual(Array(dxf.suffix(3)), ["0", "EOF", ""])
        XCTAssertEqual(open.csv(), "point,east_ft,north_ft\n1,0.000,0.000\n2,100.000,0.000\n3,100.000,100.000\n")
        let shut = ChatSketch(title: nil, unit: "ft", points: [[0, 0], [100, 0], [100, 100], [0, 0]], closed: true)
        XCTAssertFalse(shut.dxf().contains("MISCLOSURE"))
    }

    func testAMapExportsOneFilePerKindOfGeometry() throws {
        let collection: [String: Any] = ["type": "FeatureCollection", "features": [
            ["type": "Feature", "geometry": ["type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]], "properties": ["name": "tract"]],
            ["type": "Feature", "geometry": ["type": "Point", "coordinates": [0, 0]], "properties": ["name": "corner 1"]],
            ["type": "Feature", "geometry": ["type": "LineString", "coordinates": [[0, 0], [1, 1]]], "properties": [:]]]]
        let points: [String: Any] = ["type": "FeatureCollection", "features": [
            ["type": "Feature", "geometry": ["type": "Point", "coordinates": [2, 2]], "properties": [:]]]]
        let map = ChatMap(layers: [
            ChatMap.Layer(title: "Tract: 5 acres", geojson: try JSONSerialization.data(withJSONObject: collection), complete: true),
            ChatMap.Layer(title: "Wells", geojson: try JSONSerialization.data(withJSONObject: points), complete: true)])
        let files = map.exportFiles()
        XCTAssertEqual(files.map(\.name), ["Tract- 5 acres - polygons.geojson", "Tract- 5 acres - lines.geojson",
                                            "Tract- 5 acres - points.geojson", "Wells.geojson"])
        let polygons = try XCTUnwrap(JSONSerialization.jsonObject(with: files[0].data) as? [String: Any])
        let features = try XCTUnwrap(polygons["features"] as? [[String: Any]])
        XCTAssertEqual(features.count, 1)
        XCTAssertEqual((features[0]["properties"] as? [String: Any])?["layer"] as? String, "Tract: 5 acres")
    }

    func testAFigureWithAnOriginExportsAtItsCoordinates() throws {
        let sketch = try XCTUnwrap(ChatSketch(["points": [[0, 0], [0, 100], [100, 100], [0, 0]], "unit": "ft", "closed": true,
                                               "origin": [2077075.85, 303260.44],
                                               "plane": "NAD27 Texas North Central (4202), US ft"]))
        XCTAssertEqual(sketch.plane, "NAD27 Texas North Central (4202), US ft")
        let csv = sketch.csv().split(separator: "\n")
        XCTAssertEqual(csv.first, "point,x_ft,y_ft")
        XCTAssertEqual(csv[2], "2,2077075.850,303360.440")
        XCTAssertTrue(sketch.dxf().contains("10\n2077175.850\n20\n303360.440"))
        // without an origin a figure stays in its own plane
        let plain = try XCTUnwrap(ChatSketch(["points": [[0, 0], [0, 100], [0, 0]], "unit": "ft", "plane": "ignored"]))
        XCTAssertNil(plain.plane)
        XCTAssertEqual(plain.csv().split(separator: "\n").first, "point,east_ft,north_ft")
        XCTAssertFalse(plain.dxf().contains("NOTICE"))
    }

    func testAShapefileZipHoldsTheFigureAtItsCoordinatesWithItsSystem() throws {
        let sketch = try XCTUnwrap(ChatSketch(["title": "Tract", "points": [[0, 0], [0, 100], [100, 100], [100, 0], [1, 1]],
                                               "unit": "ft", "closed": true, "origin": [2000, 3000],
                                               "plane": "Zone, US ft", "prj": "PROJCS[\"Zone\"]"]))
        let zip = sketch.shapefile(named: "Tract")
        // the local headers in order: name, then the stored bytes, whose CRC the header gives
        var files: [String: Data] = [:]
        var at = 0
        func le<T: FixedWidthInteger>(_ offset: Int, _: T.Type) -> T {
            zip.subdata(in: offset..<offset + MemoryLayout<T>.size).withUnsafeBytes { T(littleEndian: $0.loadUnaligned(as: T.self)) }
        }
        while le(at, UInt32.self) == 0x0403_4B50 {
            let size = Int(le(at + 18, UInt32.self)), nameLength = Int(le(at + 26, UInt16.self))
            let name = String(decoding: zip.subdata(in: at + 30..<at + 30 + nameLength), as: UTF8.self)
            let data = zip.subdata(in: at + 30 + nameLength..<at + 30 + nameLength + size)
            XCTAssertEqual(le(at + 14, UInt32.self), StoredZip.crc32(data), name)
            files[name] = data
            at += 30 + nameLength + size
        }
        XCTAssertEqual(Set(files.keys), ["Tract.shp", "Tract.shx", "Tract.dbf", "Tract.cpg", "Tract.prj"])
        XCTAssertEqual(String(decoding: try XCTUnwrap(files["Tract.prj"]), as: UTF8.self), "PROJCS[\"Zone\"]")
        let shp = try XCTUnwrap(files["Tract.shp"])
        func double(_ offset: Int) -> Double {
            Double(bitPattern: shp.subdata(in: offset..<offset + 8).withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) })
        }
        XCTAssertEqual(Array(shp.prefix(4)), [0, 0, 0x27, 0x0A])
        XCTAssertEqual(shp[32], 5, "a closed figure is a polygon")
        XCTAssertEqual(Int(shp[24]) << 24 | Int(shp[25]) << 16 | Int(shp[26]) << 8 | Int(shp[27]), shp.count / 2)
        // six points: the five drawn and the first again, clockwise from the start at its coordinates
        XCTAssertEqual(shp[100 + 8 + 40], 6)
        let first = 100 + 8 + 48
        XCTAssertEqual([double(first), double(first + 8)], [2000, 3000])
        XCTAssertEqual([double(first + 16), double(first + 24)], [2000, 3100])
        XCTAssertEqual([double(first + 80), double(first + 88)], [2000, 3000])
        let dbf = try XCTUnwrap(files["Tract.dbf"])
        XCTAssertEqual(dbf[4], 1, "one record")
        XCTAssertTrue(String(decoding: dbf, as: UTF8.self).contains("NAME"))
        XCTAssertFalse(String(decoding: dbf, as: UTF8.self).contains("NOTE"))
        // without coordinates there is no system to write
        let plain = try XCTUnwrap(ChatSketch(["points": [[0, 0], [0, 100]], "prj": "PROJCS[\"Zone\"]"]))
        XCTAssertNil(plain.prj)
    }

    func testANoticeTravelsWithEveryExportOnlyWhenGiven() throws {
        let sketch = try XCTUnwrap(ChatSketch(["points": [[0, 0], [0, 100], [100, 100], [0, 0]], "unit": "ft", "closed": true,
                                               "notice": "Not survey quality, \"approximate\""]))
        let csv = sketch.csv().split(separator: "\n")
        XCTAssertEqual(csv.first, "point,east_ft,north_ft,note")
        XCTAssertEqual(csv[1], "1,0.000,0.000,\"Not survey quality, \"\"approximate\"\"\"")
        XCTAssertTrue(sketch.dxf().contains("8\nNOTICE\n10\n0.000\n20\n"))
        XCTAssertNil(try XCTUnwrap(ChatSketch(["points": [[0, 0], [1, 1]], "notice": ""])).notice)
    }

    func testTheStatusAndAGapThePolygonClosesTravelWithEveryExport() throws {
        let draft = try XCTUnwrap(ChatSketch(["points": [[0, 0], [0, 100], [100, 100], [100, 0], [2, 1]], "unit": "ft",
                                              "closed": true, "standing": "review", "notice": "Draft"]))
        XCTAssertEqual(draft.csv().split(separator: "\n").first, "point,east_ft,north_ft,status,note")
        XCTAssertTrue(draft.csv().split(separator: "\n")[1].hasSuffix(",\"review\",\"Draft\""))
        XCTAssertTrue(draft.dxf().contains("8\nSTATUS\n"), "the drawing says what the figure is")
        XCTAssertTrue(draft.dxf().contains("STATUS: review"))
        let table = String(decoding: draft.shapefile(named: "T"), as: UTF8.self)
        XCTAssertTrue(table.contains("STATUS") && table.contains("review"))
        // the polygon closes what the points leave open, and its table says how much
        XCTAssertTrue(table.contains("GAP") && table.contains("2.236 ft closed by the polygon"), table)
        // a figure that ends where it began closes nothing, and one without a status carries none
        let exact = try XCTUnwrap(ChatSketch(["points": [[0, 0], [0, 100], [100, 100], [0, 0]], "unit": "ft",
                                              "closed": true]))
        XCTAssertNil(exact.closedGap)
        XCTAssertNil(exact.standing)
        XCTAssertFalse(exact.dxf().contains("STATUS"))
        XCTAssertFalse(String(decoding: exact.shapefile(named: "T"), as: UTF8.self).contains("STATUS"))
    }

    func testTheFigureStaysWithTheAnswerItCameWith() throws {
        var message = ChatMessage(role: "assistant", content: "It closes.")
        message.sketches = [ChatSketch(title: "Tract 1", unit: "ft", points: [[0, 0], [10, 0], [10, 10], [0, 0]], closed: true),
                            ChatSketch(title: "Tract 2", unit: "ft", points: [[0, 0], [5, 0], [5, 5], [0, 0]], closed: true)]
        let saved = try JSONDecoder().decode(ChatMessage.self, from: JSONEncoder().encode(message))
        XCTAssertEqual(saved.sketches, message.sketches)
        let older = try JSONDecoder().decode(ChatMessage.self, from: JSONEncoder().encode(ChatMessage(role: "assistant", content: "x")))
        XCTAssertNil(older.sketches)
    }
}
