// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class ChatMapToolTests: XCTestCase {
    private func features(_ map: ChatMap?) throws -> [[String: Any]] {
        let layer = try XCTUnwrap(map?.layers.first)
        let collection = try XCTUnwrap(try JSONSerialization.jsonObject(with: layer.geojson) as? [String: Any])
        return collection["features"] as? [[String: Any]] ?? []
    }

    func testPointsLinesAndAreasAreDrawnAsTheModelsOwn() throws {
        let drawn = ChatMapTool.draw(arguments: [
            "title": "Route",
            "features": [
                ["kind": "point", "points": [["lat": 10.48, "lon": -66.90]], "label": "Caracas"],
                ["kind": "line", "points": [["lat": 10.48, "lon": -66.90], ["lat": 4.71, "lon": -74.07]], "label": "Road"],
                ["kind": "area", "points": [["lat": 0, "lon": 0], ["lat": 0, "lon": 1], ["lat": 1, "lon": 1]], "note": "Lot"]
            ]], loc: Localizer())
        XCTAssertFalse(drawn.result.isError)
        let layer = try XCTUnwrap(drawn.map?.layers.first)
        XCTAssertEqual(layer.drawn, true)
        XCTAssertEqual(layer.title, "Route")
        XCTAssertEqual(layer.count, 3)
        XCTAssertTrue(layer.complete)
        let shapes = try features(drawn.map)
        XCTAssertEqual(shapes.compactMap { ($0["geometry"] as? [String: Any])?["type"] as? String }, ["Point", "LineString", "Polygon"])
        // GeoJSON writes longitude first, and an area's ring is closed
        let point = (shapes[0]["geometry"] as? [String: Any])?["coordinates"] as? [Double]
        XCTAssertEqual(point, [-66.90, 10.48])
        let ring = ((shapes[2]["geometry"] as? [String: Any])?["coordinates"] as? [[[Double]]])?.first
        XCTAssertEqual(ring?.first, ring?.last)
        XCTAssertEqual(ring?.count, 4)
    }

    func testImpossibleCoordinatesAndShortShapesAreLeftOutAndNamed() throws {
        let drawn = ChatMapTool.draw(arguments: ["features": [
            ["kind": "point", "points": [["lat": 95, "lon": 0]], "label": "North of the pole"],
            ["kind": "area", "points": [["lat": 0, "lon": 0], ["lat": 1, "lon": 1]], "label": "Two corners"],
            ["kind": "point", "points": [["lat": "40.4", "lon": "-3.7"]], "label": "Madrid"]
        ]], loc: Localizer())
        XCTAssertFalse(drawn.result.isError)
        XCTAssertEqual(try features(drawn.map).count, 1)
        XCTAssertEqual(drawn.map?.layers.first?.complete, false)
        XCTAssertTrue(drawn.result.content.contains("North of the pole"))
        XCTAssertTrue(drawn.result.content.contains("Two corners"))
    }

    func testCoordinatesTheUserGaveAndTheMapLacksAreNamed() {
        let drawn = ChatMapTool.draw(arguments: ["features": [["kind": "point", "points": [["lat": 32.751, "lon": -97.331]], "label": "W1"]]],
                                     loc: Localizer(), request: "Mark these wells: W1 32.751, -97.331; W2 32.760, -97.340")
        XCTAssertTrue(drawn.result.content.contains("32.76, -97.34"))
        let whole = ChatMapTool.draw(arguments: ["features": [
            ["kind": "point", "points": [["lat": 32.751, "lon": -97.331]]], ["kind": "point", "points": [["lat": 32.760, "lon": -97.340]]]]],
                                     loc: Localizer(), request: "Mark these wells: W1 32.751, -97.331; W2 32.760, -97.340")
        XCTAssertFalse(whole.result.content.contains("does not have"))
    }

    func testNothingToDrawIsAnError() {
        let drawn = ChatMapTool.draw(arguments: ["features": []], loc: Localizer())
        XCTAssertTrue(drawn.result.isError)
        XCTAssertNil(drawn.map)
    }

    func testTheMapNeverTakesATurnFromTheServerAgent() {
        XCTAssertFalse(ChatStore.hasOwnTools([ChatMapTool.tool]))
        XCTAssertTrue(ChatStore.hasOwnTools([ChatMapTool.tool] + ChatMemoryService.tools))
        XCTAssertTrue(ChatToolsService.isAlwaysAllowed(ChatMapTool.toolName))
    }

    func testTheAnswerTakesTheMapDrawnInItsTurn() throws {
        let drawn = ChatMapTool.draw(arguments: ["features": [["kind": "point", "points": [["lat": 1, "lon": 2]]]]],
                                     loc: Localizer())
        var call = ChatMessage(role: "assistant", content: "")
        call.map = drawn.map
        var earlier = ChatMessage(role: "assistant", content: "Older answer")
        earlier.map = drawn.map
        var conversation = Conversation(title: "Map", messages: [
            ChatMessage(role: "user", content: "Where?"), earlier, ChatMessage(role: "user", content: "And now?"),
            call, ChatMessage(role: "assistant", content: "Here.")])
        ChatStore.carryDrawnMap(&conversation, to: 4)
        XCTAssertNil(conversation.messages[3].map)
        XCTAssertEqual(conversation.messages[4].map?.layers.first?.drawn, true)
        // a map of an earlier turn stays where it was
        XCTAssertNotNil(conversation.messages[1].map)
    }

    func testARoadRouteNeedsTwoValidStopsAndAsksFirst() {
        let ok = ChatRoadRoute.stops(["stops": [["lat": 40.4, "lon": -3.7, "label": "Madrid"], ["lat": 41.6, "lon": -0.9]]])
        XCTAssertNil(ok.problem)
        XCTAssertEqual(ok.stops.map(\.label), ["Madrid", "stop 2"])
        XCTAssertNotNil(ChatRoadRoute.stops(["stops": [["lat": 40.4, "lon": -3.7]]]).problem)
        XCTAssertNotNil(ChatRoadRoute.stops(["stops": [["lat": 140, "lon": -3.7], ["lat": 41.6, "lon": -0.9]]]).problem)
        // it sends the points to Apple: never run without the user's approval, nor counted as the chat's own tool
        XCTAssertFalse(ChatToolsService.isAlwaysAllowed(ChatRoadRoute.toolName))
        XCTAssertFalse(ChatStore.hasOwnTools([ChatMapTool.tool, ChatRoadRoute.tool]))
        XCTAssertTrue(ChatRoadRoute.declined.contains("straight line"))
        XCTAssertEqual(ChatRoadRoute.duration(201 * 60), "3 h 21 min")
        XCTAssertEqual(ChatRoadRoute.duration(45 * 60), "45 min")
    }

    func testMapsOfSeveralCallsInATurnAllReachTheAnswer() throws {
        let first = ChatMapTool.draw(arguments: ["features": [["kind": "point", "points": [["lat": 1, "lon": 2]]]]], loc: Localizer())
        let second = ChatMapTool.draw(arguments: ["features": [["kind": "point", "points": [["lat": 3, "lon": 4]]]]], loc: Localizer())
        var one = ChatMessage(role: "assistant", content: "")
        one.map = first.map
        var two = ChatMessage(role: "assistant", content: "")
        two.map = second.map
        var conversation = Conversation(title: "Map", messages: [ChatMessage(role: "user", content: "Both"), one, two,
                                                                 ChatMessage(role: "assistant", content: "Done.")])
        ChatStore.carryDrawnMap(&conversation, to: 3)
        XCTAssertEqual(conversation.messages[3].map?.layers.count, 2)
    }

    func testMapsSavedBeforeTheDrawnFlagStillLoad() throws {
        let saved = #"{"layers": [{"title": "States", "geojson": "e30=", "complete": true}]}"#
        let map = try JSONDecoder().decode(ChatMap.self, from: Data(saved.utf8))
        XCTAssertNil(map.layers.first?.drawn)
    }
}
