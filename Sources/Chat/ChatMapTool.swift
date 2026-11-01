// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// A map the model draws in the chat from coordinates it gives: points, lines and areas, shown under the answer
/// apart from the maps of data an agent returns, since the positions may be the model's own estimate.
enum ChatMapTool {
    static let toolName = "show_map"
    static let maximumFeatures = 200
    static let maximumPoints = 2_000
    static let maximumCalledFeatures = 100
    static let maximumCalledPoints = 50

    static var isEnabled: Bool {
        UserDefaults.standard.object(forKey: SettingsKeys.chatMapToolEnabled) as? Bool ?? true
    }

    static let tool: BuiltinToolInfo = try! BuiltinToolInfo(
        displayName: "Map",
        name: toolName,
        writesData: false,
        definition: [
            "type": "function",
            "function": [
                "name": toolName,
                "description": "Draws a map under your answer from latitude and longitude. Use it when the user asks where places are, wants to see places, a route or an area, or gives coordinates. Each place, well, site or address is its own feature of kind point; use a line only for a route or path the user asks for, and an area only for a region, lot or boundary. A line is straight between its points: for a route by road use show_road_route when you have it, and otherwise draw the stops joined by a straight line, never a made-up road, and say it is a straight line, not the road. Positions you estimate yourself are shown as approximate.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "title": ["type": "string", "description": "What the map shows", "maxLength": 120],
                        "features": [
                            "type": "array",
                            "description": "What to draw",
                            // the engine turns these bounds into the grammar of the call, so a model that loops stops
                            "maxItems": maximumCalledFeatures,
                            "items": [
                                "type": "object",
                                "properties": [
                                    "kind": ["type": "string", "enum": ["point", "line", "area"]],
                                    "points": [
                                        "type": "array",
                                        "description": "One point for a point, two or more for a line, three or more for an area",
                                        "maxItems": maximumCalledPoints,
                                        "items": [
                                            "type": "object",
                                            "properties": ["lat": ["type": "number"], "lon": ["type": "number"]],
                                            "required": ["lat", "lon"]
                                        ]
                                    ],
                                    "label": ["type": "string", "maxLength": 120],
                                    "note": ["type": "string", "maxLength": 400]
                                ],
                                "required": ["kind", "points"]
                            ]
                        ]
                    ],
                    "required": ["features"]
                ]
            ]
        ])

    struct Drawn {
        let result: ToolExecutionResult
        let map: ChatMap?
    }

    /// The map for a call's arguments, and what the model is told back. Features with impossible coordinates
    /// or too few points are left out and named in the reply.
    static func draw(arguments: [String: Any], loc: Localizer, request: String = "") -> Drawn {
        let wanted = (arguments["features"] as? [[String: Any]]) ?? []
        var features: [[String: Any]] = []
        var skipped: [String] = []
        for (index, item) in wanted.prefix(maximumFeatures).enumerated() {
            let kind = (item["kind"] as? String ?? "point").lowercased()
            let label = (item["label"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "\(kind) \(index + 1)"
            let raw = (item["points"] as? [[String: Any]]) ?? []
            let points = raw.prefix(maximumPoints).compactMap(coordinate)
            if points.count < raw.prefix(maximumPoints).count {
                skipped.append("\(label): a coordinate is outside latitude -90...90 or longitude -180...180")
                continue
            }
            let geometry: [String: Any]
            switch kind {
            case "line" where points.count >= 2:
                geometry = ["type": "LineString", "coordinates": points]
            case "area" where points.count >= 3:
                let ring = points.first! == points.last! ? points : points + [points.first!]
                geometry = ["type": "Polygon", "coordinates": [ring]]
            case "point" where points.count == 1:
                geometry = ["type": "Point", "coordinates": points[0]]
            default:
                skipped.append("\(label): a \(kind) needs \(kind == "area" ? "three or more points" : kind == "line" ? "two or more points" : "exactly one point")")
                continue
            }
            var properties: [String: Any] = ["name": label]
            if let note = item["note"] as? String, !note.isEmpty { properties["note"] = note }
            features.append(["type": "Feature", "geometry": geometry, "properties": properties])
        }
        if wanted.count > maximumFeatures {
            skipped.append("\(wanted.count - maximumFeatures) features past the first \(maximumFeatures)")
        }
        guard !features.isEmpty,
              let geojson = try? JSONSerialization.data(withJSONObject: ["type": "FeatureCollection", "features": features])
        else {
            let reason = skipped.isEmpty ? "no features to draw" : skipped.joined(separator: "; ")
            return Drawn(result: ToolExecutionResult(content: "Nothing was drawn: \(reason).", isError: true), map: nil)
        }
        let title = (arguments["title"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        let layer = ChatMap.Layer(
            title: title ?? loc.t("Mapa", "Map"), geojson: geojson, linkTitle: nil, linkURL: nil,
            complete: skipped.isEmpty,
            subtitle: loc.t("Dibujado por el modelo · posiciones aproximadas", "Drawn by the model · approximate positions"),
            fields: [ChatMap.Field(name: "name", alias: loc.t("Nombre", "Name")),
                     ChatMap.Field(name: "note", alias: loc.t("Nota", "Note"))],
            count: features.count, drawn: true)
        var reply = "Drawn on a map under your answer: " + summary(features) + "."
        if !skipped.isEmpty { reply += " Left out: " + skipped.joined(separator: "; ") + "." }
        let missing = missingPairs(request, drawn: features)
        if !missing.isEmpty {
            reply += " The user's message gives coordinates the map does not have: "
                + missing.prefix(10).map { "\($0.lat), \($0.lon)" }.joined(separator: "; ")
                + ". Draw them too unless the user did not ask for them."
        }
        reply += " If you estimated the coordinates, say the positions are approximate."
        return Drawn(result: ToolExecutionResult(content: reply, isError: false), map: ChatMap(layers: [layer]))
    }

    /// What was drawn, by kind, so the answer can say it as it is: "2 points (W1, W2)", "1 line (Route)".
    private static func summary(_ features: [[String: Any]]) -> String {
        let kinds = [("Point", "point"), ("LineString", "line"), ("Polygon", "area")]
        return kinds.compactMap { geometry, word -> String? in
            let named = features.filter { ($0["geometry"] as? [String: Any])?["type"] as? String == geometry }
                .map { ($0["properties"] as? [String: Any])?["name"] as? String ?? "" }
            guard !named.isEmpty else { return nil }
            let shown = named.prefix(8).joined(separator: ", ") + (named.count > 8 ? ", …" : "")
            return "\(named.count) \(word)\(named.count == 1 ? "" : "s") (\(shown))"
        }.joined(separator: ", ")
    }

    /// Latitude, longitude pairs written in decimals in the user's message that no drawn point has: a pasted
    /// list of wells must reach the map whole.
    static func missingPairs(_ request: String, drawn features: [[String: Any]]) -> [(lat: Double, lon: Double)] {
        let pattern = #"(?<![\d.])(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)(?![\d.])"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        var drawn = Set<String>()
        for feature in features {
            let coordinates = (feature["geometry"] as? [String: Any])?["coordinates"]
            let points = (coordinates as? [Double]).map { [$0] } ?? (coordinates as? [[Double]])
                ?? (coordinates as? [[[Double]]])?.first ?? []
            for point in points where point.count == 2 { drawn.insert(String(format: "%.4f,%.4f", point[1], point[0])) }
        }
        let text = request as NSString
        return regex.matches(in: request, range: NSRange(location: 0, length: text.length)).compactMap { match in
            guard let lat = Double(text.substring(with: match.range(at: 1))),
                  let lon = Double(text.substring(with: match.range(at: 2))),
                  abs(lat) <= 90, abs(lon) <= 180,
                  !drawn.contains(String(format: "%.4f,%.4f", lat, lon)) else { return nil }
            return (lat, lon)
        }
    }

    /// [longitude, latitude], as GeoJSON writes it, from {"lat", "lon"}.
    private static func coordinate(_ point: [String: Any]) -> [Double]? {
        guard let lat = number(point["lat"] ?? point["latitude"]), let lon = number(point["lon"] ?? point["lng"] ?? point["longitude"]),
              lat.isFinite, lon.isFinite, abs(lat) <= 90, abs(lon) <= 180 else { return nil }
        return [lon, lat]
    }

    private static func number(_ value: Any?) -> Double? {
        if let value = value as? NSNumber { return value.doubleValue }
        if let value = value as? String { return Double(value.trimmingCharacters(in: .whitespaces)) }
        return nil
    }
}
