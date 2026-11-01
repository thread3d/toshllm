// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// A figure an agent attached to an answer in a plane of its own, such as a tract traced from its deed's calls:
/// drawn without a map, since nothing in it places it on the ground.
struct ChatSketch: Codable, Equatable {
    var title: String?
    /// The unit of the coordinates, such as "ft".
    var unit: String
    /// The path in order, east and north of its first point.
    var points: [[Double]]
    /// The points to mark and number; the path's own points when nil (a tract with curves marks only its corners).
    var corners: [[Double]]?
    /// The figure should end where it began: the gap left between its last point and its first is drawn in red.
    var closed: Bool
    /// Where the first point lies in a coordinate system the agent names, such as a State Plane zone: exports put the
    /// figure there.
    var origin: [Double]?
    var plane: String?
    /// A line every export carries, such as a disclaimer that the figure is not a survey.
    var notice: String?
    /// The coordinate system of `origin` as a shapefile's .prj holds it.
    var prj: String?
    /// What the agent says the figure is, such as whether it closes or is only a draft: every export carries it.
    var standing: String?

    static let maximumPoints = 5000

    init(title: String?, unit: String, points: [[Double]], corners: [[Double]]? = nil, closed: Bool,
         origin: [Double]? = nil, plane: String? = nil, notice: String? = nil, prj: String? = nil,
         standing: String? = nil) {
        self.title = title
        self.unit = unit
        self.points = points
        self.corners = corners
        self.closed = closed
        self.origin = origin
        self.plane = plane
        self.notice = notice
        self.prj = prj
        self.standing = standing
    }

    /// One of an agent's `tosh.sketches`, or nil when it is not a figure that can be drawn.
    init?(_ object: [String: Any]) {
        func pairs(_ value: Any?) -> [[Double]]? {
            guard let items = value as? [Any], (2...Self.maximumPoints).contains(items.count) else { return nil }
            let read = items.compactMap { item -> [Double]? in
                guard let pair = item as? [Any], pair.count == 2,
                      let x = (pair[0] as? NSNumber)?.doubleValue, let y = (pair[1] as? NSNumber)?.doubleValue,
                      x.isFinite, y.isFinite else { return nil }
                return [x, y]
            }
            return read.count == items.count ? read : nil
        }
        guard let points = pairs(object["points"]) else { return nil }
        let corners = object["corners"] == nil ? nil : pairs(object["corners"])
        if object["corners"] != nil && corners == nil { return nil }
        let origin = (object["origin"] as? [Any]).flatMap { pair -> [Double]? in
            let values = pair.compactMap { ($0 as? NSNumber)?.doubleValue }.filter(\.isFinite)
            return values.count == 2 && pair.count == 2 ? values : nil
        }
        self.init(title: (object["title"] as? String).map { String($0.prefix(120)) },
                  unit: String((object["unit"] as? String ?? "").prefix(12)), points: points, corners: corners,
                  closed: object["closed"] as? Bool ?? false, origin: origin,
                  plane: origin == nil ? nil : (object["plane"] as? String).map { String($0.prefix(120)) },
                  notice: (object["notice"] as? String).map { String($0.prefix(200)) }.flatMap { $0.isEmpty ? nil : $0 },
                  prj: origin == nil ? nil : (object["prj"] as? String).map { String($0.prefix(4000)) },
                  standing: (object["standing"] as? String).map { String($0.prefix(24)) }.flatMap { $0.isEmpty ? nil : $0 })
    }

    /// How far the last point ends from the first, in the sketch's unit.
    var gap: Double {
        guard let first = points.first, let last = points.last else { return 0 }
        return hypot(last[0] - first[0], last[1] - first[1])
    }

    /// The width of a figure, for the gap to be read against.
    var span: Double {
        let xs = points.map { $0[0] }, ys = points.map { $0[1] }
        return max((xs.max() ?? 0) - (xs.min() ?? 0), (ys.max() ?? 0) - (ys.min() ?? 0))
    }

    /// A round length near a quarter of the figure, for the scale bar.
    var scaleLength: Double {
        let target = max(span / 4, 1e-9)
        let power = pow(10, floor(log10(target)))
        return [1.0, 2, 5, 10].map { $0 * power }.last { $0 <= target } ?? power
    }
}

extension ChatSketch {
    private var cornerPoints: [[Double]] {
        let corners = self.corners ?? points
        return closed ? Array(corners.dropLast()) : corners
    }

    private static func number(_ value: Double) -> String { String(format: "%.3f", value) }

    /// A point where an export writes it: at the figure's coordinates when it has an origin, else from its first point.
    func placed(_ point: [Double]) -> [Double] {
        guard let origin else { return point }
        return [point[0] + origin[0], point[1] + origin[1]]
    }

    /// The figure as a CAD drawing in its own unit, at its coordinates or with the first point at the origin: an R12
    /// DXF, which CAD and GIS applications open. The path stays open as it was drawn, and what it lacks to close is a line
    /// of its own layer.
    func dxf() -> String {
        var out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009",
                   "9", "$INSUNITS", "70", unit.lowercased().hasPrefix("m") ? "6" : "2", "0", "ENDSEC",
                   "0", "SECTION", "2", "ENTITIES",
                   "0", "POLYLINE", "8", "FIGURE", "66", "1", "70", "0", "10", "0.0", "20", "0.0", "30", "0.0"]
        for point in points.map(placed) {
            out += ["0", "VERTEX", "8", "FIGURE", "10", Self.number(point[0]), "20", Self.number(point[1]), "30", "0.0"]
        }
        out += ["0", "SEQEND", "8", "FIGURE"]
        if closed, let first = points.first.map(placed), let last = points.last.map(placed), gap > span * 1e-4 {
            out += ["0", "LINE", "8", "MISCLOSURE", "10", Self.number(last[0]), "20", Self.number(last[1]), "30", "0.0",
                    "11", Self.number(first[0]), "21", Self.number(first[1]), "31", "0.0"]
        }
        let height = max(span / 60, 1e-3)
        for (index, corner) in cornerPoints.map(placed).enumerated() {
            out += ["0", "POINT", "8", "CORNERS", "10", Self.number(corner[0]), "20", Self.number(corner[1]), "30", "0.0",
                    "0", "TEXT", "8", "LABELS", "10", Self.number(corner[0] + height / 2), "20", Self.number(corner[1] + height / 2),
                    "30", "0.0", "40", Self.number(height), "1", "\(index + 1)"]
        }
        let placedPoints = (points + cornerPoints).map(placed)
        let left = placedPoints.map { $0[0] }.min() ?? 0, bottom = placedPoints.map { $0[1] }.min() ?? 0
        if let notice {
            out += ["0", "TEXT", "8", "NOTICE", "10", Self.number(left), "20", Self.number(bottom - 3 * height), "30", "0.0",
                    "40", Self.number(height), "1", notice.replacingOccurrences(of: "\n", with: " ")]
        }
        if let standing {
            out += ["0", "TEXT", "8", "STATUS", "10", Self.number(left), "20", Self.number(bottom - 5 * height), "30", "0.0",
                    "40", Self.number(height), "1", "STATUS: " + standing]
        }
        out += ["0", "ENDSEC", "0", "EOF"]
        return out.joined(separator: "\n") + "\n"
    }

    /// The numbered corners, at their coordinates or east and north of the first one.
    func csv() -> String {
        let suffix = unit.isEmpty ? "" : "_" + unit
        let header = origin == nil ? "point,east\(suffix),north\(suffix)" : "point,x\(suffix),y\(suffix)"
        // a column rather than a comment line, which some table imports reject
        let quoted = { (text: String) in ",\"" + text.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
        let extra = (standing.map(quoted) ?? "") + (notice.map(quoted) ?? "")
        let columns = (standing == nil ? "" : ",status") + (notice == nil ? "" : ",note")
        return ([header + columns] + cornerPoints.map(placed).enumerated().map { index, corner in
            "\(index + 1),\(Self.number(corner[0])),\(Self.number(corner[1]))" + extra
        }).joined(separator: "\n") + "\n"
    }
}

extension ChatStore {
    /// Keeps the figures an agent sent with its answer on that answer.
    func attachSketches(_ sketches: [ChatSketch], conversation id: UUID) {
        guard let i = conversations.firstIndex(where: { $0.id == id }),
              let j = conversations[i].messages.indices.last, conversations[i].messages[j].role == "assistant" else { return }
        conversations[i].messages[j].sketches = sketches
        save()
    }
}

/// An answer's figures: small, several to a row, and one at a time enlarged.
struct ChatSketchList: View {
    let sketches: [ChatSketch]

    @State private var enlarged: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let index = enlarged, sketches.indices.contains(index) {
                ChatSketchSection(sketch: sketches[index], enlarged: true) { enlarged = nil }
            }
            let rest = sketches.indices.filter { $0 != enlarged }
            if sketches.count == 1 {
                ForEach(rest, id: \.self) { index in
                    ChatSketchSection(sketch: sketches[index], enlarged: false) { enlarged = index }
                }
            } else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 10, alignment: .top)],
                          alignment: .leading, spacing: 10) {
                    ForEach(rest, id: \.self) { index in
                        ChatSketchSection(sketch: sketches[index], enlarged: false) { enlarged = index }
                    }
                }
            }
        }
    }
}

struct ChatSketchSection: View {
    let sketch: ChatSketch
    let enlarged: Bool
    let onResize: () -> Void

    @EnvironmentObject var loc: Localizer

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                if let title = sketch.title, !title.isEmpty {
                    Text(title).font(.caption.weight(.medium)).lineLimit(enlarged ? nil : 1)
                }
                Spacer()
                ChatFigureResizeButton(enlarged: enlarged, action: onResize)
            }
            ChatSketchCanvas(sketch: sketch)
                .frame(height: enlarged ? 360 : 170)
                .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
                .help(loc.t("Croquis según los datos de la respuesta: el punto 1 es el inicio y cada tramo va al número siguiente.",
                            "Sketch from the answer's data: point 1 is the start and each leg runs to the next number."))
            HStack(alignment: .top) {
                Text(caption).font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Menu(loc.t("Exportar", "Export")) {
                    Button(loc.t("Imagen PNG", "PNG image")) { save(png(), as: "png") }
                    Button(loc.t("DXF (CAD)", "DXF (CAD)")) { save(Data(sketch.dxf().utf8), as: "dxf") }
                    Button(loc.t("Esquinas en CSV", "Corners as CSV")) { save(Data(sketch.csv().utf8), as: "csv") }
                    Button(loc.t("Shapefile (ZIP)", "Shapefile (ZIP)")) {
                        save(sketch.shapefile(named: ChatMap.fileStem(sketch.title ?? loc.t("croquis", "sketch"))), as: "zip")
                    }
                }
                .menuStyle(.borderlessButton).fixedSize().font(.caption)
                .help(loc.t("PNG para un informe; DXF para aplicaciones CAD y GIS, en %@ desde el punto 1; CSV con las esquinas numeradas.",
                            "PNG for a report; DXF for CAD and GIS applications, in %@ from point 1; CSV with the numbered corners.",
                            sketch.unit))
            }
        }
    }

    private func png() -> Data? {
        let renderer = ImageRenderer(content: VStack(spacing: 0) {
            ChatSketchCanvas(sketch: sketch).frame(width: 1200, height: 800)
            if let notice = sketch.notice {
                Text(notice).font(.system(size: 20)).foregroundStyle(.black).padding(.bottom, 18)
            }
        }
        .frame(width: 1200).background(Color.white).environment(\.colorScheme, .light))
        renderer.scale = 2
        guard let image = renderer.cgImage else { return nil }
        return NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
    }

    private func save(_ data: Data?, as kind: String) {
        guard let data else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = ChatMap.fileStem(sketch.title ?? loc.t("croquis", "sketch")) + "." + kind
        panel.allowedContentTypes = [UTType(filenameExtension: kind) ?? .data]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? data.write(to: url)
    }

    private var caption: String {
        var text = sketch.plane.map { loc.t("Croquis en %@", "Sketch in %@", $0) }
            ?? loc.t("Croquis sin ubicar en el mapa", "Sketch, not placed on a map")
        if sketch.closed && sketch.gap > sketch.span * 1e-4 {
            text += " · " + loc.t("en rojo, lo que le falta para cerrar: %@ %@", "in red, what it lacks to close: %@ %@",
                                  ChatSketchCanvas.figure(sketch.gap), sketch.unit)
        }
        if let notice = sketch.notice { text += " · " + notice }
        return text
    }
}

/// Enlarges a figure or a map under an answer, or brings it back to its small size.
struct ChatFigureResizeButton: View {
    let enlarged: Bool
    let action: () -> Void

    @EnvironmentObject var loc: Localizer

    var body: some View {
        Button(action: action) {
            Image(systemName: enlarged ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
                .font(.caption)
        }
        .buttonStyle(.borderless)
        .help(enlarged ? loc.t("Reducir", "Make smaller") : loc.t("Ampliar", "Enlarge"))
    }
}

struct ChatSketchCanvas: View {
    let sketch: ChatSketch

    static func figure(_ value: Double) -> String {
        value >= 100 ? String(format: "%.0f", value) : value >= 10 ? String(format: "%.1f", value) : String(format: "%.2f", value)
    }

    var body: some View {
        Canvas { context, size in
            let corners = sketch.corners ?? sketch.points
            let all = sketch.points + corners
            let xs = all.map { $0[0] }, ys = all.map { $0[1] }
            let (minX, maxX, minY, maxY) = (xs.min() ?? 0, xs.max() ?? 0, ys.min() ?? 0, ys.max() ?? 0)
            let inset: CGFloat = 34
            let scale = min((size.width - 2 * inset) / max(maxX - minX, 1e-9), (size.height - 2 * inset) / max(maxY - minY, 1e-9))
            let offsetX = (size.width - (maxX - minX) * scale) / 2, offsetY = (size.height - (maxY - minY) * scale) / 2
            func place(_ p: [Double]) -> CGPoint {
                CGPoint(x: offsetX + (p[0] - minX) * scale, y: size.height - offsetY - (p[1] - minY) * scale)
            }

            var path = Path()
            path.addLines(sketch.points.map(place))
            context.stroke(path, with: .color(.accentColor), lineWidth: 2)
            let open = sketch.closed && sketch.gap > sketch.span * 1e-4
            if open, let first = sketch.points.first, let last = sketch.points.last {
                var gap = Path()
                gap.move(to: place(last))
                gap.addLine(to: place(first))
                context.stroke(gap, with: .color(.red), style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
                let end = place(last)
                context.fill(Path(ellipseIn: CGRect(x: end.x - 4.5, y: end.y - 4.5, width: 9, height: 9)), with: .color(.red))
            }
            // the last point of a closed figure is its first one again, or the red end of what it lacks
            let marked = sketch.closed ? Array(corners.dropLast()) : corners
            var labelled: [CGPoint] = []
            for (index, corner) in marked.enumerated() {
                let point = place(corner)
                let dot = CGRect(x: point.x - 3.5, y: point.y - 3.5, width: 7, height: 7)
                context.fill(Path(ellipseIn: dot), with: .color(.accentColor))
                if index == 0 {
                    // the start is a ring, which no accent colour hides
                    context.stroke(Path(ellipseIn: dot.insetBy(dx: -4, dy: -4)), with: .color(.primary), lineWidth: 1.5)
                }
                // a number that would cover another is left out; the figure enlarged has room for more
                guard index == 0 || labelled.allSatisfy({ hypot($0.x - point.x, $0.y - point.y) >= 16 }) else { continue }
                labelled.append(point)
                context.draw(Text("\(index + 1)").font(.caption2.weight(.semibold)).foregroundColor(.primary),
                             at: CGPoint(x: point.x + 9, y: point.y - 9))
            }

            // north up, and a scale bar of a round length
            let north = CGPoint(x: size.width - 18, y: 16)
            var arrow = Path()
            arrow.move(to: CGPoint(x: north.x, y: north.y + 18))
            arrow.addLine(to: north)
            arrow.move(to: CGPoint(x: north.x - 4, y: north.y + 6))
            arrow.addLine(to: north)
            arrow.addLine(to: CGPoint(x: north.x + 4, y: north.y + 6))
            context.stroke(arrow, with: .color(.secondary), lineWidth: 1.2)
            context.draw(Text("N").font(.caption2.weight(.bold)).foregroundColor(.secondary), at: CGPoint(x: north.x - 11, y: north.y + 4))
            let length = sketch.scaleLength * scale
            var bar = Path()
            bar.move(to: CGPoint(x: 14, y: size.height - 18))
            bar.addLine(to: CGPoint(x: 14 + length, y: size.height - 18))
            context.stroke(bar, with: .color(.secondary), lineWidth: 2)
            context.draw(Text("\(Self.figure(sketch.scaleLength)) \(sketch.unit)").font(.caption2).foregroundColor(.secondary),
                         at: CGPoint(x: 14 + length / 2, y: size.height - 8))
        }
    }
}
