// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// A ZIP of files stored without compression, which every unzip tool reads.
enum StoredZip {
    private static let table: [UInt32] = (0..<256).map { n in
        (0..<8).reduce(UInt32(n)) { c, _ in c & 1 == 1 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
    }

    static func crc32(_ data: Data) -> UInt32 {
        ~data.reduce(~UInt32(0)) { table[Int(($0 ^ UInt32($1)) & 0xFF)] ^ ($0 >> 8) }
    }

    static func archive(_ files: [(name: String, data: Data)]) -> Data {
        var out = Data(), directory = Data()
        for file in files {
            let name = Data(file.name.utf8), crc = crc32(file.data), size = UInt32(file.data.count)
            let offset = UInt32(out.count)
            // UTF-8 names, and 1 January 1980 as the date
            var shared = Data()
            shared.le16(20); shared.le16(0x0800); shared.le16(0); shared.le16(0); shared.le16(0x21)
            shared.le32(crc); shared.le32(size); shared.le32(size); shared.le16(UInt16(name.count)); shared.le16(0)
            out.le32(0x0403_4B50); out.append(shared); out.append(name); out.append(file.data)
            directory.le32(0x0201_4B50); directory.le16(20); directory.append(shared)
            directory.le16(0); directory.le16(0); directory.le16(0); directory.le32(0); directory.le32(offset)
            directory.append(name)
        }
        let start = UInt32(out.count)
        out.append(directory)
        out.le32(0x0605_4B50); out.le16(0); out.le16(0); out.le16(UInt16(files.count)); out.le16(UInt16(files.count))
        out.le32(UInt32(directory.count)); out.le32(start); out.le16(0)
        return out
    }
}

extension ChatSketch {
    /// The figure as a shapefile in a ZIP: a polygon when it should close, else a line, at its coordinates and with
    /// its coordinate system as the .prj when it has them, and its title, status, notice and any gap the polygon closes
    /// as attributes.
    func shapefile(named stem: String) -> Data {
        var path = points.map(placed)
        if closed {
            if let first = path.first, first != path.last { path.append(first) }
            // an outer ring runs clockwise
            let doubled = zip(path, path.dropFirst()).reduce(0.0) { $0 + $1.0[0] * $1.1[1] - $1.1[0] * $1.0[1] }
            if doubled > 0 { path.reverse() }
        }
        let xs = path.map { $0[0] }, ys = path.map { $0[1] }
        let box: [Double] = [xs.min() ?? 0, ys.min() ?? 0, xs.max() ?? 0, ys.max() ?? 0]
        let type: UInt32 = closed ? 5 : 3

        var record = Data()
        record.le32(type)
        box.forEach { record.leDouble($0) }
        record.le32(1); record.le32(UInt32(path.count)); record.le32(0)
        path.forEach { record.leDouble($0[0]); record.leDouble($0[1]) }

        func header(bytes: Int) -> Data {
            var out = Data()
            out.be32(9994)
            (0..<5).forEach { _ in out.be32(0) }
            out.be32(UInt32(bytes / 2)); out.le32(1000); out.le32(type)
            (box + [0, 0, 0, 0]).forEach { out.leDouble($0) }
            return out
        }
        var shp = header(bytes: 100 + 8 + record.count)
        shp.be32(1); shp.be32(UInt32(record.count / 2)); shp.append(record)
        var shx = header(bytes: 100 + 8)
        shx.be32(50); shx.be32(UInt32(record.count / 2))

        var files: [(name: String, data: Data)] = [
            ("\(stem).shp", shp), ("\(stem).shx", shx),
            ("\(stem).dbf", Self.table([("NAME", title ?? ""), ("STATUS", standing), ("NOTE", notice),
                                         ("GAP", closedGap)].compactMap { name, value in value.map { (name, $0) } })),
            ("\(stem).cpg", Data("UTF-8".utf8))]
        if let prj { files.append(("\(stem).prj", Data(prj.utf8))) }
        return StoredZip.archive(files)
    }

    /// What a polygon closes that its points leave open, which the drawing shows apart and a polygon cannot.
    var closedGap: String? {
        guard closed, gap > span * 1e-4 else { return nil }
        return String(format: "%.3f", gap) + (unit.isEmpty ? "" : " " + unit) + " closed by the polygon"
    }

    /// A dBase table of one record with a text field per value, cut at a character boundary to fit.
    private static func table(_ fields: [(name: String, value: String)]) -> Data {
        let width = 254
        var out = Data([0x03])
        let today = Calendar(identifier: .gregorian).dateComponents([.year, .month, .day], from: Date())
        out.append(contentsOf: [UInt8((today.year ?? 1980) - 1900), UInt8(today.month ?? 1), UInt8(today.day ?? 1)])
        out.le32(1); out.le16(UInt16(32 + 32 * fields.count + 1)); out.le16(UInt16(1 + width * fields.count))
        out.append(Data(count: 20))
        for field in fields {
            out.append(Data(field.name.utf8.prefix(10)) + Data(count: 11 - min(10, field.name.utf8.count)))
            out.append(contentsOf: [UInt8(ascii: "C"), 0, 0, 0, 0, UInt8(width), 0])
            out.append(Data(count: 14))
        }
        out.append(0x0D)
        out.append(0x20)
        for field in fields {
            var bytes = Data()
            for character in field.value {
                let next = Data(String(character).utf8)
                if bytes.count + next.count > width { break }
                bytes.append(next)
            }
            out.append(bytes + Data(repeating: 0x20, count: width - bytes.count))
        }
        out.append(0x1A)
        return out
    }
}

private extension Data {
    mutating func le16(_ value: UInt16) { Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) } }
    mutating func le32(_ value: UInt32) { Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) } }
    mutating func be32(_ value: UInt32) { Swift.withUnsafeBytes(of: value.bigEndian) { append(contentsOf: $0) } }
    mutating func leDouble(_ value: Double) { le64(value.bitPattern) }
    mutating func le64(_ value: UInt64) { Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) } }
}
