// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class EngineExtensionsTests: XCTestCase {
    private var ids: [String] = []

    private func manifest(_ id: String, agent: Bool = false, mathTools: Bool = false, extra: String = "") -> String {
        """
        {"format": 1, "id": "\(id)", "version": "1.0", "name": {"es": "Mapas", "en": "Maps"},
         "servers": {"maps": {"command": "{python}", "args": ["-I", "{extension}/maps.py"],
                              "env": {"KEY": "{trust_key}"}, "timeout_ms": 30000,
                              "enabled_by": "TEST_MAPS_ON", "agent_only": true},
                     "cloud": {"url": "https://mcp.example/mcp", "transport": "http", "timeout_ms": 20000,
                               "header_env": {"Authorization": "TEST_MAPS_TOKEN"}}}
         \(agent ? #", "agent": {"command": "{python}", "args": ["{extension}/agent.py", "--runtime", "{runtime}"]}"# : "")
         \(mathTools ? #", "math_tools": true"# : "")
         \(extra),
         "settings": [{"id": "maps", "title": {"es": "Mapas", "en": "Maps"},
           "notice": {"when_empty": ["TEST_MAPS_URL"], "text": {"es": "Prueba", "en": "Test"}},
           "fields": [
             {"key": "TEST_MAPS_ON", "kind": "toggle", "default": "1",
              "title": {"es": "Activo", "en": "On"}, "help": {"es": "a", "en": "a"}},
             {"key": "TEST_MAPS_URL", "kind": "text", "title": {"es": "URL", "en": "URL"}, "help": {"es": "b", "en": "b"}},
             {"key": "TEST_MAPS_TOKEN", "kind": "text", "title": {"es": "T", "en": "T"}, "help": {"es": "c", "en": "c"}}]}]}
        """
    }

    private func folder() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tosh-ext-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }

    @discardableResult
    private func write(_ json: String, in root: URL, as name: String) throws -> URL {
        let folder = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: folder.appendingPathComponent("extension.json"))
        return folder
    }

    private func id() -> String {
        let id = "test-" + UUID().uuidString.prefix(8).lowercased()
        ids.append(id)
        return id
    }

    override func tearDown() {
        let defaults = UserDefaults.standard
        for id in ids {
            let values = ["TEST_MAPS_ON", "TEST_MAPS_URL", "TEST_MAPS_TOKEN"].map { SettingsKeys.extensionValue(id, $0) }
            for key in [SettingsKeys.extensionApproval(id), SettingsKeys.extensionOff(id)] + values {
                defaults.removeObject(forKey: key)
            }
        }
        super.tearDown()
    }

    private func runtime() throws -> URL {
        let resources = try folder()
        let bin = resources.appendingPathComponent("tosh-sympy/python/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.copyItem(atPath: "/usr/bin/true", toPath: bin.appendingPathComponent("python3").path)
        return resources
    }

    func testAFolderOfExtensionsAndASingleOneAreFound() throws {
        let root = try folder()
        let (a, b) = (id(), id())
        try write(manifest(a), in: root, as: "a")
        try write(manifest(b), in: root, as: "b")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("not-one"), withIntermediateDirectories: true)
        XCTAssertEqual(EngineExtensions.discover(in: root, bundled: false).extensions.map(\.id).sorted(), [a, b].sorted())
        let single = try write(manifest(id()), in: try folder(), as: "only")
        XCTAssertEqual(EngineExtensions.discover(in: single, bundled: false).extensions.count, 1)
    }

    func testBrokenManifestsAreReportedNotLoaded() throws {
        let root = try folder()
        try write("{", in: root, as: "json")
        try write(manifest("Bad Id"), in: root, as: "id")
        try write(manifest(id()).replacingOccurrences(of: #""format": 1"#, with: #""format": 2"#), in: root, as: "format")
        try write(manifest(id()).replacingOccurrences(of: #""maps": {"command""#, with: #""sympy": {"command""#),
                  in: root, as: "reserved")
        let found = EngineExtensions.discover(in: root, bundled: false)
        XCTAssertTrue(found.extensions.isEmpty)
        XCTAssertEqual(found.problems.count, 4)
    }

    func testAnIdTakenInsideTheAppIsNotLoadedAgainFromTheFolder() throws {
        let shared = id()
        let inside = try folder()
        let outside = try folder()
        try write(manifest(shared), in: inside, as: "x")
        try write(manifest(shared), in: outside, as: "x")
        let all = EngineExtensions.all(bundled: inside, chosen: outside)
        XCTAssertEqual(all.extensions.map(\.bundled), [true])
        XCTAssertEqual(all.problems.count, 1)
    }

    func testAFolderExtensionNeedsApprovalOfThisExactFile() throws {
        let root = try folder()
        let ext = id()
        let path = try write(manifest(ext), in: root, as: "x")
        var item = try XCTUnwrap(EngineExtensions.discover(in: root, bundled: false).extensions.first)
        XCTAssertFalse(EngineExtensions.isEnabled(item))
        EngineExtensions.setEnabled(item, true)
        XCTAssertTrue(EngineExtensions.isEnabled(item))
        try Data(manifest(ext, extra: #", "author": "changed""#).utf8).write(to: path.appendingPathComponent("extension.json"))
        item = try XCTUnwrap(EngineExtensions.discover(in: root, bundled: false).extensions.first)
        XCTAssertFalse(EngineExtensions.isEnabled(item), "a changed extension.json needs a new approval")
    }

    func testAnExtensionInsideTheAppStartsOnAndCanBeSwitchedOff() throws {
        let root = try folder()
        try write(manifest(id()), in: root, as: "x")
        let item = try XCTUnwrap(EngineExtensions.discover(in: root, bundled: true).extensions.first)
        XCTAssertTrue(EngineExtensions.isEnabled(item))
        EngineExtensions.setEnabled(item, false)
        XCTAssertFalse(EngineExtensions.isEnabled(item))
    }

    func testOnlyOneExtensionBringsItsAgent() throws {
        let root = try folder()
        try write(manifest(id(), agent: true), in: root, as: "a")
        try write(manifest(id(), agent: true), in: root, as: "b")
        try write(manifest(id()), in: root, as: "c")
        let active = EngineExtensions.active(EngineExtensions.discover(in: root, bundled: true).extensions)
        XCTAssertEqual(active.filter(\.hasAgent).count, 1)
        XCTAssertEqual(active.count, 2)
    }

    func testEnvironmentCarriesSetValuesAndLeavesOffTogglesOut() throws {
        let root = try folder()
        try write(manifest(id()), in: root, as: "x")
        let item = try XCTUnwrap(EngineExtensions.discover(in: root, bundled: true).extensions.first)
        XCTAssertEqual(EngineExtensions.environment([item]), ["TEST_MAPS_ON": "1"])
        EngineExtensions.setValue("https://maps.example", for: item.fields[1], of: item)
        EngineExtensions.setValue("0", for: item.fields[0], of: item)
        XCTAssertEqual(EngineExtensions.environment([item]), ["TEST_MAPS_URL": "https://maps.example"])
    }

    func testChoicesKeepTheManifestsOrderAndReachTheEngineEvenWhenEmpty() throws {
        let root = try folder()
        let option = { (value: String) in #"{"value": "\#(value)", "title": {"es": "\#(value)", "en": "\#(value)"}}"# }
        try write("""
            {"format": 1, "id": "\(id())", "version": "1.0", "name": {"es": "Datos", "en": "Data"},
             "settings": [{"id": "data", "title": {"es": "Datos", "en": "Data"}, "fields": [
               {"key": "TEST_PICK", "kind": "choices", "default": "c,a", "options": [\(option("a")), \(option("b")), \(option("c"))],
                "title": {"es": "Fuentes", "en": "Sources"}, "help": {"es": "d", "en": "d"}}]}]}
            """, in: root, as: "x")
        let item = try XCTUnwrap(EngineExtensions.discover(in: root, bundled: true).extensions.first)
        let field = try XCTUnwrap(item.fields.first)
        XCTAssertEqual(EngineExtensions.chosen(field, of: item), ["a", "c"])
        XCTAssertEqual(EngineExtensions.environment([item]), ["TEST_PICK": "a,c"])
        EngineExtensions.setValue("b, zz", for: field, of: item)
        XCTAssertEqual(EngineExtensions.chosen(field, of: item), ["b"], "an option the manifest no longer lists is dropped")
        EngineExtensions.setValue("", for: field, of: item)
        XCTAssertEqual(EngineExtensions.environment([item]), ["TEST_PICK": ""])
    }

    func testEntriesFillPlaceholdersFollowTheirSwitchAndRelayRemoteServers() throws {
        let root = try folder()
        try write(manifest(id(), agent: true), in: root, as: "x")
        let item = try XCTUnwrap(EngineExtensions.discover(in: root, bundled: true).extensions.first)
        let path = item.folder
        let rt = URL(fileURLWithPath: "/rt")
        let on = EngineExtensions.engineEntries([item], runtime: rt, trustKey: "k")
        let maps = try XCTUnwrap(on.servers["maps"] as? [String: Any])
        XCTAssertEqual(maps["command"] as? String, "/rt/python/bin/python3")
        XCTAssertEqual(maps["args"] as? [String], ["-I", path.path + "/maps.py"])
        XCTAssertEqual(maps["env"] as? [String: String], ["KEY": "k"])
        XCTAssertEqual(on.agentOnly, ["maps"])
        XCTAssertEqual(on.agent?["args"] as? [String], [path.path + "/agent.py", "--runtime", "/rt"])
        let cloud = try XCTUnwrap(on.servers["cloud"] as? [String: Any])
        XCTAssertEqual(cloud["args"] as? [String], ["-I", "-B", "/rt/tosh_sympy/remote.py", "--url", "https://mcp.example/mcp",
                                                    "--timeout", "20", "--transport", "http",
                                                    "--header-env", "Authorization=TEST_MAPS_TOKEN"])

        EngineExtensions.setValue("0", for: item.fields[0], of: item)
        XCTAssertNil(EngineExtensions.engineEntries([item], runtime: rt, trustKey: "k").servers["maps"])
    }

    func testAnExtensionAgentReplacesTheMathAgentAndBringsTheMathServers() throws {
        let root = try folder()
        try write(manifest(id(), agent: true, mathTools: true), in: root, as: "x")
        let item = try XCTUnwrap(EngineExtensions.discover(in: root, bundled: true).extensions.first)
        let args = SymPyToolsService.serverArguments(enabled: false, resources: try runtime(), extensions: [item])
        let json = try XCTUnwrap(args.firstIndex(of: "--mcp-servers-json").map { args[$0 + 1] })
        let servers = try XCTUnwrap((JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])?["mcpServers"]
            as? [String: Any])
        XCTAssertEqual(Set(servers.keys), ["maps", "cloud", "sympy", "scientific", SymPyToolsService.agentName])
        let agent = try XCTUnwrap(servers[SymPyToolsService.agentName] as? [String: Any])
        XCTAssertTrue((agent["args"] as? [String])?.first?.hasSuffix("/agent.py") == true)
        XCTAssertEqual(args.firstIndex(of: "--mcp-agent-only").map { args[$0 + 1] }, "maps")
    }

    func testAnAgentThatTakesTheChatsToolsGetsThemFromTheEngine() throws {
        let root = try folder()
        try write(manifest(id(), agent: true, extra: #", "client_tools": true"#), in: root, as: "x")
        let item = try XCTUnwrap(EngineExtensions.discover(in: root, bundled: true).extensions.first)
        XCTAssertTrue(item.agentTakesClientTools)
        XCTAssertTrue(SymPyToolsService.serverArguments(enabled: false, resources: try runtime(), extensions: [item])
            .contains("--mcp-agent-client-tools"))
        let plain = try folder()
        try write(manifest(id(), agent: true), in: plain, as: "y")
        let other = try XCTUnwrap(EngineExtensions.discover(in: plain, bundled: true).extensions.first)
        XCTAssertFalse(other.agentTakesClientTools)
        XCTAssertFalse(SymPyToolsService.serverArguments(enabled: false, resources: try runtime(), extensions: [other])
            .contains("--mcp-agent-client-tools"))
    }

    func testTheChatCallsOnlyTheServersNotKeptForTheAgent() throws {
        let root = try folder()
        try write(manifest(id()), in: root, as: "x")
        let found = EngineExtensions.discover(in: root, bundled: true).extensions
        XCTAssertEqual(EngineExtensions.chatToolPrefixes(found), ["cloud_"])
    }

    func testTheExampleExtensionsLoad() throws {
        let examples = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("examples/extensions")
        let found = EngineExtensions.discover(in: examples, bundled: false)
        XCTAssertEqual(found.extensions.map(\.id).sorted(), ["clock", "places"])
        XCTAssertTrue(found.problems.isEmpty)
        // the complete example shows every kind of setting and an agent
        let places = try XCTUnwrap(found.extensions.first { $0.id == "places" })
        XCTAssertEqual(Set(places.fields.map(\.kind.rawValue)),
                       ["toggle", "text", "secret", "choice", "choices", "json", "paths"])
        XCTAssertTrue(places.hasAgent)
    }

    func testTheMathAgentOffersTheToolsOfAnExtensionWithoutAgent() throws {
        let root = try folder()
        try write(manifest(id()), in: root, as: "x")
        let item = try XCTUnwrap(EngineExtensions.discover(in: root, bundled: true).extensions.first)
        let args = SymPyToolsService.serverArguments(enabled: false, resources: try runtime(), extensions: [item])
        let json = try XCTUnwrap(args.firstIndex(of: "--mcp-servers-json").map { args[$0 + 1] })
        let servers = try XCTUnwrap((JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])?["mcpServers"]
            as? [String: Any])
        let env = (servers[SymPyToolsService.agentName] as? [String: Any])?["env"] as? [String: String]
        XCTAssertEqual(env?["TOSH_AGENT_SERVERS"], "cloud,maps")
    }

    func testWithoutExtensionsNothingChanges() throws {
        XCTAssertEqual(SymPyToolsService.serverArguments(enabled: false, resources: try runtime()), [])
    }

    func testAProductBringsItsOwnDefaults() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let defaults: NSDictionary = [SettingsKeys.chatMaxImageMegapixels: 4.0]
        XCTAssertTrue(defaults.write(to: folder.appendingPathComponent("AppDefaults.plist"), atomically: true))
        let values = ProductDefaults.values(in: try XCTUnwrap(Bundle(url: folder)))
        XCTAssertEqual((values[SettingsKeys.chatMaxImageMegapixels] as? NSNumber)?.doubleValue, 4)
        XCTAssertTrue(ProductDefaults.values().isEmpty, "a plain build keeps ToshLLM's defaults")
        XCTAssertEqual(ProductDefaults.double(SettingsKeys.chatMaxImageMegapixels, or: 1), 1)
    }
}
