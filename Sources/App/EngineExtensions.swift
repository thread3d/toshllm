// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import CryptoKit
import Foundation

/// Extensions in the format of docs/extensions.md: folders with an extension.json that add MCP servers,
/// an agent and a settings page to the engine. They come from inside the app (TOSH_APP_EXTENSIONS in
/// make-app.sh) and from a folder the person picks; the ones from the folder stay off until approved.
enum EngineExtensions {
    struct Text: Decodable, Sendable {
        let es: String
        let en: String
        func pick(spanish: Bool) -> String { spanish ? es : en }
    }

    struct Server: Decodable, Sendable {
        /// A local server: the executable and its arguments.
        var command: String?
        var args: [String]?
        /// A remote one, reached through the bundled relay: its URL, "http" or "sse" to skip the
        /// detection, and headers filled from settings (header name to setting key).
        var url: String?
        var transport: String?
        var headerEnv: [String: String]?
        var env: [String: String]?
        var timeoutMs: Int?
        /// The setting that switches the server on; without one it always runs.
        var enabledBy: String?
        /// Its tools are listed only to the agent, never to the chat's own tool loop.
        var agentOnly: Bool?
    }

    struct Option: Decodable, Sendable {
        let value: String
        let title: Text
    }

    struct Field: Decodable, Sendable, Identifiable {
        enum Kind: String, Decodable, Sendable { case toggle, text, secret, choice, choices, json, paths }
        let key: String
        let kind: Kind
        let title: Text
        let help: Text
        var defaultValue: String?
        var placeholder: String?
        var options: [Option]?
        /// For `paths`: file extensions to pick, and folder extensions (such as gdb) picked as a whole.
        var fileTypes: [String]?
        var folderTypes: [String]?
        var id: String { key }

        enum CodingKeys: String, CodingKey {
            case key, kind, title, help, placeholder, options, fileTypes, folderTypes
            case defaultValue = "default"
        }
    }

    struct Notice: Decodable, Sendable {
        /// Shown while every one of these settings is empty.
        let whenEmpty: [String]
        let text: Text
    }

    struct Section: Decodable, Sendable, Identifiable {
        let id: String
        let title: Text
        var detail: Text?
        var notice: Notice?
        let fields: [Field]
    }

    struct Manifest: Decodable, Sendable {
        let format: Int
        let id: String
        var version: String?
        let name: Text
        var description: Text?
        var author: String?
        var servers: [String: Server]?
        /// Replaces the engine's agent, so every agent turn goes through it.
        var agent: Server?
        /// Starts the app's math servers with it even when they are off, for an agent that relies on them.
        var mathTools: Bool?
        /// The agent takes the chat's own tools with every turn and hands back the calls to them, so the
        /// chat runs those and the next round goes to the agent again.
        var clientTools: Bool?
        var settings: [Section]?
    }

    struct Extension: Sendable, Identifiable {
        let folder: URL
        let manifest: Manifest
        /// Inside the app, so on unless switched off; the rest need an approval of this exact file.
        let bundled: Bool
        let digest: String
        var id: String { manifest.id }
        var fields: [Field] { manifest.settings?.flatMap(\.fields) ?? [] }
        var hasAgent: Bool { manifest.agent != nil }
        var agentTakesClientTools: Bool { hasAgent && manifest.clientTools == true }
    }

    struct Problem: Error, Sendable, Identifiable {
        let folder: URL
        let reason: String
        var id: String { folder.path }
    }

    static let reservedServers: Set = ["sympy", "scientific", "agent"]

    // MARK: discovery

    static var bundledFolder: URL? { Bundle.main.resourceURL?.appendingPathComponent("extensions") }
    static var defaultFolder: URL { AppSupport.directory.appendingPathComponent("Extensions", isDirectory: true) }

    static var chosenFolder: URL {
        let custom = UserDefaults.standard.string(forKey: SettingsKeys.extensionsFolder) ?? ""
        return custom.isEmpty ? defaultFolder : URL(fileURLWithPath: custom, isDirectory: true)
    }

    /// Every extension found, the app's own first; an id already taken is reported, not loaded.
    static func all(bundled: URL? = bundledFolder, chosen: URL? = chosenFolder) -> (extensions: [Extension], problems: [Problem]) {
        var extensions: [Extension] = []
        var problems: [Problem] = []
        for (root, inside) in [(bundled, true), (chosen, false)] {
            guard let root else { continue }
            let found = discover(in: root, bundled: inside)
            problems += found.problems
            for item in found.extensions {
                if extensions.contains(where: { $0.id == item.id }) {
                    problems.append(Problem(folder: item.folder, reason: "the id \(item.id) is already in use"))
                } else {
                    extensions.append(item)
                }
            }
        }
        return (extensions, problems)
    }

    /// A folder with an extension.json is one extension; otherwise each subfolder that has one is.
    static func discover(in root: URL, bundled: Bool) -> (extensions: [Extension], problems: [Problem]) {
        let fm = FileManager.default
        var folders: [URL]
        if fm.fileExists(atPath: root.appendingPathComponent("extension.json").path) {
            folders = [root]
        } else {
            folders = ((try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey],
                                                    options: [.skipsHiddenFiles])) ?? [])
                .filter { fm.fileExists(atPath: $0.appendingPathComponent("extension.json").path) }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        var extensions: [Extension] = []
        var problems: [Problem] = []
        for folder in folders {
            switch load(folder, bundled: bundled) {
            case .success(let item): extensions.append(item)
            case .failure(let problem): problems.append(problem)
            }
        }
        return (extensions, problems)
    }

    static func load(_ folder: URL, bundled: Bool) -> Result<Extension, Problem> {
        let fail = { (reason: String) in Result<Extension, Problem>.failure(Problem(folder: folder, reason: reason)) }
        guard let data = try? Data(contentsOf: folder.appendingPathComponent("extension.json")) else {
            return fail("extension.json cannot be read")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let manifest: Manifest
        do {
            manifest = try decoder.decode(Manifest.self, from: data)
        } catch {
            return fail("extension.json is not valid: \(error.localizedDescription)")
        }
        guard manifest.format == 1 else { return fail("format \(manifest.format) is not supported") }
        guard manifest.id.range(of: "^[a-z0-9-]+$", options: .regularExpression) != nil else {
            return fail("the id may only hold lowercase letters, digits and -")
        }
        for (name, server) in manifest.servers ?? [:] {
            guard name.range(of: "^[a-z0-9_-]+$", options: .regularExpression) != nil, !reservedServers.contains(name) else {
                return fail("the server name \(name) is not allowed")
            }
            guard server.url != nil || server.command != nil else { return fail("the server \(name) has no command or url") }
        }
        if let agent = manifest.agent, agent.command == nil { return fail("the agent must be a local server") }
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return .success(Extension(folder: folder, manifest: manifest, bundled: bundled, digest: digest))
    }

    // MARK: approval

    static func isEnabled(_ item: Extension) -> Bool {
        if item.bundled { return !UserDefaults.standard.bool(forKey: SettingsKeys.extensionOff(item.id)) }
        return UserDefaults.standard.string(forKey: SettingsKeys.extensionApproval(item.id)) == item.digest
    }

    static func setEnabled(_ item: Extension, _ on: Bool) {
        if item.bundled {
            UserDefaults.standard.set(!on, forKey: SettingsKeys.extensionOff(item.id))
        } else if on {
            UserDefaults.standard.set(item.digest, forKey: SettingsKeys.extensionApproval(item.id))
        } else {
            UserDefaults.standard.removeObject(forKey: SettingsKeys.extensionApproval(item.id))
        }
    }

    /// The extensions that are on. Only one agent can replace the engine's, so a second one that
    /// declares an agent is left out (the settings page switches the first off before that happens).
    static func active(_ extensions: [Extension] = all().extensions) -> [Extension] {
        var withAgent = false
        return extensions.filter { item in
            guard isEnabled(item) else { return false }
            if item.hasAgent {
                if withAgent { return false }
                withAgent = true
            }
            return true
        }
    }

    /// Tool name prefixes of the servers the chat's own tool loop may call: those of active extensions
    /// that are not kept for the agent.
    static func chatToolPrefixes(_ extensions: [Extension] = active()) -> [String] {
        extensions.flatMap { item in
            (item.manifest.servers ?? [:]).filter { $0.value.agentOnly != true }.map { $0.key + "_" }
        }
    }

    // MARK: values

    /// What a setting holds, or its declared default. Secrets live in the Keychain, the rest in defaults.
    static func value(_ field: Field, of item: Extension) -> String {
        if field.kind == .secret { return Keychain.get(account(field.key, item)) ?? "" }
        return UserDefaults.standard.string(forKey: SettingsKeys.extensionValue(item.id, field.key))
            ?? field.defaultValue ?? ""
    }

    static func setValue(_ value: String, for field: Field, of item: Extension) {
        if field.kind == .secret {
            if value.isEmpty { Keychain.delete(account(field.key, item)) } else { Keychain.set(value, account: account(field.key, item)) }
        } else {
            UserDefaults.standard.set(value, forKey: SettingsKeys.extensionValue(item.id, field.key))
        }
    }

    private static func account(_ key: String, _ item: Extension) -> String { "extension.\(item.id).\(key)" }

    /// The ticked options of a `choices` field, in the manifest's order.
    static func chosen(_ field: Field, of item: Extension) -> [String] {
        let ticked = Set(value(field, of: item).split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
        return (field.options ?? []).map(\.value).filter(ticked.contains)
    }

    static func isOn(_ key: String, of item: Extension) -> Bool {
        guard let field = item.fields.first(where: { $0.key == key }) else { return false }
        return value(field, of: item) == "1"
    }

    /// Every setting that holds something, for the engine's environment: the servers it starts inherit
    /// it, so a secret never shows on a command line. A switched-off toggle is left out.
    static func environment(_ extensions: [Extension] = active()) -> [String: String] {
        var env: [String: String] = [:]
        for item in extensions {
            for field in item.fields {
                let value = value(field, of: item)
                if field.kind == .toggle {
                    if value == "1" { env[field.key] = "1" }
                } else if field.kind == .choices {
                    // none ticked is a choice too, not the default
                    env[field.key] = chosen(field, of: item).joined(separator: ",")
                } else if !value.isEmpty {
                    env[field.key] = value
                }
            }
        }
        return env
    }

    // MARK: engine

    /// The servers whose switch is on and the agent, with {python}, {runtime}, {extension} and
    /// {trust_key} filled in, in the shape --mcp-servers-json takes.
    static func engineEntries(_ extensions: [Extension], runtime: URL, trustKey: String)
        -> (servers: [String: Any], agent: [String: Any]?, agentOnly: [String], mathTools: Bool, clientTools: Bool) {
        let python = runtime.appendingPathComponent("python/bin/python3").path
        var servers: [String: Any] = [:]
        var agent: [String: Any]?
        var agentOnly: [String] = []
        for item in extensions {
            let fill: (String) -> String = {
                $0.replacingOccurrences(of: "{python}", with: python)
                    .replacingOccurrences(of: "{runtime}", with: runtime.path)
                    .replacingOccurrences(of: "{extension}", with: item.folder.path)
                    .replacingOccurrences(of: "{trust_key}", with: trustKey)
            }
            func entry(_ server: Server) -> [String: Any]? {
                var entry: [String: Any]
                if let url = server.url {
                    let headers = (server.headerEnv ?? [:]).sorted { $0.key < $1.key }
                        .flatMap { ["--header-env", "\($0.key)=\($0.value)"] }
                    entry = SymPyToolsService.relayEntry(runtime: runtime, url: fill(url), transport: server.transport,
                                                         timeoutSeconds: (server.timeoutMs ?? 60_000) / 1000,
                                                         headerArguments: headers)
                } else if let command = server.command {
                    entry = ["command": fill(command), "args": (server.args ?? []).map(fill),
                             "timeout_ms": server.timeoutMs ?? 60_000]
                } else {
                    return nil
                }
                if let env = server.env { entry["env"] = env.mapValues(fill) }
                return entry
            }
            for (name, server) in item.manifest.servers ?? [:] where servers[name] == nil {
                if let key = server.enabledBy, !isOn(key, of: item) { continue }
                guard let built = entry(server) else { continue }
                servers[name] = built
                if server.agentOnly == true { agentOnly.append(name) }
            }
            if agent == nil, let declared = item.manifest.agent { agent = entry(declared) }
        }
        return (servers, agent, agentOnly.sorted(), extensions.contains { $0.manifest.mathTools == true },
                extensions.contains(where: \.agentTakesClientTools))
    }
}
