// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The extensions found inside the app and in the chosen folder (docs/extensions.md): each with its
/// switch and, once on, the settings it declares. Everything applies at the next server start.
struct ExtensionsSettingsView: View {
    @EnvironmentObject private var loc: Localizer
    @EnvironmentObject private var server: ServerController
    @State private var found: [EngineExtensions.Extension] = []
    @State private var problems: [EngineExtensions.Problem] = []
    @State private var enabled: Set<String> = []
    @State private var values: [String: [String: String]] = [:]
    @State private var pendingRestart = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                folderRow
                if found.isEmpty && problems.isEmpty { emptyState }
                ForEach(found) { item in extensionCard(item) }
                if !problems.isEmpty { problemList }
                restartRow
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onAppear(perform: reload)
    }

    private func text(_ text: EngineExtensions.Text) -> String { text.pick(spanish: loc.isSpanish) }

    private func reload() {
        let result = EngineExtensions.all()
        found = result.extensions
        problems = result.problems
        enabled = Set(found.filter(EngineExtensions.isEnabled).map(\.id))
        values = found.reduce(into: [:]) { all, item in
            all[item.id] = item.fields.reduce(into: [:]) { $0[$1.key] = EngineExtensions.value($1, of: item) }
        }
    }

    // MARK: folder

    private var folderRow: some View {
        let custom = !(UserDefaults.standard.string(forKey: SettingsKeys.extensionsFolder) ?? "").isEmpty
        return SettingsRowGroup {
            SettingsRow(icon: "folder", title: loc.t("Carpeta de extensiones", "Extensions folder"),
                        subtitle: EngineExtensions.chosenFolder.path,
                        help: loc.t("Cada subcarpeta con un extension.json es una extensión. Las que encuentres aquí empiezan apagadas.",
                                    "Every subfolder with an extension.json is one extension. The ones found here start off.")) {
                HStack(spacing: 8) {
                    Button(loc.t("Abrir", "Open")) {
                        let folder = EngineExtensions.chosenFolder
                        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                        NSWorkspace.shared.open(folder)
                    }
                    .glassButton()
                    .help(loc.t("Abre la carpeta en el Finder; si no existe, la crea.", "Opens the folder in Finder, creating it if needed."))
                    Button(loc.t("Cambiar…", "Change…"), action: chooseFolder)
                        .glassButton()
                        .help(loc.t("Elige otra carpeta de extensiones.", "Choose another extensions folder."))
                    if custom {
                        Button(loc.t("Restablecer", "Reset")) {
                            UserDefaults.standard.removeObject(forKey: SettingsKeys.extensionsFolder)
                            reload()
                            pendingRestart = true
                        }
                        .glassButton()
                        .help(loc.t("Vuelve a la carpeta por defecto.", "Goes back to the default folder."))
                    }
                    Button(action: reload) { Image(systemName: "arrow.clockwise") }
                        .glassButton()
                        .help(loc.t("Vuelve a leer la carpeta.", "Reads the folder again."))
                }
            }
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        UserDefaults.standard.set(url.path, forKey: SettingsKeys.extensionsFolder)
        reload()
        pendingRestart = true
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(loc.t("No hay extensiones en esta carpeta.", "There are no extensions in this folder."))
                .font(.callout)
            Text(loc.t("Una extensión es una carpeta con un extension.json que declara servidores MCP y sus ajustes. Cualquier persona o programa puede crearla.",
                       "An extension is a folder with an extension.json that declares MCP servers and their settings. Any person or program can create one."))
                .font(.caption)
                .foregroundStyle(.secondary)
            if let docs = URL(string: AppInfo.repositoryURL + "/blob/main/docs/extensions.md") {
                Link(loc.t("Formato de las extensiones", "Extension format"), destination: docs)
                    .font(.caption)
                    .help(loc.t("Abre la especificación en GitHub.", "Opens the specification on GitHub."))
            }
        }
    }

    private var problemList: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(loc.t("No se pudieron cargar", "Could not be loaded")).font(.headline)
            ForEach(problems) { problem in
                Label("\(problem.folder.lastPathComponent): \(problem.reason)", systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    // MARK: one extension

    private func extensionCard(_ item: EngineExtensions.Extension) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(text(item.manifest.name)).font(.headline)
                if let version = item.manifest.version {
                    Text(version).font(.caption).foregroundStyle(.secondary)
                }
                Text(item.bundled ? loc.t("Incluida en la app", "Included in the app") : loc.t("De la carpeta", "From the folder"))
                    .font(.caption2)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(.quaternary.opacity(0.5), in: Capsule())
                Spacer(minLength: 8)
                SettingsToggle(isOn: Binding(get: { enabled.contains(item.id) }, set: { setEnabled(item, $0) }))
                    .help(loc.t("Activa o apaga esta extensión", "Turns this extension on or off"))
            }
            if let description = item.manifest.description {
                Text(text(description)).font(.caption).foregroundStyle(.secondary)
            }
            Text(programs(item).joined(separator: " · "))
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.tertiary)
                .lineLimit(2)
                .truncationMode(.middle)
            if enabled.contains(item.id) {
                ForEach(item.manifest.settings ?? []) { section in
                    sectionView(section, item)
                }
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 12))
    }

    /// What the extension starts, as the person approves it: commands for local servers, URLs for remote ones.
    private func programs(_ item: EngineExtensions.Extension) -> [String] {
        func describe(_ name: String, _ server: EngineExtensions.Server) -> String {
            if let url = server.url { return "\(name): \(url)" }
            return "\(name): " + ([server.command ?? ""] + (server.args ?? [])).joined(separator: " ")
                .replacingOccurrences(of: "{extension}", with: item.folder.path)
        }
        var lines = (item.manifest.servers ?? [:]).sorted { $0.key < $1.key }.map { describe($0.key, $0.value) }
        if let agent = item.manifest.agent { lines.append(describe(loc.t("agente", "agent"), agent)) }
        return lines
    }

    private func setEnabled(_ item: EngineExtensions.Extension, _ on: Bool) {
        var replaced: [EngineExtensions.Extension] = []
        if on && item.hasAgent {
            replaced = found.filter { $0.id != item.id && $0.hasAgent && enabled.contains($0.id) }
        }
        if on && !item.bundled {
            let alert = NSAlert()
            alert.messageText = loc.t("¿Activar %@?", "Turn on %@?", text(item.manifest.name))
            var detail = loc.t("Ejecutará estos programas con tus permisos:", "It will run these programs with your permissions:")
                + "\n\n" + programs(item).map { "• " + $0 }.joined(separator: "\n")
            if !replaced.isEmpty {
                detail += "\n\n" + loc.t("Se apagará %@, porque solo una extensión puede traer su agente.",
                                         "%@ will be turned off, since only one extension can bring its agent.",
                                         replaced.map { text($0.manifest.name) }.joined(separator: ", "))
            }
            alert.informativeText = detail
            alert.addButton(withTitle: loc.t("Activar", "Turn on"))
            alert.addButton(withTitle: loc.t("Cancelar", "Cancel"))
            guard alert.runModal() == .alertFirstButtonReturn else { return }
        }
        for other in replaced {
            EngineExtensions.setEnabled(other, false)
            enabled.remove(other.id)
        }
        EngineExtensions.setEnabled(item, on)
        if on { enabled.insert(item.id) } else { enabled.remove(item.id) }
        pendingRestart = true
    }

    private func sectionView(_ section: EngineExtensions.Section, _ item: EngineExtensions.Extension) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(text(section.title)).font(.subheadline.weight(.semibold))
            if let detail = section.detail {
                Text(text(detail)).font(.caption).foregroundStyle(.secondary)
            }
            if let notice = section.notice,
               notice.whenEmpty.allSatisfy({ (values[item.id]?[$0] ?? "").isEmpty }) {
                Label(text(notice.text), systemImage: "testtube.2")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            SettingsRowGroup {
                ForEach(section.fields) { field in
                    fieldRow(field, item)
                }
            }
        }
    }

    // MARK: fields

    private func binding(_ field: EngineExtensions.Field, _ item: EngineExtensions.Extension) -> Binding<String> {
        Binding(get: { values[item.id]?[field.key] ?? "" }, set: { store($0, field, item) })
    }

    private func ticked(_ field: EngineExtensions.Field, _ item: EngineExtensions.Extension) -> Set<String> {
        Set((values[item.id]?[field.key] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
    }

    private func store(_ value: String, _ field: EngineExtensions.Field, _ item: EngineExtensions.Extension) {
        guard values[item.id]?[field.key] != value else { return }
        values[item.id, default: [:]][field.key] = value
        EngineExtensions.setValue(value, for: field, of: item)
        pendingRestart = true
    }

    @ViewBuilder
    private func fieldRow(_ field: EngineExtensions.Field, _ item: EngineExtensions.Extension) -> some View {
        let title = text(field.title)
        let help = text(field.help)
        switch field.kind {
        case .toggle:
            SettingsRow(icon: "power", title: title, help: help) {
                SettingsToggle(isOn: Binding(get: { values[item.id]?[field.key] == "1" },
                                             set: { store($0 ? "1" : "0", field, item) }))
            }
        case .text:
            SettingsRow(icon: "textformat", title: title, help: help) {
                DeferredSettingsTextField("", text: binding(field, item), prompt: field.placeholder, width: 300)
            }
        case .secret:
            SettingsRow(icon: "key", title: title, help: help) {
                SecretSettingsField(stored: !(values[item.id]?[field.key] ?? "").isEmpty) { store($0, field, item) }
            }
        case .choice:
            SettingsRow(icon: "list.bullet", title: title, help: help) {
                Picker("", selection: binding(field, item)) {
                    ForEach(field.options ?? [], id: \.value) { option in
                        Text(text(option.title)).tag(option.value)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: 200)
            }
        case .choices:
            VStack(alignment: .leading, spacing: 6) {
                SettingsRow(icon: "checklist", title: title, help: help) { EmptyView() }
                LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)],
                          alignment: .leading, spacing: 6) {
                    ForEach(field.options ?? [], id: \.value) { option in
                        Toggle(text(option.title), isOn: Binding(
                            get: { ticked(field, item).contains(option.value) },
                            set: { on in
                                var selected = ticked(field, item)
                                if on { selected.insert(option.value) } else { selected.remove(option.value) }
                                store((field.options ?? []).map(\.value).filter(selected.contains).joined(separator: ","), field, item)
                            }))
                        .toggleStyle(.checkbox)
                        .help(text(option.title))
                    }
                }
                .padding(.horizontal, 14)
                .padding(.bottom, 10)
            }
        case .json:
            VStack(alignment: .leading, spacing: 6) {
                SettingsRow(icon: "curlybraces", title: title, help: help) { EmptyView() }
                JSONSettingsEditor(text: binding(field, item), placeholder: field.placeholder ?? "")
                    .padding(.horizontal, 14)
                    .padding(.bottom, 10)
            }
        case .paths:
            VStack(alignment: .leading, spacing: 6) {
                SettingsRow(icon: "folder", title: title, help: help) {
                    Button(loc.t("Añadir…", "Add…")) { addPath(field, item) }
                        .glassButton()
                        .help(takesAnyFolder(field) && (field.fileTypes ?? []).isEmpty
                              ? loc.t("Elige una carpeta de datos.", "Pick a data folder.")
                              : loc.t("Elige un archivo o una carpeta de datos.", "Pick a data file or folder."))
                }
                ForEach(paths(field, item)) { entry in
                    HStack(spacing: 8) {
                        Text(entry.name).font(.system(.callout, design: .monospaced))
                        Text(entry.path).font(.caption).foregroundStyle(.secondary)
                            .lineLimit(1).truncationMode(.middle)
                        Spacer(minLength: 8)
                        Button {
                            setPaths(paths(field, item).filter { $0.name != entry.name }, field, item)
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.plain)
                        .help(loc.t("Quitar", "Remove"))
                    }
                    .padding(.horizontal, 14)
                }
                .padding(.bottom, 4)
            }
        }
    }

    private struct PathEntry: Identifiable {
        let name: String
        let path: String
        var id: String { name }
    }

    private func paths(_ field: EngineExtensions.Field, _ item: EngineExtensions.Extension) -> [PathEntry] {
        guard let data = (values[item.id]?[field.key] ?? "").data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: String] else { return [] }
        return object.map { PathEntry(name: $0.key, path: $0.value) }.sorted { $0.name < $1.name }
    }

    private func setPaths(_ entries: [PathEntry], _ field: EngineExtensions.Field, _ item: EngineExtensions.Extension) {
        guard !entries.isEmpty else { store("", field, item); return }
        let object = Dictionary(entries.map { ($0.name, $0.path) }, uniquingKeysWith: { _, last in last })
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else { return }
        store(String(decoding: data, as: UTF8.self), field, item)
    }

    /// A setting whose `folder_types` holds "*" takes any folder, for extensions that read everything in one.
    private func takesAnyFolder(_ field: EngineExtensions.Field) -> Bool {
        (field.folderTypes ?? []).contains("*")
    }

    private func addPath(_ field: EngineExtensions.Field, _ item: EngineExtensions.Extension) {
        let panel = NSOpenPanel()
        let folders = Set((field.folderTypes ?? []).map { $0.lowercased() })
        let files = (field.fileTypes ?? []).map { $0.lowercased() }
        panel.canChooseFiles = !(takesAnyFolder(field) && files.isEmpty)
        panel.canChooseDirectories = !folders.isEmpty
        panel.allowsMultipleSelection = false
        let types = files.compactMap { UTType(filenameExtension: $0) }
        if !types.isEmpty && folders.isEmpty { panel.allowedContentTypes = types }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        var isFolder: ObjCBool = false
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isFolder)
        let ext = url.pathExtension.lowercased()
        let allowed = isFolder.boolValue ? takesAnyFolder(field) || folders.contains(ext) : files.contains(ext)
        guard allowed else {
            let alert = NSAlert()
            alert.messageText = loc.t("Ese tipo de dato no se puede leer aquí", "That kind of data cannot be read here")
            alert.informativeText = (files.map { "." + $0 } + folders.map {
                $0 == "*" ? loc.t("cualquier carpeta", "any folder") : "." + $0
            }).joined(separator: ", ")
            alert.runModal()
            return
        }
        var current = paths(field, item)
        current.append(PathEntry(name: uniqueName(url.deletingPathExtension().lastPathComponent, taken: current.map(\.name)),
                                 path: url.path))
        setPaths(current, field, item)
    }

    /// Names are how the model refers to a source, so they stay short, lowercase and unique.
    private func uniqueName(_ stem: String, taken: [String]) -> String {
        let base = String(stem.lowercased().map { $0.isLetter || $0.isNumber ? $0 : "_" })
            .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        let root = base.isEmpty ? "source" : base
        var name = root
        var n = 2
        while taken.contains(name) { name = "\(root)_\(n)"; n += 1 }
        return name
    }

    // MARK: restart

    private var serverRunning: Bool {
        if case .stopped = server.state { return false }
        if case .failed = server.state { return false }
        return true
    }

    private var restartRow: some View {
        HStack(spacing: 10) {
            Label(loc.t("Los cambios se aplican al reiniciar el servidor.", "Changes apply when the server restarts."),
                  systemImage: "arrow.clockwise")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            if pendingRestart && serverRunning {
                Button(loc.t("Reiniciar servidor", "Restart server")) {
                    server.restart(.fromDefaults())
                    pendingRestart = false
                }
                .glassButton(prominent: true)
                .help(loc.t("Reinicia el servidor con estos ajustes.", "Restarts the server with these settings."))
            }
        }
    }
}

/// A secret is written once and never shown again; the field only says whether one is stored.
private struct SecretSettingsField: View {
    let stored: Bool
    let save: (String) -> Void
    @EnvironmentObject private var loc: Localizer
    @State private var draft = ""

    var body: some View {
        HStack(spacing: 6) {
            SecureField(stored ? loc.t("Guardado en el Llavero", "Stored in the Keychain") : "", text: $draft)
                .textFieldStyle(.roundedBorder)
                .frame(width: 220)
                .onSubmit(commit)
                .help(loc.t("Se guarda en el Llavero de macOS y solo llega al motor al arrancarlo.",
                            "Stored in the macOS Keychain and handed to the engine only when it starts."))
            Button(loc.t("Guardar", "Save"), action: commit)
                .glassButton()
                .disabled(draft.isEmpty)
                .help(loc.t("Guarda el valor en el Llavero.", "Saves the value in the Keychain."))
            if stored {
                Button(loc.t("Borrar", "Clear")) { save("") }
                    .glassButton()
                    .help(loc.t("Borra el valor guardado.", "Removes the stored value."))
            }
        }
    }

    private func commit() {
        guard !draft.isEmpty else { return }
        save(draft)
        draft = ""
    }
}

/// Multi-line JSON that is stored only once it parses as an object, or when cleared.
private struct JSONSettingsEditor: View {
    @Binding var text: String
    let placeholder: String
    @EnvironmentObject private var loc: Localizer
    @State private var draft = ""
    @State private var invalid = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            TextEditor(text: $draft)
                .font(.system(.caption, design: .monospaced))
                .frame(minHeight: 70, maxHeight: 140)
                .scrollContentBackground(.hidden)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
                .overlay(alignment: .topLeading) {
                    if draft.isEmpty {
                        Text(placeholder)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.tertiary)
                            .padding(10)
                            .allowsHitTesting(false)
                    }
                }
                .onChange(of: draft) { _, value in commit(value) }
            if invalid {
                Text(loc.t("No es un objeto JSON válido; se guarda en cuanto lo sea.",
                           "Not a valid JSON object; it is saved as soon as it is one."))
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
        .onAppear { draft = text }
    }

    private func commit(_ value: String) {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            invalid = false
            text = ""
            return
        }
        let parsed = trimmed.data(using: .utf8).flatMap { try? JSONSerialization.jsonObject(with: $0) }
        invalid = !(parsed is [String: Any])
        if !invalid { text = trimmed }
    }
}
