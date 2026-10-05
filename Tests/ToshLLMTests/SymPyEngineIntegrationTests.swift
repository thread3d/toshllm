// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// Runs the real engine with the math runtime from vendor/, both tool sets on. Needs a tool-calling model:
///   TOSH_SYMPY_E2E_MODEL=~/models/Qwen3-4B-Q4_K_M.gguf ./scripts/test.sh --filter SymPyEngine
final class SymPyEngineIntegrationTests: XCTestCase {
    private static let port = 18_433
    private static var server: Process?
    private static var skipReason: String?

    private static var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    override class func setUp() {
        super.setUp()
        let environment = ProcessInfo.processInfo.environment
        guard let model = environment["TOSH_SYMPY_E2E_MODEL"] else {
            skipReason = "Set TOSH_SYMPY_E2E_MODEL to run the engine integration tests"
            return
        }
        let binary = environment["TOSH_BIN"]
            ?? repository.appendingPathComponent("vendor/llama.cpp/build-static/bin/llama-server").path
        let sympy = SymPyToolsService.serverArguments(
            enabled: true, scientific: true, resources: repository.appendingPathComponent("vendor"))
        guard FileManager.default.isExecutableFile(atPath: binary), !sympy.isEmpty else {
            skipReason = "Build the engine and the math runtime first"
            return
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: binary)
        process.arguments = ["-m", (model as NSString).expandingTildeInPath, "--host", "127.0.0.1",
                             "--port", String(port), "-ngl", "99", "-c", "8192", "--load-mode", "none",
                             "--jinja", "--tools", "all"] + sympy
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch {
            skipReason = "Could not start the engine: \(error.localizedDescription)"
            return
        }
        server = process
        let health = URL(string: "http://127.0.0.1:\(port)/health")!
        for _ in 0..<240 {
            if let data = try? Data(contentsOf: health), String(decoding: data, as: UTF8.self).contains("ok") {
                return
            }
            Thread.sleep(forTimeInterval: 0.5)
        }
        skipReason = "The engine did not become ready"
    }

    override class func tearDown() {
        server?.terminate()
        server?.waitUntilExit()
        server = nil
        super.tearDown()
    }

    override func setUpWithError() throws {
        if let reason = Self.skipReason { throw XCTSkip(reason) }
        UserDefaults.standard.set(true, forKey: SettingsKeys.agentToolsEnabled)
        UserDefaults.standard.set(true, forKey: SettingsKeys.sympyEnabled)
        UserDefaults.standard.set(true, forKey: SettingsKeys.scientificEnabled)
    }

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: SettingsKeys.agentToolsEnabled)
        UserDefaults.standard.removeObject(forKey: SettingsKeys.sympyEnabled)
        UserDefaults.standard.removeObject(forKey: SettingsKeys.scientificEnabled)
        super.tearDown()
    }

    private func run(_ tool: String, _ arguments: [String: Any], cwd: String? = nil) async throws -> ToolExecutionResult {
        try await ChatToolsService.execute(name: tool, arguments: arguments, port: Self.port,
                                           workingDirectory: cwd)
    }

    func testToolsAreDiscoveredThroughTheEngine() async throws {
        let tools = try await ChatToolsService.listEnabled(port: Self.port)
        let names = Set(tools.map(\.name))
        for name in ["sympy_expression", "sympy_solve", "sympy_matrix", "sympy_verify",
                     "scientific_compute", "scientific_linalg", "scientific_optimize",
                     "scientific_signal", "scientific_ode", "scientific_stats",
                     "read_file", "write_file", "edit_file", "grep_search", "file_glob_search",
                     "exec_shell_command"] {
            XCTAssertTrue(names.contains(name), name)
        }
        for tool in tools where SymPyToolsService.isTool(tool.name) || ScientificToolsService.isTool(tool.name) {
            XCTAssertFalse(tool.writesData)
            XCTAssertFalse(tool.usesCwd)
            XCTAssertNotNil(tool.openAIDefinition)
        }
    }

    func testEachSettingHidesItsOwnTools() async throws {
        UserDefaults.standard.set(false, forKey: SettingsKeys.sympyEnabled)
        var names = try await ChatToolsService.listEnabled(port: Self.port).map(\.name)
        XCTAssertFalse(names.contains { SymPyToolsService.isTool($0) })
        XCTAssertTrue(names.contains("read_file"))

        UserDefaults.standard.set(true, forKey: SettingsKeys.sympyEnabled)
        UserDefaults.standard.set(false, forKey: SettingsKeys.scientificEnabled)
        UserDefaults.standard.set(false, forKey: SettingsKeys.agentToolsEnabled)
        names = try await ChatToolsService.listEnabled(port: Self.port).map(\.name)
        XCTAssertEqual(Set(names), ["sympy_expression", "sympy_solve", "sympy_matrix", "sympy_verify"])

        UserDefaults.standard.set(false, forKey: SettingsKeys.sympyEnabled)
        UserDefaults.standard.set(true, forKey: SettingsKeys.scientificEnabled)
        names = try await ChatToolsService.listEnabled(port: Self.port).map(\.name)
        XCTAssertEqual(Set(names), ["scientific_compute", "scientific_linalg", "scientific_optimize",
                                    "scientific_signal", "scientific_ode", "scientific_stats"])
    }

    func testScientificToolsRunThroughTheEngine() async throws {
        let integral = try await run("scientific_compute", ["operation": "integrate", "expression": "sin(x**2)",
                                                            "lower": 0, "upper": 10])
        XCTAssertFalse(integral.isError, integral.content)
        XCTAssertTrue(integral.content.contains(#""value": 0.5836708999"#), integral.content)

        let determinant = try await run("scientific_linalg", ["operation": "determinant",
                                                              "matrix": [[1, 2], [3, 4]]])
        XCTAssertTrue(determinant.content.contains(#""determinant": -2.0"#), determinant.content)

        let minimum = try await run("scientific_optimize", ["operation": "minimize",
                                                            "expression": "(x - 3)**2 + (y + 1)**2",
                                                            "initial_guess": ["x": 0, "y": 0]])
        XCTAssertFalse(minimum.isError, minimum.content)
        XCTAssertTrue(minimum.content.contains(#""success": true"#), minimum.content)

        let spectrum = try await run("scientific_signal", ["operation": "fft", "expression": "sin(2*pi*50*t)",
                                                           "sample_rate": 1000, "duration": 1])
        XCTAssertTrue(spectrum.content.contains("50.0"), spectrum.content)

        let decay = try await run("scientific_ode", ["operation": "solve_ivp", "equations": ["dy/dt = -y"],
                                                     "initial_conditions": ["y": 1], "interval": [0, 1],
                                                     "at": [1]])
        XCTAssertTrue(decay.content.contains("0.36787"), decay.content)

        let summary = try await run("scientific_stats", ["operation": "describe", "values": [1, 2, 3, 4]])
        XCTAssertTrue(summary.content.contains(#""mean": 2.5"#), summary.content)

        // a failed computation is an error with its code, never a number
        let singular = try await run("scientific_linalg", ["operation": "solve", "matrix": [[1, 2], [2, 4]],
                                                           "other": [1, 2]])
        XCTAssertTrue(singular.isError)
        XCTAssertTrue(singular.content.contains("singular_matrix"), singular.content)
        let divergent = try await run("scientific_compute", ["operation": "integrate", "expression": "1/x",
                                                             "lower": 0, "upper": 1])
        XCTAssertTrue(divergent.isError)
        XCTAssertFalse(divergent.content.contains(#""value""#), divergent.content)

        let injected = try await run("scientific_compute", ["operation": "evaluate",
                                                            "expression": "__import__('os').system('id')"])
        XCTAssertTrue(injected.isError)
        XCTAssertTrue(injected.content.contains("invalid_expression"), injected.content)
        let named = try await run("scientific_compute", ["operation": "evaluate",
                                                         "expression": "np.sin(1)"])
        XCTAssertTrue(named.isError, named.content)
    }

    func testToolsRunThroughTheEngine() async throws {
        let factored = try await run("sympy_expression", ["operation": "factor", "expression": "x**2 - 5*x + 6"])
        XCTAssertFalse(factored.isError)
        XCTAssertTrue(factored.content.contains(#""exact": "(x - 3)*(x - 2)""#), factored.content)

        let verified = try await run("sympy_verify", ["operation": "equivalent",
                                                      "left": "(x + 1)**2", "right": "x**2 + 2*x"])
        XCTAssertTrue(verified.content.contains(#""equivalent": false"#), verified.content)

        let solved = try await run("sympy_solve", ["operation": "solve", "equations": ["e**2 - 4 = 0"],
                                                   "variables": ["e"]])
        XCTAssertTrue(solved.content.contains(#""solutions": [{"e": "-2"}, {"e": "2"}]"#), solved.content)

        let determinant = try await run("sympy_matrix", ["operation": "determinant",
                                                         "matrix": [["1", "2"], ["3", "4"]]])
        XCTAssertTrue(determinant.content.contains(#""exact": "-2""#), determinant.content)

        let numeric = try await run("sympy_expression", ["operation": "integrate", "expression": "exp(sin(x))",
                                                         "variable": "x", "lower": "0", "upper": "1"])
        XCTAssertFalse(numeric.isError)
        XCTAssertTrue(numeric.content.contains(#""method": "numerical_integration""#), numeric.content)
        XCTAssertEqual(SymPyToolsService.readable(numeric.content).components(separatedBy: "\n").prefix(2).joined(separator: " "),
                       "Integral(exp(sin(x)), (x, 0, 1)) ≈ 1.63186960841805")

        // no closed form is an answer, not a failed call
        let open = try await run("sympy_expression", ["operation": "integrate", "expression": "sin(sin(x))",
                                                      "variable": "x"])
        XCTAssertFalse(open.isError)
        XCTAssertTrue(open.content.contains(#""code": "no_closed_form""#), open.content)

        let injected = try await run("sympy_expression", ["operation": "simplify",
                                                          "expression": "__import__('os').system('id')"])
        XCTAssertTrue(injected.isError)
        XCTAssertTrue(injected.content.contains("invalid_expression"), injected.content)
    }

    func testACallThatDropsPartOfTheRequestIsNotComputed() async throws {
        let source: [String: Any] = ["request": "Find the limit of (sin(x) - x)/x^3 as x approaches 0.", "context": ""]
        // a request text the model slips into the arguments is dropped; the app's own text is what counts
        let refused = try await MathTranscriptionService.execute(
            name: "sympy_expression",
            arguments: ["operation": "limit", "expression": "sin(x) - x", "variable": "x", "point": "0",
                        "_source": ["request": "Find the limit of sin(x) - x."], "_reviewed": "consistent"],
            source: source, port: Self.port)
        XCTAssertTrue(refused.isError)
        XCTAssertTrue(refused.content.contains("transcription_mismatch"), refused.content)
        let computed = try await MathTranscriptionService.execute(
            name: "sympy_expression",
            arguments: ["operation": "limit", "expression": "(sin(x) - x)/x**3", "variable": "x", "point": "0"],
            source: source, port: Self.port)
        XCTAssertTrue(computed.content.contains(#""exact": "-1/6""#), computed.content)
        XCTAssertTrue(computed.content.contains("interpreted_input"), computed.content)
        // nothing to compare in the request: the model reviews the call before anything runs
        let word = try await MathTranscriptionService.execute(
            name: "sympy_solve", arguments: ["operation": "solve", "equations": ["x + y = 9", "x*y = 20"]],
            source: ["request": "The sum of two numbers is 9 and their product is 20. What are they?", "context": ""],
            port: Self.port)
        XCTAssertTrue(word.content.contains("solutions") || word.content.contains("needs_review"), word.content)
    }

    func testAfterARefusalTheEngineOnlyLetsTheModelCallAToolOrAsk() async throws {
        let tools = try await ChatToolsService.listEnabled(port: Self.port)
            .filter { MathTranscriptionService.isMathTool($0.name) }.compactMap(\.openAIDefinition)
            + [MathTranscriptionService.clarifyTool]
        let refused = #"{"success": false, "operation": "limit", "error": {"code": "transcription_mismatch", "message": "Not computed: 'expression' is `sin(x) - x`, which is only a part of `(sin(x) - x)/x^3` in the request."}}"#
        let messages: [[String: Any]] = [
            ["role": "user", "content": "Find the limit of (sin(x) - x)/x^3 as x approaches 0. /no_think"],
            ["role": "assistant", "content": "", "tool_calls": [["id": "c1", "type": "function", "function": [
                "name": "sympy_expression",
                "arguments": #"{"operation": "limit", "expression": "sin(x) - x", "variable": "x", "point": "0"}"#]]]],
            ["role": "tool", "tool_call_id": "c1", "content": refused],
        ]
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(Self.port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 300
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "messages": messages, "tools": tools, "tool_choice": "required", "temperature": 0, "max_tokens": 512,
        ])
        let (data, _) = try await URLSession.shared.data(for: request)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let message = try XCTUnwrap(((object["choices"] as? [[String: Any]])?.first)?["message"] as? [String: Any])
        let calls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]], "the engine let the model answer in text")
        let name = try XCTUnwrap((calls.first?["function"] as? [String: Any])?["name"] as? String)
        XCTAssertTrue(MathTranscriptionService.isMathTool(name) || name == MathTranscriptionService.clarifyToolName, name)
    }

    func testFileAndShellToolsStillWork() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-tools-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let cwd = directory.path

        let written = try await run("write_file", ["path": "note.txt", "content": "alpha\nbeta\n"], cwd: cwd)
        XCTAssertFalse(written.isError, written.content)
        let read = try await run("read_file", ["path": "note.txt"], cwd: cwd)
        XCTAssertEqual(read.content, "alpha\nbeta\n")
        let edited = try await run("edit_file", ["path": "note.txt",
                                                 "edits": [["old_text": "beta", "new_text": "gamma"]]], cwd: cwd)
        XCTAssertFalse(edited.isError, edited.content)
        let grep = try await run("grep_search", ["path": ".", "pattern": "gamma"], cwd: cwd)
        XCTAssertTrue(grep.content.contains("note.txt:gamma"), grep.content)
        let glob = try await run("file_glob_search", ["path": ".", "include": "*.txt"], cwd: cwd)
        XCTAssertTrue(glob.content.contains("note.txt"), glob.content)
        let shell = try await ChatToolsService.executeStreaming(
            name: "exec_shell_command", arguments: ["command": "cat note.txt"], port: Self.port,
            workingDirectory: cwd) { _ in }
        XCTAssertTrue(shell.content.contains("gamma") && shell.content.contains("[exit code: 0]"), shell.content)
    }

    func testExactRequestStaysSymbolicWithBothToolSets() async throws {
        let tools = try await ChatToolsService.listEnabled(port: Self.port)
            .filter { SymPyToolsService.isTool($0.name) || ScientificToolsService.isTool($0.name) }
            .compactMap(\.openAIDefinition)
        XCTAssertEqual(tools.count, 10)
        let system = ScientificToolsService.system("", sympy: true, scientific: true)
        for (prompt, family) in [("Integrate x^2 from 0 to 3.", "sympy_"),
                                 ("Numerically integrate exp(sin(x)) from 0 to 2.", "scientific_")] {
            let reply = try await complete([["role": "system", "content": system],
                                            ["role": "user", "content": prompt + " Use the tools. /no_think"]],
                                           tools: tools)
            let call = try XCTUnwrap((reply["tool_calls"] as? [[String: Any]])?.first, prompt)
            let name = try XCTUnwrap((call["function"] as? [String: Any])?["name"] as? String)
            XCTAssertTrue(name.hasPrefix(family), "\(prompt) -> \(name)")
        }
    }

    private func complete(_ messages: [[String: Any]], tools: [[String: Any]]) async throws -> [String: Any] {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(Self.port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 300
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "messages": messages, "tools": tools, "temperature": 0, "max_tokens": 1024,
        ])
        let (data, _) = try await URLSession.shared.data(for: request)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        return try XCTUnwrap(choice["message"] as? [String: Any])
    }

    func testModelCallsTheToolAndUsesItsResult() async throws {
        let tools = try await ChatToolsService.listEnabled(port: Self.port)
            .filter { SymPyToolsService.isTool($0.name) }.compactMap(\.openAIDefinition)
        var messages: [[String: Any]] = [[
            "role": "user",
            "content": "Use the tool to factor x^2 - 5x + 6, then tell me the factors. /no_think",
        ]]
        let first = try await complete(messages, tools: tools)
        let call = try XCTUnwrap((first["tool_calls"] as? [[String: Any]])?.first, "the model made no tool call")
        let function = try XCTUnwrap(call["function"] as? [String: Any])
        let name = try XCTUnwrap(function["name"] as? String)
        XCTAssertTrue(SymPyToolsService.isTool(name), name)

        let arguments = try ChatToolsService.parseArguments(try XCTUnwrap(function["arguments"] as? String))
        let result = try await run(name, arguments)
        XCTAssertFalse(result.isError, result.content)
        XCTAssertTrue(result.content.contains("(x - 3)*(x - 2)"), result.content)

        messages.append(first)
        messages.append(["role": "tool", "tool_call_id": call["id"] as? String ?? "", "content": result.content])
        let answer = try await complete(messages, tools: tools)
        let text = try XCTUnwrap(answer["content"] as? String)
        XCTAssertTrue(text.contains("3") && text.contains("2"), text)
        XCTAssertNil(answer["tool_calls"] as? [[String: Any]])
    }
    private static let reportedRequest = #"""
        Calcula exactamente
        \[
        I=\int_{0}^{\infty}\frac{x^3}{e^x-1}\,dx
        \]
        Después:
        1. expresa el resultado en forma exacta;
        2. dame su valor decimal con 10 cifras decimales;
        3. verifica el resultado mediante integración numérica independiente;
        4. indica el error absoluto entre el valor exacto evaluado numéricamente y la integración numérica.
        Usa las herramientas matemáticas/científicas disponibles cuando corresponda. No hagas el cálculo solo de memoria.
        """#

    /// Runs one request through the chat's own agent loop and returns the turn, how many model
    /// passes it took and how long.
    @MainActor
    private func agentTurn(_ text: String, maxTokens: Int = 4096) async throws
        -> (messages: [ChatMessage], passes: Int, seconds: Double, intent: MathIntent?) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tosh-agent-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatStore(storageDirectory: directory)
        var sampling = ChatSamplingSettings()
        sampling.reasoningEffort = "off"
        let started = Date()
        store.send(text: text, port: Self.port, temperature: 0.7, maxTokens: maxTokens, system: "", thinking: false,
                   sampling: sampling)
        // every pass streams into a message of its own, also one that is dropped afterwards
        var passes = Set<UUID>()
        let deadline = Date().addingTimeInterval(900)
        while store.agentFlowActive, store.pendingAgentContinuation == nil, Date() < deadline {
            if store.generating, let id = store.current?.messages.last?.id { passes.insert(id) }
            if store.pendingToolPermission != nil, !store.generating { store.respondToToolPermission(.once) }
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertFalse(store.agentFlowActive && store.pendingAgentContinuation == nil, "the turn did not end")
        XCTAssertNil(store.lastError)
        return (try XCTUnwrap(store.current?.messages), passes.count, Date().timeIntervalSince(started), store.lastMathIntent)
    }

    /// The math tools run without asking by default, which hands the chat's math turns to the engine's agent;
    /// with that off the chat runs them itself. nil leaves the default.
    private func useServerAgent(_ on: Bool?) async throws {
        let names = try await ChatToolsService.listEnabled(port: Self.port).map(\.name).filter(MathTranscriptionService.isMathTool)
        XCTAssertEqual(names.count, 10)
        for name in names { UserDefaults.standard.removeObject(forKey: "toolPermission.always.builtin.\(name)") }
        if let on { UserDefaults.standard.set(on, forKey: SettingsKeys.mathToolsAllowed) }
        else { UserDefaults.standard.removeObject(forKey: SettingsKeys.mathToolsAllowed) }
    }

    private static var serverPath: Bool { ProcessInfo.processInfo.environment["TOSH_AGENT_PATH"] == "server" }

    private func record(_ line: [String: Any], to destination: String? = nil) {
        guard let path = destination ?? ProcessInfo.processInfo.environment["TOSH_AGENT_TRACE"] else { return }
        if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
        guard let file = FileHandle(forWritingAtPath: path),
              let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else { return }
        file.seekToEndOfFile()
        file.write(data + Data("\n".utf8))
        file.closeFile()
    }

    /// A reported request through the chat's own agent loop, with the settings of the report:
    /// math tools only, no reasoning. TOSH_AGENT_RUNS repeats it, since the model samples.
    @MainActor
    func testReportedImproperIntegralThroughTheAgent() async throws {
        UserDefaults.standard.set(false, forKey: SettingsKeys.agentToolsEnabled)
        UserDefaults.standard.set(false, forKey: SettingsKeys.memoryToolsEnabled)
        defer { UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryToolsEnabled) }
        let runs = Int(ProcessInfo.processInfo.environment["TOSH_AGENT_RUNS"] ?? "") ?? 1
        // the same request through the app's own loop and through the engine's agent
        for (server, run) in [false, true].flatMap({ server in (1...max(1, runs)).map { (server, $0) } }) {
            try await useServerAgent(server)
            let (messages, passes, seconds, intent) = try await agentTurn(Self.reportedRequest)
            if server { XCTAssertEqual(intent, .computational, "the agent's intent reaches the app") }
            let ledger = MathLedger.results(messages)
            let last = try XCTUnwrap(messages.last)
            let answer = last.parts.body
            let sources = messages.filter { $0.role == "user" }.map(\.wireContent)
            let ungrounded = MathGrounding.ungrounded(answer, sources: sources, results: ledger)
            record(["path": server ? "server" : "app", "run": run, "passes": passes, "seconds": seconds,
                    "messages": messages.map(Self.trace),
                    "ledger": ledger.map { "\($0.tool) \($0.operation): \($0.result.joined(separator: "; "))" },
                    "answer": answer, "claims": MathGrounding.numbers(in: MathGrounding.withoutListMarkers(answer)).map(\.text)
                        + MathGrounding.constants(in: answer),
                    "ungrounded": ungrounded, "prompt_tokens": last.timings?.promptTokens ?? -1])

            XCTAssertEqual(last.role, "assistant")
            XCTAssertTrue(last.toolCalls?.isEmpty ?? true)
            XCTAssertEqual(ungrounded, [], "run \(run): the answer states values no source or result gives")
            for message in messages where message.role == "assistant" {
                let calls = (message.toolCalls ?? []).filter { MathTranscriptionService.isMathTool($0.name) }
                if !calls.allSatisfy(MathTranscriptionService.succeeded) {
                    XCTAssertNil(message.interim, "run \(run): text before an unsuccessful math call was kept")
                }
            }
            let integrals = ledger.filter { $0.operation == "integrate" && $0.tool == "scientific_compute" }
            for integral in integrals {
                let value = try XCTUnwrap(MathTranscriptionService.reply(integral.reply)?["value"] as? Double)
                XCTAssertEqual(value, Double.pi * Double.pi * Double.pi * Double.pi / 15, accuracy: 1e-9)
            }
            if !ledger.isEmpty {
                XCTAssertNotEqual(answer, MathTranscriptionService.unresolvedMessage(), "run \(run): a validated result was lost")
            }
            if !integrals.isEmpty {
                XCTAssertTrue(answer.contains("6.49") || answer.contains("6,49"), "run \(run): the validated integral is not shown")
            }
        }
        try await useServerAgent(nil)
    }

    /// With nothing set, a math turn of the chat runs in the engine's agent.
    @MainActor
    func testMathTurnsGoToTheServerAgentByDefault() async throws {
        UserDefaults.standard.set(false, forKey: SettingsKeys.agentToolsEnabled)
        UserDefaults.standard.set(false, forKey: SettingsKeys.memoryToolsEnabled)
        defer { UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryToolsEnabled) }
        try await useServerAgent(nil)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tosh-agent-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatStore(storageDirectory: directory)
        var sampling = ChatSamplingSettings()
        sampling.reasoningEffort = "off"
        store.send(text: "Find the determinant of [[2, -1], [4, 3]].", port: Self.port, temperature: 0.7, maxTokens: 1024,
                   system: "", thinking: false, sampling: sampling)
        let deadline = Date().addingTimeInterval(300)
        while store.agentFlowActive, Date() < deadline {
            XCTAssertNil(store.pendingToolPermission, "a math call asked for permission")
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertTrue(store.lastTurnUsedServerAgent)
        XCTAssertEqual(store.lastMathIntent, .computational)
        let messages = try XCTUnwrap(store.current?.messages)
        XCTAssertEqual(MathLedger.results(messages).count, 1)
        XCTAssertTrue(messages.last?.parts.body.contains("10") == true, messages.last?.parts.body ?? "")
    }

    /// Stop in the chat ends the engine agent's turn: no model pass keeps running on the server.
    @MainActor
    func testStopEndsTheServerAgentTurn() async throws {
        UserDefaults.standard.set(false, forKey: SettingsKeys.agentToolsEnabled)
        UserDefaults.standard.set(false, forKey: SettingsKeys.memoryToolsEnabled)
        defer { UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryToolsEnabled) }
        try await useServerAgent(true)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tosh-agent-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatStore(storageDirectory: directory)
        var sampling = ChatSamplingSettings()
        sampling.reasoningEffort = "off"
        // a long pass with nothing to compute, cut in the middle
        store.send(text: "Write a 3000-word essay on the history of calculus, with no formulas.",
                   port: Self.port, temperature: 0.7, maxTokens: 8192, system: "", thinking: false, sampling: sampling)
        try await Task.sleep(for: .seconds(12))
        XCTAssertTrue(store.generating, "the pass ended before the stop")
        store.stop()
        let stopped = Date()
        var busy = true
        while busy, Date().timeIntervalSince(stopped) < 20 {
            try await Task.sleep(for: .milliseconds(500))
            let (data, _) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(Self.port)/slots")!)
            let slots = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
            busy = slots.contains { $0["is_processing"] as? Bool == true }
        }
        record(["stop": "server agent", "stopped_at": ISO8601DateFormatter().string(from: stopped),
                "seconds_to_idle": Date().timeIntervalSince(stopped),
                "intent": store.lastMathIntent?.rawValue ?? "none"])
        XCTAssertFalse(busy, "the engine still generates for a stopped turn")
        XCTAssertFalse(store.generating)
        try await useServerAgent(nil)
    }

    /// Simple math and plain turns, to see what the final check costs when nothing is wrong.
    @MainActor
    func testFinalCheckCostOnSimpleTurns() async throws {
        UserDefaults.standard.set(false, forKey: SettingsKeys.agentToolsEnabled)
        UserDefaults.standard.set(false, forKey: SettingsKeys.memoryToolsEnabled)
        defer { UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryToolsEnabled) }
        let prompts = ["Factoriza x^2 - 5x + 6.", "Integra numéricamente exp(-x^2) de 0 a 1.",
                       "Calcula el determinante de [[2, 1], [1, 3]].", "Escribe una frase sobre el mar."]
        let runs = Int(ProcessInfo.processInfo.environment["TOSH_AGENT_RUNS"] ?? "") ?? 1
        for run in 1...max(1, runs) {
            for prompt in prompts {
                let (messages, passes, seconds, _) = try await agentTurn(prompt)
                let rounds = messages.filter { $0.role == "assistant" }.count
                let answer = messages.last?.parts.body ?? ""
                let ungrounded = MathGrounding.ungrounded(answer, sources: [prompt], results: MathLedger.results(messages))
                record(["cost": prompt, "run": run, "passes": passes, "rounds": rounds, "seconds": seconds,
                        "answer": answer, "ungrounded": ungrounded,
                        "math_calls": MathLedger.calls(messages).count,
                        "prompt_tokens": messages.compactMap { $0.timings?.promptTokens }])
                // a turn that used no math tool is not checked: the model answered on its own
                if !MathLedger.calls(messages).isEmpty { XCTAssertEqual(ungrounded, [], prompt) }
                XCTAssertFalse(answer.isEmpty, prompt)
            }
        }
    }

    /// A frozen suite through the chat's own loop: TOSH_SUITE is the suite, TOSH_SUITE_OUT gets one line per prompt.
    @MainActor
    func testFrozenSuiteThroughTheAgent() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let suitePath = environment["TOSH_SUITE"], let outPath = environment["TOSH_SUITE_OUT"] else {
            throw XCTSkip("Set TOSH_SUITE and TOSH_SUITE_OUT to run a frozen suite")
        }
        UserDefaults.standard.set(false, forKey: SettingsKeys.agentToolsEnabled)
        UserDefaults.standard.set(false, forKey: SettingsKeys.memoryToolsEnabled)
        defer { UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryToolsEnabled) }
        try await useServerAgent(Self.serverPath)
        let suite = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: suitePath))) as? [[String: String]])
        let only = environment["TOSH_SUITE_ONLY"].map { Set($0.split(separator: ",").map(String.init)) }
        let runs = Int(environment["TOSH_AGENT_RUNS"] ?? "") ?? 1
        for run in 1...max(1, runs) {
            for item in suite where only?.contains(item["id"] ?? "") ?? true {
                let prompt = item["prompt"] ?? ""
                let (messages, passes, seconds, intent) = try await agentTurn(prompt, maxTokens: 1024)
                let start = MathLedger.turnStart(messages)
                let calls = messages[min(start, messages.count)...].flatMap { $0.toolCalls ?? [] }
                let results = MathLedger.results(messages, from: start)
                let answer = messages.last.map { $0.role == "assistant" && ($0.toolCalls ?? []).isEmpty ? $0.parts.body : "" } ?? ""
                let base = MathTranscriptionService.unresolvedMessage()
                let outcome = answer.hasPrefix(base) ? (answer == base ? "unresolved" : "clarification_required")
                    : answer.hasPrefix("This is what the tools validated") ? "validated_results_only" : "answered"
                record(["id": item["id"] ?? "", "label": item["label"] ?? "", "run": run, "intent": intent?.rawValue ?? "none",
                        "path": Self.serverPath ? "server" : "app",
                        "outcome": outcome, "passes": passes, "seconds": seconds, "validated": results.map(\.reply),
                        "calls": calls.map { call -> [String: Any] in
                            let reply = call.result.flatMap(MathTranscriptionService.reply) ?? [:]
                            return ["tool": call.name, "arguments": call.arguments,
                                    "status": MathTranscriptionService.succeeded(call) ? "ok"
                                        : MathTranscriptionService.errorCode(reply) ?? call.state.rawValue]
                        },
                        "content": answer,
                        "ungrounded": MathGrounding.ungrounded(answer, sources: [prompt], results: results),
                        "prompt_tokens": messages[min(start, messages.count)...].compactMap { $0.timings?.promptTokens }.reduce(0, +)],
                       to: outPath)
            }
        }
        try await useServerAgent(nil)
    }

    private static func trace(_ message: ChatMessage) -> String {
        var line: [String: Any] = ["role": message.role, "body": message.parts.body]
        if let interim = message.interim { line["interim"] = interim }
        if let id = message.toolCallID { line["tool_call_id"] = id }
        if let calls = message.toolCalls {
            line["calls"] = calls.map { ["id": $0.serverID ?? "", "name": $0.name, "arguments": $0.arguments,
                                         "state": $0.state.rawValue, "result": $0.result ?? ""] }
        }
        let data = (try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys])) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }
}
