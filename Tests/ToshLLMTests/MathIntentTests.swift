// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class MathIntentTests: XCTestCase {
    private func decided(_ text: String) -> MathIntent? { MathIntent.classify(text).intent }

    func testTheExamplesOfTheBrief() {
        for text in ["Rewrite this paragraph.", "Summarize this message.", "Translate to Spanish: good morning."] {
            XCTAssertEqual(decided(text), .noMath, text)
        }
        for text in ["What is an eigenvalue?", "Explain Fourier transforms.", "What is an integral?", "¿Qué es una derivada?",
                     "Explica qué es una ecuación diferencial ordinaria.", "What does a determinant represent?"] {
            XCTAssertEqual(decided(text), .conceptual, text)
        }
        for text in ["Solve x^2 - 5x + 6 = 0.", "Find the determinant of [[1, 2], [3, 4]].", "Calcula ∫₀^π sin(x) dx.",
                     #"Calcula exactamente \[ \int_{0}^{\infty}\frac{x^3}{e^x-1}\,dx \] con 10 cifras decimales."#] {
            XCTAssertEqual(decided(text), .computational, text)
        }
        XCTAssertEqual(decided("Calcula la integral."), .ambiguous)
    }

    func testOnlyAModelDecidesWhatTheTextCannotTell() {
        // a calculation verb with nothing named, or an object with no verb
        XCTAssertNil(decided("Simplifica la expresión."))
        XCTAssertNil(decided("What's the standard deviation of the sample?"))
        // numbers that are not calculations
        XCTAssertEqual(decided("Write an email saying the rent of $1,200 arrives on the 5th."), .noMath)
        XCTAssertEqual(decided("Fix this function: def area(r): return 3.14 * r * r"), .noMath)
    }

    func testOnlyACalculationOrAnAmbiguousRequestNeedsTheTools() {
        XCTAssertTrue(MathIntent.computational.requiresTools)
        XCTAssertTrue(MathIntent.ambiguous.requiresTools)
        XCTAssertFalse(MathIntent.conceptual.requiresTools)
        XCTAssertFalse(MathIntent.noMath.requiresTools)
    }
}
