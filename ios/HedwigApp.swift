import SwiftUI
import CoreML

/// How to run this on a phone, with a free Apple ID. Do these once.
///
/// 1. Install Xcode from the App Store (it is free, and large).
/// 2. Open Xcode → File → New → Project → iOS → App. Name it Hedwig.
///    Interface: SwiftUI. Language: Swift. Save it anywhere.
/// 3. On the Mac, run:  .venv/bin/python export_iphone.py
///    That writes a folder called iphone/ next to the Python scripts.
/// 4. Drag these three things from iphone/ into the Xcode project navigator.
///    Tick "Copy items if needed":
///      - Hedwig.mlpackage   (the brain)
///      - tokenizer.json     (the word-piece dictionary)
///      - HedwigTokenizer.swift
/// 5. Replace the contents of the project's ContentView.swift with this file.
/// 6. Plug in the iPhone, or pick a simulator (any iPhone) at the top of Xcode.
/// 7. Press the Play triangle. The first time, the phone asks you to trust
///    your Apple ID: Settings → General → VPN & Device Management → Trust.
/// 8. Type a prompt and tap Write Story. The phone runs the brain itself.
///    No internet is used after the app is installed.
///
/// The desk holds 64 word-pieces. A short prompt plus a short story fits.
/// Empty seats on the right are zeros and do not change the next word.

struct ContentView: View {
    @State private var prompt: String = "Once upon a time, a little owl "
    @State private var story: String = "Tap Write Story. The first launch loads the brain, so give it a second."
    @State private var busy: Bool = false

    private let seq = 64
    private let vocab = 4096
    private let temperature: Float = 0.8
    private let topK = 40

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                Text("Hedwig · on this phone")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                TextField("Ask for a story...", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)
                Button(action: write) {
                    Text(busy ? "Writing..." : "Write Story")
                        .fontWeight(.bold)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .disabled(busy)
                ScrollView {
                    Text(story)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .padding()
            .navigationTitle("Hedwig")
        }
    }

    func write() {
        busy = true
        story = ""
        DispatchQueue.global(qos: .userInitiated).async {
            let text = generate(prompt)
            DispatchQueue.main.async {
                story = text
                busy = false
            }
        }
    }

    func generate(_ prompt: String) -> String {
        guard let tokURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json"),
              let modelURL = Bundle.main.url(forResource: "Hedwig", withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: "Hedwig", withExtension: "mlpackage") else {
            return "The brain files are missing. Drag Hedwig.mlpackage and tokenizer.json into Xcode."
        }
        do {
            let tok = try HedwigTokenizer(jsonURL: tokURL)
            let model = try MLModel(contentsOf: modelURL)
            var ids = tok.encode(prompt)
            let start = ids.count
            if ids.isEmpty { return "Type a few words first." }
            for _ in 0..<40 {
                if ids.count >= seq { break }
                let next = try pickNext(model: model, ids: ids)
                ids.append(next)
                let soFar = tok.decode(Array(ids[start...]))
                if soFar.contains("\n\n") {
                    return soFar.components(separatedBy: "\n\n")[0]
                }
            }
            return tok.decode(Array(ids[start...]))
        } catch {
            return "Could not run the brain: \(error.localizedDescription)"
        }
    }

    /// Pad to 64 seats, ask the brain, read the scores at the last real word.
    func pickNext(model: MLModel, ids: [Int]) throws -> Int {
        let n = ids.count
        let arr = try MLMultiArray(shape: [1, seq as NSNumber], dataType: .int32)
        for i in 0..<seq {
            arr[i] = NSNumber(value: i < n ? ids[i] : 0)
        }
        let input = try MLDictionaryFeatureProvider(dictionary: ["ids": arr])
        let out = try model.prediction(from: input)
        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            return 0
        }
        return sample(logits: logits, at: n - 1)
    }

    func sample(logits: MLMultiArray, at position: Int) -> Int {
        var best: [(Int, Float)] = []
        for token in 0..<vocab {
            let score = logits[[0, position, token] as [NSNumber]].floatValue
            best.append((token, score))
        }
        best.sort { $0.1 > $1.1 }
        let kept = Array(best.prefix(topK))
        let maxScore = kept[0].1
        let weights = kept.map { exp((($0.1 - maxScore) / temperature)) }
        let total = weights.reduce(0, +)
        var dart = Float.random(in: 0..<total)
        for (item, weight) in zip(kept, weights) {
            dart -= weight
            if dart <= 0 { return item.0 }
        }
        return kept[0].0
    }
}
