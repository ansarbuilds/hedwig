import Foundation

/// Mac-only check: Swift's dictionary must emit the same piece numbers as Python.
///   swiftc ios/HedwigTokenizer.swift ios/bpe_mac_test.swift -o /tmp/bpe_test
///   /tmp/bpe_test iphone/tokenizer.json iphone/bpe_check.json
@main
struct BPETest {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count == 3 else {
            fputs("usage: bpe_test tokenizer.json bpe_check.json\n", stderr)
            exit(1)
        }
        let tok = try HedwigTokenizer(jsonURL: URL(fileURLWithPath: args[1]))
        let data = try Data(contentsOf: URL(fileURLWithPath: args[2]))
        let rows = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
        var failed = 0
        for row in rows {
            let text = row["text"] as! String
            let want = row["ids"] as! [Int]
            let got = tok.encode(text)
            let back = tok.decode(got)
            if got != want || back != text {
                failed += 1
                print("MISMATCH for \(text.debugDescription)")
                print("  python \(want)")
                print("  swift  \(got)")
            } else {
                print("OK \(text.debugDescription) -> \(got.count) pieces")
            }
        }
        exit(failed == 0 ? 0 : 1)
    }
}
