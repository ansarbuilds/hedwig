import Foundation

/// The same word-piece dictionary as tokenizer.py.
/// A piece is often one word, or a chunk of a word. Numbers 0...255 are single bytes.
/// Later numbers are "glue these two pieces together" rules learned from the stories.
struct HedwigTokenizer {
    let merges: [Pair: Int]
    let pieces: [[UInt8]]

    struct Pair: Hashable {
        let a: Int
        let b: Int
    }

    init(jsonURL: URL) throws {
        let data = try Data(contentsOf: jsonURL)
        let rows = try JSONSerialization.jsonObject(with: data) as! [[Int]]
        var rank: [Pair: Int] = [:]
        var vocab = Array(repeating: [UInt8](), count: 256 + rows.count)
        for i in 0..<256 { vocab[i] = [UInt8(i)] }
        for row in rows {
            let a = row[0], b = row[1], new = row[2]
            rank[Pair(a: a, b: b)] = new
            vocab[new] = vocab[a] + vocab[b]
        }
        merges = rank
        pieces = vocab
    }

    /// Text -> piece numbers. Same rules as tokenizer.encode in Python.
    func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        for chunk in chunks(text) {
            out.append(contentsOf: encodeChunk(Array(chunk.utf8)))
        }
        return out
    }

    /// Piece numbers -> text. Glue the raw bytes first, then read them as UTF-8,
    /// because one accented letter can be split across two pieces.
    func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids where id >= 0 && id < pieces.count {
            bytes.append(contentsOf: pieces[id])
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func encodeChunk(_ bytes: [UInt8]) -> [Int] {
        var ids = bytes.map { Int($0) }
        while ids.count >= 2 {
            var bestAt: Int?
            var bestRank = Int.max
            for i in 0..<(ids.count - 1) {
                if let rank = merges[Pair(a: ids[i], b: ids[i + 1])], rank < bestRank {
                    bestRank = rank
                    bestAt = i
                }
            }
            guard let at = bestAt else { break }
            ids[at] = bestRank
            ids.remove(at: at + 1)
        }
        return ids
    }

    /// GPT-2 style split: words, numbers, and punctuation stay in their own chunks
    /// so the dictionary never glues a comma onto the next word.
    private func chunks(_ text: String) -> [String] {
        let pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+"
        guard let re = try? NSRegularExpression(pattern: pattern) else { return [text] }
        let range = NSRange(text.startIndex..., in: text)
        return re.matches(in: text, range: range).compactMap { match in
            Range(match.range, in: text).map { String(text[$0]) }
        }
    }
}
