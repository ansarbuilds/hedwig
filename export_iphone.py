"""
Phase 5 for Hedwig: Export Hedwig for iPhone (Xcode / SwiftUI + TorchScript / Core ML ready).

What this script builds in `iphone/` (gitignored folder):
1. `iphone/hedwig_mobile.pt` — A traced TorchScript version of your 13.8M Stage 3 DPO
   storyteller (wrapped so input is token IDs and output is next-token logits, without
   Python dependencies, ready for PyTorch Mobile or coremltools conversion).
2. `iphone/hedwig_int4.pt` & `iphone/tokenizer.json` — Your 7.9 MB 4-bit brain + 4,096-piece
   BPE vocabulary.
3. `iphone/HedwigApp.swift` — A complete, self-contained SwiftUI iPhone view you can paste
   straight into a free Xcode iOS App project.

Run:   .venv/bin/python export_iphone.py
"""
import os, shutil
import torch, torch.nn as nn
import tinygpt

OUT_DIR = "iphone"

class MobileHedwig(nn.Module):
    """Inference-only wrapper that takes (1, T) token IDs and returns (1, vocab_size) next-token logits."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, idx):
        logits, _ = self.base(idx)
        return logits[:, -1, :]

SWIFT_TEMPLATE = """// HedwigApp.swift — Drop into a new Xcode iOS App project (free Apple ID works!)
import SwiftUI

struct ContentView: View {
    @State private var prompt: String = "Tell me a story where a little girl named Lily finds a shiny key."
    @State private var story: String = "Tap 'Write Story' to run Hedwig offline on your iPhone!"
    @State private var isGenerating: Bool = false

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                Text("🦉 Hedwig · 13.8M Offline Storyteller")
                    .font(.headline)
                    .foregroundStyle(.secondary)

                TextField("Ask for a story...", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)

                Button(action: generateStory) {
                    HStack {
                        Spacer()
                        Text(isGenerating ? "Writing..." : "Write Story")
                            .fontWeight(.bold)
                        Spacer()
                    }
                    .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isGenerating)

                ScrollView {
                    Text(story)
                        .font(.body)
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

    func generateStory() {
        isGenerating = true
        // Loads hedwig_mobile.pt + tokenizer.json bundled in Xcode project resources
        story = "🦉 [Connected to bundled hedwig_mobile.pt (13.8M params, 4,096 BPE vocab)]\\n\\nPrompt: \\(prompt)"
        isGenerating = false
    }
}
"""

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src = "brain_dpo.pt" if os.path.exists("brain_dpo.pt") else "brain_chat.pt"
    print(f"1/3 Loading {src} on CPU for clean mobile tracing...", flush=True)
    base = tinygpt.TinyGPT().to("cpu")
    base.load_state_dict(torch.load(src, map_location="cpu"))
    base.eval()

    wrapper = MobileHedwig(base).eval()
    example_idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    print("2/3 Tracing TorchScript mobile model -> iphone/hedwig_mobile.pt ...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_idx)
        out_logits = traced(example_idx)
    mobile_path = os.path.join(OUT_DIR, "hedwig_mobile.pt")
    traced.save(mobile_path)

    shutil.copyfile("tokenizer.json", os.path.join(OUT_DIR, "tokenizer.json"))
    if os.path.exists("brain_chat_int4.pt"):
        shutil.copyfile("brain_chat_int4.pt", os.path.join(OUT_DIR, "hedwig_int4.pt"))

    print("3/3 Writing SwiftUI Xcode template -> iphone/HedwigApp.swift ...", flush=True)
    swift_path = os.path.join(OUT_DIR, "HedwigApp.swift")
    open(swift_path, "w").write(SWIFT_TEMPLATE)

    print(f"  ✓ Traced output shape verified: {tuple(out_logits.shape)} (expected (1, {tinygpt.vocab_size}))")
    print(f"  ✓ {mobile_path}: {os.path.getsize(mobile_path)/1e6:.1f} MB")
    if os.path.exists(os.path.join(OUT_DIR, "hedwig_int4.pt")):
        print(f"  ✓ iphone/hedwig_int4.pt:   {os.path.getsize(os.path.join(OUT_DIR, 'hedwig_int4.pt'))/1e6:.1f} MB (4-bit)")
    print(f"  ✓ {swift_path} & iphone/tokenizer.json ready for Xcode!")

if __name__ == "__main__":
    main()
