class Gptop < Formula
  desc "GPU/Accelerator monitor TUI — like nvtop but for Apple Silicon and more"
  homepage "https://github.com/evilsocket/gptop"
  version "0.1.0"
  url "https://github.com/evilsocket/gptop/archive/refs/tags/v#{version}.tar.gz"
  sha256 "UPDATE_ME"
  license "GPL-3.0"

  depends_on "rust" => :build
  depends_on :macos

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    assert_match "gptop", shell_output("#{bin}/gptop --version")
  end
end
