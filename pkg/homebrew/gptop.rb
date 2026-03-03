class Gptop < Formula
  desc "GPU/Accelerator monitor TUI — like nvtop but for Apple Silicon and more"
  homepage "https://github.com/evilsocket/gptop"
  version "0.1.1"
  url "https://github.com/evilsocket/gptop/archive/refs/tags/v#{version}.tar.gz"
  sha256 "d5558cd419c8d46bdc958064cb97f963d1ea793866414c025906ec15033512ed"
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
