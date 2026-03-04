class Gptop < Formula
  desc "GPU/Accelerator monitor TUI — like nvtop but for Apple Silicon and more"
  homepage "https://github.com/evilsocket/gptop"
  version "0.2.0"
  url "https://github.com/evilsocket/gptop/archive/refs/tags/#{version}.tar.gz"
  sha256 "bdb7411ff57e99ebdd6464d75dc4259c328e3440bd88ab4049640a1061878a51"
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
