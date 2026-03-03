#!/usr/bin/env python3
import datetime
import os
import re
import shutil
import subprocess

UPDATES = {
    'Cargo.toml': {
        'parse': r'^version\s*=\s*"([^"]+)"$',
        'format': 'version = "%s"'
    }
}

def is_correct_working_dir():
    return os.path.exists("Cargo.toml") and os.path.exists('.git') and os.path.exists('src')

def set_working_dir():
    while not is_correct_working_dir():
        os.chdir('..')

    print(f'working dir: {os.getcwd()}')

def get_current_version():
    with open('Cargo.toml', 'rt') as fp:
        manifest = fp.read()

    m = re.findall(UPDATES['Cargo.toml']['parse'], manifest, re.MULTILINE)
    if len(m) != 1:
        print("could not parse current version from Cargo.toml")
        quit()

    return m[0]

def generate_changelog(new_version):
    print(f'generating changelog for version {new_version} ...')

    if not shutil.which("claude"):
        print("error: 'claude' CLI not found in PATH. Install it from https://claude.ai/code")
        quit()

    # get commits since last tag (or all commits if no tags exist)
    try:
        last_tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        log_range = f"{last_tag}..HEAD"
    except subprocess.CalledProcessError:
        log_range = "HEAD"

    commits = subprocess.check_output(
        ["git", "log", log_range, "--oneline", "--pretty=format:%h %s"]
    ).decode().strip()

    if not commits:
        print("no new commits found")
        quit()

    prompt = f"""You are a helpful assistant that generates changelogs for software projects.

Given the following git commits for version {new_version}, generate a changelog.

## Commits

{commits}

## Format

Use this exact format (no version header, just the categorized list):

🚀 New Features
- Description of feature

🐛 Fixes
- Description of fix

📚 Documentation
- Description of doc change

🔧 Miscellaneous
- Description of other change

## Guidance

- Focus on the new features and major fixes.
- Group and summarize other minor changes into "Miscellaneous".
- Add relevant and catchy emojis but ONLY to important changes.
- Only include sections that have entries.
- Do not include the version header, only the categorized list.
- Be concise — one line per item."""

    result = subprocess.run(
        ["claude", "--model", "haiku", "-p", prompt, "--output-format", "text"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"claude failed: {result.stderr}")
        quit()

    changelog = result.stdout.strip()

    old_changelog = ''
    if os.path.exists('CHANGELOG.md'):
        with open('CHANGELOG.md', 'rt') as fp:
            old_changelog = fp.read().strip()

    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    new_changelog = f'## Version {new_version} ({current_date})\n\n{changelog}\n\n{old_changelog}'.strip()

    with open('CHANGELOG.md', 'wt') as fp:
        fp.write(new_changelog + '\n')

def update_files(new_version):
    for file, data in UPDATES.items():
        print(f'updating {file} ...')
        with open(file, 'rt') as fp:
            contents = fp.read()

        result = re.sub(data['parse'], data['format'] % next_ver, contents, 0, re.MULTILINE)
        with open(file, 'wt') as fp:
            fp.write(result)

if __name__ == '__main__':
    # make sure we're in the correct working directory
    set_working_dir()
    # make sure linting is clean
    if os.system("cargo clippy --all-targets -- -D warnings") != 0:
        print("clippy failed")
        quit()

    # make sure tests are passing
    if os.system("cargo test --all-targets") != 0:
        print("tests failed")
        quit()

    os.system("clear")

    # get current version
    current_ver = get_current_version()
    # get next version from user
    next_ver = input("current version is %s, enter next: " % current_ver)
    # generate the changelog using AI
    generate_changelog(next_ver)
    # update files with new version
    update_files(next_ver)
    # make sure Cargo.lock is up to date
    os.system("cargo update -p gptop")
    # show what changed
    os.system("git status")

    print("")
    print("Now remember to:\n")

    print("- Commit, push and create the new tag:\n")
    print("git add Cargo.*")
    print("git add CHANGELOG.md")
    print("git commit -m 'releasing version %s'" % next_ver)
    print("git push")
    print("git tag -a %s -m 'releasing v%s'" % (next_ver, next_ver))
    print("git push origin %s" % next_ver)
    print()

    print("- Verify and publish on crates.io:\n")
    print("cargo publish --dry-run && cargo publish")
    print()

    print("- Add the generated changelog to the GitHub release.")
    print("- Update pkg/brew/gptop.rb once the precompiled binaries are available.")