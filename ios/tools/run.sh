#!/usr/bin/env bash
# Build the iPhone app and launch it in the iOS Simulator.
#
#   ios/tools/run.sh             build, install and launch, then return
#   ios/tools/run.sh --logs      the same, then stream the app's console output
#                                here until it quits (Ctrl-C stops it)
#
# Environment:
#   SIMULATOR=<udid>   use this simulator instead of the newest iPhone available
#
# The simulator has no camera, no world tracking and no depth, so the app stops
# at the camera step and says so: this is for looking at the screens, not for a
# survey.  ios/TESTING.md has what can be tested without a phone.

set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
derived="$here/build"            # ignored by git, shared with DEPLOYING.md
logs=""

case "${1:-}" in
    "")     ;;
    --logs) logs=1 ;;
    *)      sed -n '2,13p' "$0" >&2; exit 2 ;;
esac

# The same choice as test.sh: the newest iOS runtime's last-listed iPhone,
# preferring a Pro.
pick_simulator() {
    local list
    list="$(xcrun simctl list devices available | grep -E '^[[:space:]]+iPhone' || true)"
    { echo "$list" | grep ' Pro (' || echo "$list"; } | tail -1 \
        | sed -E 's/.*\(([0-9A-F-]{36})\).*/\1/'
}

sim="${SIMULATOR:-$(pick_simulator)}"
if [[ -z "$sim" ]]; then
    echo "No iPhone simulator found. Install one in Xcode > Settings > Components." >&2
    exit 1
fi

echo "==> Booting the simulator ($sim)"
xcrun simctl boot "$sim" 2>/dev/null || true      # already booted is fine
xcrun simctl bootstatus "$sim" -b >/dev/null
# Bring up the Simulator window if this Xcode ships it as an app of its own.
# Not every Xcode does; the app still runs on the booted device either way.
open -b com.apple.iphonesimulator --args -CurrentDeviceUDID "$sim" 2>/dev/null \
    || echo "    (no Simulator app to open; open the device from Xcode to watch it)"

echo "==> Building STIV"
xcodebuild build \
    -project "$here/StivApp.xcodeproj" \
    -scheme STIV \
    -destination "platform=iOS Simulator,id=$sim" \
    -derivedDataPath "$derived" \
    -quiet

app="$derived/Build/Products/Debug-iphonesimulator/STIV.app"
bundle_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$app/Info.plist")"

echo "==> Installing and launching $bundle_id"
xcrun simctl install "$sim" "$app"
if [[ -n "$logs" ]]; then
    xcrun simctl launch --terminate-running-process --console-pty "$sim" "$bundle_id"
else
    xcrun simctl launch --terminate-running-process "$sim" "$bundle_id" >/dev/null
    echo "==> running"
fi
