#!/usr/bin/env bash
# Run the iPhone app's automated tests.  See ios/TESTING.md for what each
# layer covers and what it cannot.
#
#   ios/tools/test.sh            StivKit on the Mac, then the app in a simulator
#   ios/tools/test.sh kit        StivKit only: no simulator, ~10 s
#   ios/tools/test.sh app        the app tests only, in a simulator, ~1 min
#   ios/tools/test.sh device     the app tests on a connected iPhone, including
#                                the ARKit checks the simulator has to skip
#
# Environment:
#   SIMULATOR=<udid>   use this simulator instead of the newest iPhone available
#   DEVICE=<udid>      use this iPhone instead of the first one connected
#   TEAM=<team id>     development team for signing; needed for `device`
#                      unless the project already has one selected in Xcode

set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-all}"

run_kit() {
    echo "==> StivKit (swift test, on this Mac)"
    (cd "$here/StivKit" && swift test)
}

# The newest iOS runtime's last-listed iPhone, preferring a Pro (the LiDAR
# models -- irrelevant in a simulator, but it keeps runs comparable).
pick_simulator() {
    local list
    list="$(xcrun simctl list devices available | grep -E '^[[:space:]]+iPhone' || true)"
    { echo "$list" | grep ' Pro (' || echo "$list"; } | tail -1 \
        | sed -E 's/.*\(([0-9A-F-]{36})\).*/\1/'
}

pick_device() {
    xcrun xctrace list devices 2>/dev/null \
        | sed -n '/^== Devices ==/,/^== Simulators ==/p' \
        | grep -E 'iPhone|iPad' | grep -v Simulator | head -1 \
        | sed -E 's/.*\(([0-9A-Fa-f-]{20,})\)$/\1/'
}

run_app() {
    local destination="$1"; shift
    echo "==> STIV app tests ($destination)"
    xcodebuild test \
        -project "$here/StivApp.xcodeproj" \
        -scheme STIV \
        -destination "$destination" \
        -quiet \
        "$@"
}

case "$mode" in
    kit)
        run_kit
        ;;
    app|all)
        [[ "$mode" == all ]] && run_kit
        sim="${SIMULATOR:-$(pick_simulator)}"
        if [[ -z "$sim" ]]; then
            echo "No iPhone simulator found. Install one in Xcode > Settings > Components." >&2
            exit 1
        fi
        # Boot and wait first.  Left to xcodebuild, a cold simulator can still be
        # booting when the test host is launched, which fails the whole run
        # with "Failed to install or launch the test runner".
        xcrun simctl bootstatus "$sim" -b >/dev/null
        run_app "platform=iOS Simulator,id=$sim"
        ;;
    device)
        dev="${DEVICE:-$(pick_device)}"
        if [[ -z "$dev" ]]; then
            echo "No iPhone connected. Plug one in, unlock it, and trust this Mac." >&2
            exit 1
        fi
        extra=(-allowProvisioningUpdates)
        [[ -n "${TEAM:-}" ]] && extra+=("DEVELOPMENT_TEAM=$TEAM")
        echo "Keep the phone unlocked. On the first run, allow camera access; then move"
        echo "it slowly over a textured surface within a couple of metres until done."
        run_app "platform=iOS,id=$dev" "${extra[@]}"
        ;;
    *)
        sed -n '2,15p' "$0" >&2
        exit 2
        ;;
esac

echo "==> all passed"
