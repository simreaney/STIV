# Putting STIV on a test iPhone

There are three routes. Most people want the first.

- **[Xcode and a cable](#first-time-xcode-and-a-cable):** your own phone, in
  about fifteen minutes, with a free Apple ID.
- **[The command line](#repeat-installs-from-the-command-line):** the same
  install, scripted, once the first one has worked.
- **[TestFlight](#other-peoples-phones-testflight):** other people's phones,
  with no Mac on their end. Needs a paid developer account and some
  preparation.

Then read **[Before a field day](#before-a-field-day)**. Two of its points can
cost you a site visit.

## What you need

- A Mac with **Xcode 16 or later**. The project uses Xcode 16's file format and
  will not open in 15.
- An iPhone on **iOS 17 or later**. To measure the water level with LiDAR, it
  must be a **Pro or Pro Max from the iPhone 12 onwards**. Any other iPhone runs
  the app with a typed-in camera height instead.
- A cable, for the first connection.
- An **Apple ID**. A free one is enough for your own phone. A paid
  [Apple Developer Program](https://developer.apple.com/programs/) membership
  lifts the limits below. So does being added to your institution's team.

| | free Apple ID | Developer Program |
|---|---|---|
| an install keeps working for | **7 days** | 1 year |
| phones | your own | up to 100 registered iPhones |
| TestFlight | no | yes |

## First time: Xcode and a cable

1. **Add your Apple ID to Xcode.** Go to *Xcode → Settings → Accounts*, press
   *+*, choose *Apple ID* and sign in.

2. **Open the project.**

   ```bash
   open ios/StivApp.xcodeproj
   ```

3. **Choose who signs the app.** In the left sidebar, click the blue *StivApp*
   project. Under *Targets*, select **STIV** and open *Signing & Capabilities*.
   Leave *Automatically manage signing* ticked and choose your *Team*. A free
   Apple ID appears as "Your Name (Personal Team)". If you will run the
   on-device tests, do the same for **StivAppTests**.

   If Xcode then says the bundle identifier is not available, someone else has
   already registered `org.stiv.app`. Change *Bundle Identifier* to something
   only you would use, such as `uk.ac.durham.yourname.stiv`, and give the tests
   the same name plus `.tests`.

   Choosing a team writes your team ID into `project.pbxproj`. A team ID is not
   a secret. If everyone working on the app signs with the same team, commit
   it. If collaborators use their own teams, leave the change uncommitted, or
   each person's choice will overwrite the last.

4. **Get the phone ready.**
   1. Connect it with the cable and unlock it.
   2. Tap *Trust* and enter the passcode.
   3. Turn on Developer Mode: *Settings → Privacy & Security → Developer Mode*.
      The switch only appears after the phone has been connected to Xcode once.
   4. The phone restarts. When it is back, tap *Turn On* to confirm.

5. **Choose the phone in Xcode.** Pick it from the run-destination menu at the
   top centre of the window. The first time, Xcode spends a few minutes
   preparing the phone ("Copying shared cache symbols"). Let it finish.

6. **Run.** Press **⌘R**. Xcode builds the app, installs it and launches it.

7. **Free Apple ID only: trust yourself.** The first launch stops with
   *Untrusted Developer*. On the phone, go to *Settings → General → VPN &
   Device Management*, tap your Apple ID, then *Trust*. Press ⌘R again.

8. **Allow camera access** when the app asks.

**Check it worked.** Tap *Start* and move the phone slowly until tracking
settles. Then type `parity` into the filter box of Xcode's console. You should
see a line like this:

```
projection parity: 30 points agree with ARKit to 0.004 px
```

If it says **FAILED** instead, stop. Every measurement from that build would be
mirrored. Then run the on-device tests once, still on the cable:

```bash
ios/tools/test.sh device
```

([TESTING.md](TESTING.md) explains what they check.)

**After the first time**, there are only two steps: pick the phone and press
⌘R. To skip the cable, open *Window → Devices and Simulators*, select the
phone, and tick *Connect via network*. The Mac and the phone must be on the
same network.

## Build for the field: Release

⌘R builds the **Debug** configuration, which turns off the compiler's
optimisation. For this app that is a large cost.

- **Speed.** The analysis runs about 85× slower in Debug. On StivKit's
  parabolic-profile test it takes 1.4 s against 0.016 s. On a full 30-station
  recording, that is tens of seconds of waiting on the riverbank against well
  under one.
- **Frame rate.** The per-frame sampling at 60 frames per second has much less
  headroom in Debug.

The one thing Debug has that Release lacks is the projection-parity check
above. Use Debug for that first check, and Release for any phone going outside.

To build Release from Xcode:

1. Open *Product → Scheme → Edit Scheme…* (⌘<).
2. Select *Run*, open the *Info* tab, and set *Build Configuration* to
   **Release**.
3. Press ⌘R.

This edits the shared scheme file `STIV.xcscheme`. Set it back before
committing, or use the command line below, which leaves the scheme alone.

## Repeat installs from the command line

Once the Xcode route has worked once, so that your account, team and Developer
Mode are all set up, this installs a Release build without opening Xcode. Run
it from the repository root.

```bash
xcrun xctrace list devices      # the phone's UDID is the last thing in brackets
UDID=00008130-001234567890ABCD  # yours
TEAM=ABCDE12345                 # yours; see below

xcodebuild build -project ios/StivApp.xcodeproj -scheme STIV \
  -configuration Release -destination "platform=iOS,id=$UDID" \
  -derivedDataPath ios/build -allowProvisioningUpdates DEVELOPMENT_TEAM=$TEAM

xcrun devicectl device install app --device $UDID \
  ios/build/Build/Products/Release-iphoneos/STIV.app
xcrun devicectl device process launch --device $UDID org.stiv.app
```

The team ID is the value Xcode wrote in step 3:

```bash
grep -m1 DEVELOPMENT_TEAM ios/StivApp.xcodeproj/project.pbxproj
```

If you changed the bundle identifier in step 3, use the new one in the last
command. `ios/build` is ignored by git.

## Other people's phones: TestFlight

TestFlight installs the app over the air on anyone's iPhone, with no cable, no
Mac and no Developer Mode on their end. It needs a paid Developer Program
membership, or a role on an institution's team that can upload builds.

**Before the first upload, the project needs two things it does not have yet:**

- **An app icon.** App Store Connect rejects any build without a 1024 × 1024
  icon, and `StivApp/` has no asset catalog yet.
- **An encryption answer.** Add `ITSAppUsesNonExemptEncryption` to
  `Info.plist`, or App Store Connect will ask on every upload. The app makes no
  network connections, so the answer is normally `NO`, but the declaration is
  yours to make.

Then, for each build:

1. Increase *Build* (`CURRENT_PROJECT_VERSION`) under the STIV target's
   *General* tab. Every upload needs a higher build number than the last.
2. Set the run destination to *Any iOS Device (arm64)*, then choose *Product →
   Archive*.
3. In the Organizer window that opens, choose *Distribute App → TestFlight
   Internal Only*. Choose *TestFlight & App Store* instead if you want external
   testers. Then upload.
4. The first time only, in [App Store Connect](https://appstoreconnect.apple.com),
   create the app under *Apps → +* with the same bundle identifier.
5. Wait for the build to finish processing, usually 10–30 minutes. Then open
   the app's *TestFlight* tab and add testers:
   - **Internal testers** are members of your team. There can be up to 100, and
     their builds need no review.
   - **External testers** are invited by email or a public link, but each build
     first needs Apple's beta review.
6. Testers install the *TestFlight* app from the App Store and accept the
   invitation.

TestFlight builds expire after 90 days.

## Before a field day

- **Free Apple ID? Reinstall in the last day or two.** A free-signed install
  stops opening seven days after it was installed. On a riverbank there is no
  fixing that without a Mac.
- **Install a Release build** (see above).
- **Turn Auto-Lock off.** Go to *Settings → Display & Brightness → Auto-Lock →
  Never*. The app does not yet stop the screen locking. A 30-second recording
  can outlast a short Auto-Lock, and a lock interrupts ARKit, which ends the
  recording.
- **Leave Developer Mode on.** Apps installed from Xcode will not launch without
  it. TestFlight installs do not need it.
- **Charge the phone, and keep it out of the sun between runs.** The camera,
  LiDAR and world tracking together are hard work, and a hot phone delivers
  fewer frames.
- **Share every result before moving on.** Nothing is saved on the phone.
  On the Results screen, tap the share button (□↑) and AirDrop the files to a
  Mac, save them to *Files*, or email them. Do this before tapping *New*,
  starting another recording, or leaving the app.
- **Do the null test first**, at each new site. The protocol is in
  [TESTING.md](TESTING.md#4-in-the-field).

## When something goes wrong

| what you see | what to do |
|---|---|
| "…is in a future Xcode project file format" | Install Xcode 16 or later. Failing that, use the XcodeGen fallback in [README.md](README.md). |
| "Signing for 'STIV' requires a development team" | Choose a team (step 3). |
| "Failed to register bundle identifier", or "No profiles for 'org.stiv.app'" | Change the bundle identifier (step 3). |
| The phone is missing from Xcode's destination menu | Unlock it, check you tapped *Trust*, and turn on Developer Mode (step 4). |
| "iOS … is not supported by this version of Xcode" | The phone's iOS is newer than your Xcode. Update Xcode. |
| *Untrusted Developer* on launch | Trust your Apple ID on the phone (step 7). |
| "Could not launch 'STIV'" mentioning a lock | Unlock the phone and run again. |
| The app stopped opening after about a week | The free Apple ID install expired. Run from Xcode again. |
| A black camera view, or nothing happens after *Start* | You declined camera access. Turn it on in *Settings → STIV → Camera*. |
| The welcome screen says there is no LiDAR | Not a Pro iPhone. Enter the camera height by hand instead. |
| Recordings end early | Turn Auto-Lock off (see [Before a field day](#before-a-field-day)). |
