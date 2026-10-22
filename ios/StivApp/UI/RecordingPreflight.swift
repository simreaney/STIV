//  RecordingPreflight.swift
//  The two questions to answer before a recording starts: will it fit in
//  memory, and is the phone actually pointing at the section?
//
//  Both are cheap to answer at Start and expensive to discover afterwards.  A
//  recording that outgrows memory is killed by iOS with no warning and nothing
//  saved; one recorded from a pose that does not see the section produces
//  thirty seconds of NaN or of thousand-pixel search lines.  Both are pure
//  functions of the layout, so they are tested without a camera.

import Foundation
import StivKit

/// How much memory a recording may take, and what gives way to stay inside it.
///
/// The recorder keeps every frame -- that is what makes the frame step
/// retunable after the fact -- as one `Float` per sample, and a search line has
/// roughly one sample per pixel of its length in the image.  So the cost is
/// `stations x samples x 4 bytes x frame rate`: about 5 MB/s at the defaults,
/// and nearly 80 MB/s with eighty long stations.  Nothing else in the app comes
/// close, so this is the one budget worth enforcing.
enum RecordingBudget {
    /// The recording's own samples.  The analysis briefly needs a small
    /// multiple of this again while it resamples and conditions the STIs, and
    /// this leaves that comfortably inside what iOS allows a foreground app on
    /// any iPhone with LiDAR.
    static let bytes = 300_000_000
    static let bytesPerSample = MemoryLayout<Float>.size
    /// Below this an STI holds too few streaks to be worth analysing, so
    /// samples give way before duration does.
    static let minimumSeconds = 10.0
    /// Samples are never cut below this many per line: two or three
    /// structure-tensor windows across, which is about the least that gives an
    /// orientation histogram anything to work with.
    static let minimumSamplesPerLine = 64

    struct Decision: Equatable {
        /// The duration to record.
        var seconds: Double
        /// `nil` keeps the layout's own one-sample-per-pixel counts; otherwise
        /// every line is resampled to this many.
        var samplesPerLine: Int?
        var bytesPerSecond: Double
        /// What the recording will hold at the end.
        var totalBytes: Double
        /// What was given up and why, for the user; `nil` if nothing was.
        var notice: String?
    }

    /// Fit a recording of `requestedSeconds` to the budget: shorten it first,
    /// down to `minimumSeconds`, and only then thin the search lines.
    /// - Parameter sampleCounts: the layout's own counts, as `buildSection`
    ///   chose them -- whatever rule it used, so the budget follows it.
    /// - Parameter minimumSamplesPerLine: the floor for thinning; the caller
    ///   raises it to the analysis window's own minimum when that is larger.
    static func decide(
        sampleCounts: [Int],
        framesPerSecond: Double,
        requestedSeconds: Double,
        minimumSamplesPerLine floor: Int = minimumSamplesPerLine,
        budget: Int = bytes
    ) -> Decision {
        let stations = max(sampleCounts.count, 1)
        let perRow = sampleCounts.reduce(0, +)
        let rate = Double(perRow * bytesPerSample) * framesPerSecond
        let affordable = Double(budget) / max(rate, 1)
        let limit = "the \(megabytes(Double(budget))) memory budget"
        let cost = "\(sampleCounts.count) stations at \(megabytes(rate))/s"

        if requestedSeconds <= affordable {
            return Decision(seconds: requestedSeconds, samplesPerLine: nil,
                            bytesPerSecond: rate, totalBytes: rate * requestedSeconds,
                            notice: nil)
        }
        if affordable >= minimumSeconds {
            let seconds = affordable.rounded(.down)
            return Decision(
                seconds: seconds, samplesPerLine: nil,
                bytesPerSecond: rate, totalBytes: rate * seconds,
                notice: "Recording for \(Int(seconds)) s rather than "
                      + "\(Int(requestedSeconds.rounded())) s: \(cost) would outgrow \(limit). "
                      + "Fewer stations or shorter search lines allow longer.")
        }

        // Even the shortest useful recording does not fit: thin the lines.
        let seconds = min(requestedSeconds, minimumSeconds)
        let perSecondPerSample = Double(bytesPerSample) * framesPerSecond
        var perLine = Int(Double(budget) / (perSecondPerSample * seconds * Double(stations)))
        perLine = max(perLine, floor)
        let thinnedRate = perSecondPerSample * Double(perLine * stations)
        // At the floor, give way on duration after all.
        let finalSeconds = min(seconds, (Double(budget) / thinnedRate).rounded(.down))
        return Decision(
            seconds: finalSeconds, samplesPerLine: perLine,
            bytesPerSecond: thinnedRate, totalBytes: thinnedRate * finalSeconds,
            notice: "Search lines thinned to \(perLine) samples each and the recording "
                  + "limited to \(Int(finalSeconds)) s: \(cost) would outgrow \(limit). "
                  + "Fewer stations or shorter search lines keep full resolution.")
    }

    static func megabytes(_ bytes: Double) -> String {
        "\(decimals(bytes / 1_000_000, bytes < 10_000_000 ? 1 : 0)) MB"
    }
}

/// Whether a layout can be recorded from the view it was built for.
///
/// The layout is rebuilt from the pose the phone has when Start is pressed --
/// on the record screen, aimed at the section -- because a search line's
/// sample count is its length in pixels *from that pose*.  Built from wherever
/// the phone happened to point on another screen, a phone held flat sends
/// every line to StivKit's 4096-sample cap and one facing away gets the
/// 128-sample fallback; either way the recording is wrong before it starts.
enum RecordingCheck {
    /// StivKit's `buildSection` clamps a line's sample count to this; a line
    /// at the clamp is longer in the image than any line in the frame can be.
    static let sampleCap = 4096
    /// A line with less than this share of its samples in frame is not being
    /// filmed, whatever its neighbours are doing.
    static let minimumInFrameShare = 0.5

    /// Why this layout cannot be recorded from `projection`'s view, or `nil`.
    static func problem(layout: SectionLayout, projection: WaterPlaneProjection) -> String? {
        let width = Double(projection.imageWidth - 1)
        let height = Double(projection.imageHeight - 1)
        var behind = 0
        var mostlyOut = 0
        var capped = 0

        for line in layout.searchLines {
            var inFrame = 0
            var lineBehind = false
            for sample in line.samples {
                guard let px = projection.worldToImage(sample) else {
                    lineBehind = true
                    continue
                }
                if px.u >= 0, px.v >= 0, px.u <= width, px.v <= height { inFrame += 1 }
            }
            if lineBehind { behind += 1 }
            if line.sampleCount >= sampleCap { capped += 1 }
            if Double(inFrame) < minimumInFrameShare * Double(max(line.sampleCount, 1)) {
                mostlyOut += 1
            }
        }

        let total = layout.searchLines.count
        if behind > 0 {
            return "The cross-section is behind or beside the camera. Point the phone across "
                 + "the river so the whole section is on screen, then start again."
        }
        if capped > 0 {
            return "The search lines are thousands of pixels long from here: the phone is "
                 + "pointing nearly straight down, or is very close to them. Hold it up and "
                 + "aim across the river at the section, then start again."
        }
        if mostlyOut > 0 {
            return "\(mostlyOut) of \(total) search lines are mostly off screen. Aim so the "
                 + "whole cross-section is in view -- step back or raise the phone if it "
                 + "will not fit -- then start again."
        }
        return nil
    }
}