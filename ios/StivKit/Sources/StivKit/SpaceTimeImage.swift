//  SpaceTimeImage.swift
//  Construction and conditioning of space-time images.
//
//  Section 2.1 of the paper: for a search line laid along the main flow, the
//  brightness profile is read from N consecutive frames and stacked, giving
//  g(x, t) -- space across, time down.  Surface features advecting along the
//  line draw inclined streaks whose slope is the velocity.
//
//  A port of `stiv/sti.py`, with three changes for the phone:
//
//  * Samples are spaced uniformly in **world** metres, as in the Python, but
//    the sample points are now fixed 3-D points on the water plane rather than
//    fixed image pixels.  Each frame re-projects them through its own camera
//    pose, so the STI follows a patch of *river* however the phone moves.  That
//    is what replaces `stiv/stabilise.py`.
//  * Rows carry their real capture timestamp and are resampled onto a uniform
//    `St` grid, so a dropped frame does not tilt the streaks.  The desktop
//    pipeline can assume `St = frame_step / fps` exactly; a phone cannot.
//  * The recording is kept at the full frame rate and `frameStep` is applied
//    here, at analysis time.  `tan phi` is therefore retunable without
//    re-recording, which matters because it is the method's dominant knob.

import Foundation

/// One STI plus the scales needed to turn its slope into a velocity.
public struct SpaceTimeImage: Sendable {
    /// `(time, space)`; rows are time samples, columns are positions along the
    /// search line.
    public var data: Matrix
    /// Which station this is.
    public let stationIndex: Int
    /// Metres per column -- the `Sx` of Eq. (16).
    public let sxMetresPerColumn: Double
    /// Seconds per row -- the `St` of Eq. (16).
    public let stSecondsPerRow: Double
    /// Fraction of samples that fell outside the frame at some point.
    public let outOfFrameFraction: Double

    public var rows: Int { data.rows }
    public var cols: Int { data.cols }
    public var durationSeconds: Double { Double(rows) * stSecondsPerRow }
    public var lengthM: Double { Double(cols - 1) * sxMetresPerColumn }

    /// `Sx / St`: the velocity corresponding to `tan phi = 1`, which is where
    /// the method is most accurate (Fig. 7).
    public var velocityPerUnitSlope: Double { sxMetresPerColumn / stSecondsPerRow }
}

// MARK: - recording

/// Accumulates one row per frame for every search line at once.
///
/// The recorder is deliberately dumb: it stores raw samples and timestamps and
/// makes no decision that cannot be revisited.  Everything interpretive --
/// the frame step, the preprocessing, the angle range -- happens later, so the
/// results screen can re-analyse a finished recording in place.
///
/// Samples that fell outside the frame are stored as `NaN`, so a genuinely dark
/// pixel is distinguishable from a missing one.
public final class StiRecorder {
    /// Samples per station, in station order.
    public let sampleCounts: [Int]
    /// Metres per column, per station.  Varies across the section in an oblique
    /// view: the far bank is coarser than the near bank.
    public let sxPerStation: [Double]

    private let offsets: [Int]
    private let stride: Int
    private var samples: [Float] = []
    private var timestamps: [Double] = []

    public init(sampleCounts: [Int], sxPerStation: [Double], expectedRows: Int = 0) {
        precondition(sampleCounts.count == sxPerStation.count, "one Sx per station")
        precondition(!sampleCounts.isEmpty, "need at least one station")
        self.sampleCounts = sampleCounts
        self.sxPerStation = sxPerStation

        var offs = [0]
        for c in sampleCounts { offs.append(offs.last! + c) }
        self.offsets = offs
        self.stride = offs.last!

        if expectedRows > 0 {
            samples.reserveCapacity(expectedRows * stride)
            timestamps.reserveCapacity(expectedRows)
        }
    }

    public var rowCount: Int { timestamps.count }
    public var samplesPerRow: Int { stride }
    public var stationCount: Int { sampleCounts.count }

    /// Seconds spanned by the recording so far.
    public var recordedSeconds: Double {
        guard let first = timestamps.first, let last = timestamps.last else { return 0 }
        return last - first
    }

    /// Append one frame's worth of samples, flattened in station order.
    ///
    /// `timestamp` is `ARFrame.timestamp`: monotonic seconds on an unspecified
    /// base, which is all this needs -- only differences are ever used.
    public func append(row: [Float], timestamp: Double) {
        precondition(row.count == stride, "expected \(stride) samples, got \(row.count)")
        samples.append(contentsOf: row)
        timestamps.append(timestamp)
    }

    /// Median interval between recorded rows, i.e. the true capture period.
    ///
    /// Preferred over the nominal frame rate when deriving `St`, because a
    /// thermally-throttled phone quietly delivers fewer frames than the video
    /// format advertises.
    public var medianFramePeriod: Double? {
        guard timestamps.count >= 2 else { return nil }
        var deltas = [Double]()
        deltas.reserveCapacity(timestamps.count - 1)
        for i in 1..<timestamps.count { deltas.append(timestamps[i] - timestamps[i - 1]) }
        deltas.sort()
        return deltas[deltas.count / 2]
    }

    /// Build one STI per station, resampled onto a uniform time grid.
    ///
    /// - Parameter frameStep: stride in units of the *nominal* capture period.
    /// - Parameter nominalPeriod: seconds per recorded row; defaults to the
    ///   measured median, so `St` reflects what the device actually did.
    ///
    /// Rows are produced at `t0 + i * St` by linear interpolation between the
    /// two nearest recorded rows.  With no dropped frames this reduces to
    /// picking every `frameStep`-th row, so the common case costs nothing in
    /// fidelity.
    public func finish(frameStep: Int, nominalPeriod: Double? = nil) -> [SpaceTimeImage] {
        precondition(frameStep >= 1, "frameStep must be >= 1")
        let period = nominalPeriod ?? medianFramePeriod ?? (1.0 / 60.0)
        let st = period * Double(frameStep)

        guard let t0 = timestamps.first, let tEnd = timestamps.last, rowCount >= 2 else {
            return emptyImages(st: st)
        }

        let outRows = max(0, Int(((tEnd - t0) / st + 1e-9).rounded(.down)) + 1)
        guard outRows >= 2 else { return emptyImages(st: st) }

        // Walk the target grid and the recorded rows together: both are sorted,
        // so this is linear rather than a search per output row.
        var resampled = [Float](repeating: .nan, count: outRows * stride)
        var lower = 0
        for i in 0..<outRows {
            let t = t0 + Double(i) * st
            while lower + 2 < rowCount && timestamps[lower + 1] < t { lower += 1 }
            let upper = min(lower + 1, rowCount - 1)
            let span = timestamps[upper] - timestamps[lower]
            let w = span > 0 ? Float((t - timestamps[lower]) / span) : 0
            let a = lower * stride
            let b = upper * stride
            let o = i * stride
            if w == 0 {
                for k in 0..<stride { resampled[o + k] = samples[a + k] }
            } else {
                for k in 0..<stride {
                    resampled[o + k] = samples[a + k] + w * (samples[b + k] - samples[a + k])
                }
            }
        }

        return (0..<stationCount).map { s in
            let n = sampleCounts[s]
            var block = [Double](repeating: 0, count: outRows * n)
            var missing = 0
            for r in 0..<outRows {
                let src = r * stride + offsets[s]
                for c in 0..<n {
                    let v = resampled[src + c]
                    if v.isNaN { missing += 1 }
                    block[r * n + c] = Double(v)
                }
            }
            return SpaceTimeImage(
                data: Matrix(rows: outRows, cols: n, data: block),
                stationIndex: s,
                sxMetresPerColumn: sxPerStation[s],
                stSecondsPerRow: st,
                outOfFrameFraction: Double(missing) / Double(max(1, outRows * n))
            )
        }
    }

    private func emptyImages(st: Double) -> [SpaceTimeImage] {
        (0..<stationCount).map { s in
            SpaceTimeImage(
                data: Matrix(rows: 1, cols: sampleCounts[s], repeating: .nan),
                stationIndex: s,
                sxMetresPerColumn: sxPerStation[s],
                stSecondsPerRow: st,
                outOfFrameFraction: 1.0
            )
        }
    }
}

// MARK: - conditioning

/// Condition an STI before the orientation analysis.
///
/// The background removal is the important step for real footage.  Anything
/// that does not move -- a rock breaking the surface, a bank in the corner of
/// an oblique view, a fixed specular highlight -- writes a perfectly vertical
/// stripe into the STI.  Vertical means `tan phi = 0`, so such features vote
/// hard for zero velocity and drag the Eq. (14) average down.  Subtracting each
/// column's time-mean removes them; subtracting each row's spatial mean removes
/// frame-wide exposure flicker, which writes horizontal stripes instead.
///
/// On a phone the vertical-stripe problem is *worse* than on a drone, because
/// the reprojected search line is anchored to the world: bank vegetation
/// clipping the end of a line stays put for the whole recording.
public func preprocess(_ sti: Matrix, settings: StiSettings) -> Matrix {
    var a = fillMissingWithColumnMean(sti)

    if settings.removeStaticBackground {
        subtractColumnMeans(&a)   // static texture   -> vertical stripes
        subtractRowMeans(&a)      // exposure flicker -> horizontal stripes
    }

    if settings.normalise {
        let (mean, std) = meanAndStd(a.data)
        if std > 1e-12 {
            for i in a.data.indices { a.data[i] = (a.data[i] - mean) / std }
        } else {
            for i in a.data.indices { a.data[i] -= mean }
        }
    }

    if settings.smoothSigma > 0 {
        // Section 2.3: smoothing markedly improves accuracy on smooth (Case C)
        // patterns, and is what lets STIV work without a high-end camera.
        a = a.gaussianMean(sigma: settings.smoothSigma)
    }

    for i in a.data.indices where !a.data[i].isFinite { a.data[i] = 0 }
    return a
}

/// Replace non-finite samples (points that left the frame) with the column's
/// time-mean, so a gap does not read as a brightness edge.
func fillMissingWithColumnMean(_ m: Matrix) -> Matrix {
    var out = m
    guard m.data.contains(where: { !$0.isFinite }) else { return out }

    var grandTotal = 0.0
    var grandCount = 0
    var colMean = [Double](repeating: .nan, count: m.cols)
    for c in 0..<m.cols {
        var total = 0.0
        var count = 0
        for r in 0..<m.rows {
            let v = m[r, c]
            if v.isFinite { total += v; count += 1 }
        }
        if count > 0 { colMean[c] = total / Double(count); grandTotal += total; grandCount += count }
    }
    let grandMean = grandCount > 0 ? grandTotal / Double(grandCount) : 0.0
    for c in 0..<m.cols where !colMean[c].isFinite { colMean[c] = grandMean }

    for r in 0..<m.rows {
        for c in 0..<m.cols where !out[r, c].isFinite { out[r, c] = colMean[c] }
    }
    return out
}

func subtractColumnMeans(_ m: inout Matrix) {
    for c in 0..<m.cols {
        var total = 0.0
        for r in 0..<m.rows { total += m[r, c] }
        let mean = total / Double(m.rows)
        for r in 0..<m.rows { m[r, c] -= mean }
    }
}

func subtractRowMeans(_ m: inout Matrix) {
    for r in 0..<m.rows {
        var total = 0.0
        for c in 0..<m.cols { total += m[r, c] }
        let mean = total / Double(m.cols)
        for c in 0..<m.cols { m[r, c] -= mean }
    }
}

func meanAndStd(_ xs: [Double]) -> (mean: Double, std: Double) {
    guard !xs.isEmpty else { return (0, 0) }
    var total = 0.0
    for x in xs { total += x }
    let mean = total / Double(xs.count)
    var variance = 0.0
    for x in xs { let d = x - mean; variance += d * d }
    return (mean, (variance / Double(xs.count)).squareRoot())
}

/// Stretch an STI to 0-255 for display, using the 1st and 99th percentiles so a
/// single glinting pixel does not flatten the whole image.
public func displayBytes(_ sti: Matrix) -> [UInt8] {
    let finite = sti.data.filter { $0.isFinite }.sorted()
    guard !finite.isEmpty else { return [UInt8](repeating: 0, count: sti.count) }
    let lo = finite[Int(0.01 * Double(finite.count - 1))]
    var hi = finite[Int(0.99 * Double(finite.count - 1))]
    if hi - lo < 1e-12 { hi = finite[finite.count - 1] }
    guard hi - lo > 1e-12 else { return [UInt8](repeating: 0, count: sti.count) }
    return sti.data.map { v in
        guard v.isFinite else { return 0 }
        return UInt8(min(max((v - lo) * (255.0 / (hi - lo)), 0), 255))
    }
}
