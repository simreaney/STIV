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
///
/// The recorder is a mutable reference type and is not `Sendable`.  To hand a
/// finished recording to another thread, take its `snapshot` on the thread
/// that appended to it and pass that: it is an immutable value, and costs no
/// copy while the recorder is left alone.
public final class StiRecorder {
    /// Samples per station, in station order.
    public var sampleCounts: [Int] { recording.sampleCounts }
    /// Metres per column, per station.  Varies across the section in an oblique
    /// view: the far bank is coarser than the near bank.
    public var sxPerStation: [Double] { recording.sxPerStation }

    private var recording: StiRecording

    public init(sampleCounts: [Int], sxPerStation: [Double], expectedRows: Int = 0) {
        precondition(sampleCounts.count == sxPerStation.count, "one Sx per station")
        precondition(!sampleCounts.isEmpty, "need at least one station")
        precondition(sampleCounts.allSatisfy { $0 >= 1 }, "every station needs at least one sample")
        recording = StiRecording(sampleCounts: sampleCounts, sxPerStation: sxPerStation)
        if expectedRows > 0 {
            recording.samples.reserveCapacity(expectedRows * recording.stride)
            recording.timestamps.reserveCapacity(expectedRows)
        }
    }

    public var rowCount: Int { recording.rowCount }
    public var samplesPerRow: Int { recording.samplesPerRow }
    public var stationCount: Int { recording.stationCount }

    /// Seconds spanned by the recording so far.
    public var recordedSeconds: Double { recording.recordedSeconds }

    /// An immutable, `Sendable` copy of everything recorded so far.
    ///
    /// Copy-on-write: no samples are duplicated unless the recorder is
    /// appended to again while the snapshot is still alive.
    public var snapshot: StiRecording { recording }

    /// Append one frame's worth of samples, flattened in station order.
    ///
    /// `timestamp` is `ARFrame.timestamp`: monotonic seconds on an unspecified
    /// base, which is all this needs -- only differences are ever used.
    public func append(row: [Float], timestamp: Double) {
        precondition(row.count == recording.stride,
                     "expected \(recording.stride) samples, got \(row.count)")
        recording.samples.append(contentsOf: row)
        recording.timestamps.append(timestamp)
    }

    /// Median interval between recorded rows, i.e. the true capture period.
    /// See `StiRecording.medianFramePeriod`.
    public var medianFramePeriod: Double? { recording.medianFramePeriod }

    /// Build one STI per station, resampled onto a uniform time grid.  See
    /// `StiRecording.finish(frameStep:nominalPeriod:)`.
    public func finish(frameStep: Int, nominalPeriod: Double? = nil) -> [SpaceTimeImage] {
        recording.finish(frameStep: frameStep, nominalPeriod: nominalPeriod)
    }
}

/// A recording as an immutable value: the raw samples, their timestamps and the
/// per-station layout, with everything needed to turn them into STIs.
public struct StiRecording: Sendable {
    /// Samples per station, in station order.
    public let sampleCounts: [Int]
    /// Metres per column, per station.
    public let sxPerStation: [Double]

    let offsets: [Int]
    let stride: Int
    var samples: [Float] = []
    var timestamps: [Double] = []

    init(sampleCounts: [Int], sxPerStation: [Double]) {
        self.sampleCounts = sampleCounts
        self.sxPerStation = sxPerStation
        var offs = [0]
        for c in sampleCounts { offs.append(offs.last! + c) }
        self.offsets = offs
        self.stride = offs.last!
    }

    public var rowCount: Int { timestamps.count }
    public var samplesPerRow: Int { stride }
    public var stationCount: Int { sampleCounts.count }

    /// Seconds spanned by the recording.
    public var recordedSeconds: Double {
        guard let first = timestamps.first, let last = timestamps.last else { return 0 }
        return last - first
    }

    /// Median interval between recorded rows, i.e. the true capture period.
    ///
    /// Preferred over the nominal frame rate when deriving `St`, because a
    /// thermally-throttled phone quietly delivers fewer frames than the video
    /// format advertises.
    ///
    /// Only positive, finite intervals count.  A repeated timestamp is not a
    /// frame period of zero -- it is one frame delivered twice -- and if half
    /// the intervals were such repeats the median of them all would be zero,
    /// which used to put a division by zero, and then `Int(inf)`, at the heart
    /// of `finish`.  `nil` when no interval qualifies.
    public var medianFramePeriod: Double? {
        guard timestamps.count >= 2 else { return nil }
        var deltas = [Double]()
        deltas.reserveCapacity(timestamps.count - 1)
        for i in 1..<timestamps.count {
            let d = timestamps[i] - timestamps[i - 1]
            if d.isFinite && d > 0 { deltas.append(d) }
        }
        guard !deltas.isEmpty else { return nil }
        deltas.sort()
        return deltas[deltas.count / 2]
    }

    /// Build one STI per station, resampled onto a uniform time grid.
    ///
    /// - Parameter frameStep: stride in units of the *nominal* capture period.
    ///   `Survey.validated()` rejects anything below 1; here it is clamped to
    ///   1 instead, because a bad setting must not take the app down with it.
    /// - Parameter nominalPeriod: seconds per recorded row; defaults to the
    ///   measured median, so `St` reflects what the device actually did.
    ///
    /// Rows are produced at `t0 + i * St` by linear interpolation between the
    /// two nearest recorded rows.  With no dropped frames this reduces to
    /// picking every `frameStep`-th row, so the common case costs nothing in
    /// fidelity.
    ///
    /// **A deliberate divergence from the desktop.**  `stiv` stacks one row per
    /// decoded frame and assumes `St = frame_step / fps`; when the container's
    /// timestamps say the clip is not at a constant rate it only *warns*
    /// (`stiv.video.check_frame_timing`), because a container's time base is
    /// often too coarse to resample from.  The phone resamples instead:
    /// `ARFrame.timestamp` is the sensor's own capture time, precise enough to
    /// interpolate on, and a thermally-throttled phone really does drop frames.
    /// So on a recording with dropped frames the two pipelines see different
    /// STIs, and only the phone's streaks are straight.  The golden fixture
    /// pins nothing here -- it starts from an STI, after this step.
    ///
    /// The period is taken from the first of `nominalPeriod`, the measured
    /// median and the mean interval over the whole recording that is finite,
    /// positive and does not ask for more than sixteen output rows per recorded
    /// row -- upsampling that far would invent data rather than resample it,
    /// and an absurd period would otherwise overflow the row count.  The mean
    /// always qualifies when the recording spans any time at all.  A recording
    /// that spans none (fewer than two rows, or identical or non-finite
    /// timestamps) yields one-row STIs, which the analysis reports as too
    /// small rather than crashing on.
    public func finish(frameStep: Int, nominalPeriod: Double? = nil) -> [SpaceTimeImage] {
        let step = Double(max(1, frameStep))
        let fallback = [nominalPeriod, medianFramePeriod]
            .compactMap { $0 }
            .first { $0.isFinite && $0 > 0 } ?? (1.0 / 60.0)

        guard let t0 = timestamps.first, let tEnd = timestamps.last, rowCount >= 2,
              (tEnd - t0).isFinite, tEnd - t0 > 0 else {
            return emptyImages(st: fallback * step)
        }
        let span = tEnd - t0
        let meanPeriod = span / Double(rowCount - 1)
        let maxRows = 16.0 * Double(rowCount)
        let period = [nominalPeriod, medianFramePeriod, meanPeriod]
            .compactMap { $0 }
            .first { $0.isFinite && $0 > 0 && span / ($0 * step) < maxRows } ?? meanPeriod
        let st = period * step

        let outRows = Int((span / st + 1e-9).rounded(.down)) + 1
        guard outRows >= 2 else { return emptyImages(st: st) }

        // Walk the target grid and the recorded rows together -- both are
        // sorted, so this is linear rather than a search per output row -- and
        // note, per output row, which two recorded rows it lies between.
        var lowers = [Int](repeating: 0, count: outRows)
        var uppers = [Int](repeating: 0, count: outRows)
        var weights = [Float](repeating: 0, count: outRows)
        var lower = 0
        for i in 0..<outRows {
            let t = t0 + Double(i) * st
            while lower + 2 < rowCount && timestamps[lower + 1] < t { lower += 1 }
            let upper = min(lower + 1, rowCount - 1)
            let gap = timestamps[upper] - timestamps[lower]
            lowers[i] = lower
            uppers[i] = upper
            weights[i] = gap > 0 ? Float((t - timestamps[lower]) / gap) : 0
        }

        // Then resample straight into each station's own buffer.  An
        // intermediate all-station copy of the resampled grid would add a
        // whole recording's worth of memory -- tens of megabytes for thirty
        // seconds of thirty stations -- on every frame-step change.  The
        // arithmetic is unchanged: interpolate in Float, as stored, then widen.
        let stride = self.stride
        return (0..<stationCount).map { s in
            let n = sampleCounts[s]
            let offset = offsets[s]
            var block = [Double](repeating: 0, count: outRows * n)
            var missing = 0
            samples.withUnsafeBufferPointer { src in
                block.withUnsafeMutableBufferPointer { dst in
                    for i in 0..<outRows {
                        let a = lowers[i] * stride + offset
                        let b = uppers[i] * stride + offset
                        let w = weights[i]
                        let o = i * n
                        for c in 0..<n {
                            let v = w == 0 ? src[a + c] : src[a + c] + w * (src[b + c] - src[a + c])
                            if v.isNaN { missing += 1 }
                            dst[o + c] = Double(v)
                        }
                    }
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

// MARK: - display copies

/// A conditioned STI kept for display, in single precision.
///
/// `StivResult` used to retain every conditioned STI as `Double` -- the same
/// size again as the recording, held for as long as the result is on screen and
/// reallocated on every frame-step re-run.  Display needs eight bits per pixel
/// in the end, so `Float` loses nothing visible at half the memory.
public struct StiPreview: Sendable {
    public let stationIndex: Int
    public let rows: Int
    public let cols: Int
    /// The conditioned STI, row-major, `(time, space)`.
    public let pixels: [Float]
    public let sxMetresPerColumn: Double
    public let stSecondsPerRow: Double
    public let outOfFrameFraction: Double

    init(conditioned: Matrix, of sti: SpaceTimeImage) {
        stationIndex = sti.stationIndex
        rows = conditioned.rows
        cols = conditioned.cols
        pixels = conditioned.data.map { Float($0) }
        sxMetresPerColumn = sti.sxMetresPerColumn
        stSecondsPerRow = sti.stSecondsPerRow
        outOfFrameFraction = sti.outOfFrameFraction
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

    // `Survey.validated()` rejects a non-finite sigma; here it is simply not
    // applied, since an infinite one would otherwise trap sizing the kernel.
    if settings.smoothSigma > 0 && settings.smoothSigma.isFinite {
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
/// single glinting pixel does not flatten the whole image.  Matches
/// `stiv.sti.to_display`, including numpy's interpolated percentiles.
public func displayBytes(_ sti: Matrix) -> [UInt8] {
    stretchToBytes(sti.data)
}

/// `displayBytes` for a retained display copy, without widening it first.
public func displayBytes(_ preview: StiPreview) -> [UInt8] {
    stretchToBytes(preview.pixels.map { Double($0) })
}

func stretchToBytes(_ values: [Double]) -> [UInt8] {
    let finite = values.filter { $0.isFinite }.sorted()
    guard !finite.isEmpty else { return [UInt8](repeating: 0, count: values.count) }
    var lo = numpyPercentile(sorted: finite, 1.0)
    var hi = numpyPercentile(sorted: finite, 99.0)
    if hi - lo < 1e-12 {
        lo = finite[0]
        hi = finite[finite.count - 1]
    }
    guard hi - lo >= 1e-12 else { return [UInt8](repeating: 0, count: values.count) }
    let gain = 255.0 / (hi - lo)
    return values.map { v in
        guard v.isFinite else { return 0 }
        return UInt8(min(max((v - lo) * gain, 0), 255))
    }
}

/// `numpy.percentile(a, q)` with the default `method="linear"`, on data
/// already sorted ascending.
///
/// Reproduced operation for operation -- the virtual index `(n - 1) q`, and
/// numpy's two-sided `_lerp`, which interpolates down from the upper neighbour
/// once the weight reaches one half -- rather than by the textbook formula,
/// which differs in the last bit.
func numpyPercentile(sorted a: [Double], _ percent: Double) -> Double {
    precondition(!a.isEmpty, "percentile of an empty array")
    let q = percent / 100.0
    let virtual = Double(a.count - 1) * q
    if virtual >= Double(a.count - 1) { return a[a.count - 1] }
    if virtual < 0 { return a[0] }
    let previous = virtual.rounded(.down)
    let i = Int(previous)
    let below = a[i]
    let above = a[i + 1]
    let gamma = virtual - previous
    let diff = above - below
    return gamma >= 0.5 ? above - diff * (1 - gamma) : below + diff * gamma
}
