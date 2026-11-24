//  WaterOutline.swift
//  Finding the water automatically, so the outline need not be tapped.
//
//  The phone's counterpart of `stiv/watermask.py`.  The outline's one job is
//  to keep the pose cross-check's static-ground patches off the moving water,
//  and the water is the part of the scene that moves once camera motion is
//  accounted for.  The desktop has to estimate the camera motion by registering
//  frames; here every frame carries its own pose, so none is estimated:
//
//  1. Lay a grid of cells over the first of a short burst of frames, and send
//     each cell's ray to the water plane.
//  2. Follow that plane point into each later frame through the frame's own
//     pose, and compare the patch around it between consecutive frames.  On
//     the bank the patch is the same patch -- a bank is not on the plane, but
//     a hand-held phone barely translates, so its parallax is a pixel or two,
//     which the comparison tolerates.  On the water the pattern has moved on
//     and the patch no longer matches.
//  3. Calibrate the threshold between the two from the section the user has
//     already tapped: its middle is water, and the ground just beyond its ends
//     is bank.  A scene where the two do not differ clearly is reported as a
//     failure rather than outlined on a guess.
//  4. Keep the changing region that crosses the section, grow it by a cell --
//     too much water costs a few candidate patches, too little puts one on the
//     river -- and trace its outline back onto the plane.
//
//  As on the desktop, the water's direction of travel is compared with the
//  downstream point the user tapped, which catches a point on the wrong side.

import Foundation

/// A reduced greyscale copy of a camera frame, with the camera that took it.
public struct LumaFrame: Sendable {
    public let width: Int
    public let height: Int
    /// Row-major brightness, 0-255.
    public let pixels: [Float]
    /// The frame's camera, its intrinsics for *this* image size.
    public let projection: WaterPlaneProjection
    public let timestamp: Double

    public init(width: Int, height: Int, pixels: [Float],
                projection: WaterPlaneProjection, timestamp: Double) {
        precondition(pixels.count == width * height, "pixels must be width * height")
        precondition(projection.intrinsics.width == width
                     && projection.intrinsics.height == height,
                     "the intrinsics must describe this image")
        self.width = width
        self.height = height
        self.pixels = pixels
        self.projection = projection
        self.timestamp = timestamp
    }

    /// Bilinear sample at image coordinates (pixel centres at integers), or
    /// nil outside the image.
    @inline(__always)
    func sample(_ u: Double, _ v: Double) -> Float? {
        guard u >= 0, v >= 0, u <= Double(width - 1), v <= Double(height - 1) else { return nil }
        let u0 = Int(u), v0 = Int(v)
        let u1 = min(u0 + 1, width - 1), v1 = min(v0 + 1, height - 1)
        let fu = Float(u - Double(u0)), fv = Float(v - Double(v0))
        let top = pixels[v0 * width + u0] * (1 - fu) + pixels[v0 * width + u1] * fu
        let bottom = pixels[v1 * width + u0] * (1 - fu) + pixels[v1 * width + u1] * fu
        return top * (1 - fv) + bottom * fv
    }
}

extension PinholeIntrinsics {
    /// The same camera for its image reduced `factor` times each way, by
    /// averaging `factor`-square blocks: reduced pixel `i` covers full pixels
    /// `factor * i ..< factor * (i + 1)`, so pixel centres map as
    /// `u' = (u + 0.5) / factor - 0.5`.
    public func reduced(by factor: Int) -> PinholeIntrinsics {
        let f = Double(factor)
        return PinholeIntrinsics(fx: fx / f, fy: fy / f,
                                 cx: (cx + 0.5) / f - 0.5, cy: (cy + 0.5) / f - 0.5,
                                 width: width / factor, height: height / factor)
    }
}

/// What `detectWaterOutline` found.
public struct WaterOutline: Sendable {
    public enum Confidence: String, Sendable {
        case good, weak, failed
    }

    /// The outline on the water plane; empty when detection failed.
    public let polygon: [PlanePoint]
    public let confidence: Confidence
    /// Why it failed, what to check, or a one-line summary.
    public let message: String
    /// Median change between frames over the water samples and over the bank
    /// samples, as 1 - normalised cross-correlation: 0 for an unchanged patch.
    public let waterChange: Double
    public let bankChange: Double
    /// Fraction of the cells that see the water plane found to be water.
    public let waterFraction: Double
    /// Whether the water moves to the downstream point's side of the section;
    /// nil when that could not be told.
    public let flowAgrees: Bool?

    public var ok: Bool { confidence != .failed && polygon.count >= 3 }

    static func failed(_ message: String, water: Double = .nan,
                       bank: Double = .nan) -> WaterOutline {
        WaterOutline(polygon: [], confidence: .failed, message: message,
                     waterChange: water, bankChange: bank, waterFraction: 0, flowAgrees: nil)
    }
}

/// Water must change at least this many times as much as the bank...
public let waterOutlineMinimumContrast = 2.0
/// ...and by at least this much, in 1 - correlation.
public let waterOutlineMinimumChange = 0.05

/// Outline the moving water around a tapped cross-section.
///
/// - Parameters:
///   - frames: a burst of reduced frames, a fraction of a second apart, in
///     any order; at least two.  All on the same water plane.
///   - sectionStart, sectionEnd: the tapped section ends, at the water's edge.
///   - downstream: the tapped downstream point, for the direction check.
///   - cellPx: grid spacing, in reduced pixels.
public func detectWaterOutline(
    frames unsorted: [LumaFrame],
    sectionStart: PlanePoint,
    sectionEnd: PlanePoint,
    downstream: PlanePoint?,
    cellPx: Int = 8
) -> WaterOutline {
    let frames = unsorted.sorted { $0.timestamp < $1.timestamp }
    guard frames.count >= 2, let reference = frames.first else {
        return .failed("too few frames arrived to see the water move")
    }
    let cols = reference.width / cellPx
    let rows = reference.height / cellPx
    guard cols >= 4, rows >= 4 else { return .failed("the frames are too small") }

    // -- each cell's point on the plane, and its change between frames ----- //
    let cellCount = cols * rows
    var world = [PlanePoint?](repeating: nil, count: cellCount)
    for r in 0..<rows {
        for c in 0..<cols {
            let px = PixelPoint(u: (Double(c) + 0.5) * Double(cellPx) - 0.5,
                                v: (Double(r) + 0.5) * Double(cellPx) - 0.5)
            world[r * cols + c] = reference.projection.imageToWorld(px)
        }
    }

    let flowNormal = downstreamNormal(start: sectionStart, end: sectionEnd, downstream: downstream)
    var changes = [[Double]](repeating: [], count: cellCount)
    var shifts: [Int: [(pair: Int, along: Double, size: Double)]] = [:]
    var usablePairs = 0
    // The change is measured across two gaps where there are frames enough:
    // the water's change grows with time, a tracking error's does not.  The
    // direction is measured across one, where the pattern has moved least and
    // is easiest to find again.
    let span = frames.count >= 4 ? 2 : 1
    for later in 1..<frames.count {
        let b = frames[later]
        let a = frames[later - 1]
        let c = later >= span ? frames[later - span] : nil
        var any = false
        for cell in 0..<cellCount {
            guard let point = world[cell],
                  let pa = a.projection.worldToImage(point),
                  let pb = b.projection.worldToImage(point) else { continue }
            if let c, let pc = c.projection.worldToImage(point),
               let long = PatchMatch(from: c, at: pc, to: b, near: pb, searching: false) {
                changes[cell].append(long.change)
                any = true
            }
            guard let match = PatchMatch(from: a, at: pa, to: b, near: pb) else { continue }
            let pair = later
            // Which way did the pattern go?  Only where it plainly moved and
            // was found again.
            if let flowNormal, let shift = match.shift,
               let ahead = b.projection.worldToImage(point + flowNormal * 0.5) {
                let e = (u: ahead.u - pb.u, v: ahead.v - pb.v)
                let length = (e.u * e.u + e.v * e.v).squareRoot()
                if length > 1e-6 {
                    let along = (shift.u * e.u + shift.v * e.v) / length
                    let size = (shift.u * shift.u + shift.v * shift.v).squareRoot()
                    shifts[cell, default: []].append((pair, along, size))
                }
            }
        }
        if any { usablePairs += 1 }
    }
    let score: [Double?] = changes.map { $0.isEmpty ? nil : median($0) }

    // -- calibrate water against bank from the tapped section ------------- //
    func cellIndex(_ p: PlanePoint) -> Int? {
        guard let px = reference.projection.worldToImage(p) else { return nil }
        let c = Int(((px.u + 0.5) / Double(cellPx)).rounded(.down))
        let r = Int(((px.v + 0.5) / Double(cellPx)).rounded(.down))
        guard c >= 0, c < cols, r >= 0, r < rows else { return nil }
        return r * cols + c
    }
    let along = sectionEnd - sectionStart
    let waterCells = Set(stride(from: 0.2, through: 0.8, by: 0.025)
        .compactMap { cellIndex(sectionStart + along * $0) })
    let bankTs = Array(stride(from: -0.30, through: -0.06, by: 0.02))
        + Array(stride(from: 1.06, through: 1.30, by: 0.02))
    let bankCells = Set(bankTs.compactMap { cellIndex(sectionStart + along * $0) })
    let waterSamples = waterCells.compactMap { score[$0] }
    let bankSamples = bankCells.compactMap { score[$0] }
    let known = score.compactMap { $0 }
    guard waterSamples.count >= 3 else {
        return .failed("the middle of the cross-section is not in view, or shows no texture")
    }
    let water = median(waterSamples)
    let bank = bankSamples.count >= 3 ? median(bankSamples) : percentile(known, 0.1)
    let floor = 0.03
    guard water >= waterOutlineMinimumChange,
          water >= waterOutlineMinimumContrast * max(bank, floor) else {
        return .failed(
            "the water around the cross-section does not change clearly more than the banks "
            + "beyond its ends (\(decimals(water, 2)) against \(decimals(bank, 2)))",
            water: water, bank: bank)
    }

    // -- the changing region that crosses the section, grown by a cell ---- //
    let threshold = (water * max(bank, floor)).squareRoot()
    var mask = score.map { ($0 ?? 0) > threshold }
    mask = dilate(erode(mask, cols, rows), cols, rows)          // open: specks
    mask = erode(dilate(mask, cols, rows), cols, rows)          // close: gaps
    mask = fillHoles(mask, cols, rows)
    let kept = components(mask, cols, rows, touching: waterCells)
    guard kept.contains(true) else {
        return .failed("no changing region crosses the cross-section", water: water, bank: bank)
    }
    let grown = dilate(kept, cols, rows)
    let waterFraction = Double(zip(kept, score).filter { $0 && $1 != nil }.count)
        / Double(max(known.count, 1))

    let boundary = traceBoundary(grown, cols, rows)
    let pixels = boundary.map { PixelPoint(u: (Double($0.c) + 0.5) * Double(cellPx) - 0.5,
                                           v: (Double($0.r) + 0.5) * Double(cellPx) - 0.5) }
    var epsilon = 0.75 * Double(cellPx)
    var simplified = simplifyClosed(pixels, epsilon: epsilon)
    while simplified.count > 48 {
        epsilon *= 1.4
        simplified = simplifyClosed(pixels, epsilon: epsilon)
    }
    let polygon = simplified.compactMap { reference.projection.imageToWorld($0) }
    guard polygon.count >= 3 else {
        return .failed("the water region is too small to outline", water: water, bank: bank)
    }

    // -- the direction check, over the water cells ------------------------ //
    // A tracking error shifts the whole image a fraction of a pixel, one way in
    // one pair and another way in the next; flowing water moves a consistent
    // way in every pair.  So each pair votes with its own median, and the
    // pairs must agree.
    var flowAgrees: Bool?
    let waterShifts = kept.indices.filter { kept[$0] }.flatMap { shifts[$0] ?? [] }
    var votes: [Double] = []
    for pair in 1..<frames.count {
        let these = waterShifts.filter { $0.pair == pair }
        guard these.count >= 5 else { continue }
        let along = median(these.map(\.along))
        if median(these.map(\.size)) >= 1, abs(along) >= 0.5 { votes.append(along) }
    }
    if votes.count * 2 >= frames.count - 1 {
        let forward = votes.filter { $0 > 0 }.count
        if forward * 4 >= votes.count * 3 { flowAgrees = true }
        if (votes.count - forward) * 4 >= votes.count * 3 { flowAgrees = false }
    }

    var concerns: [String] = []
    let middle = PlanePoint(x: 0.5 * (sectionStart.x + sectionEnd.x),
                            z: 0.5 * (sectionStart.z + sectionEnd.z))
    if !contains(polygon: polygon, point: middle) {
        concerns.append("the outline misses the middle of the cross-section")
    }
    if waterFraction < 0.03 { concerns.append("very little of the view was found to be water") }
    if waterFraction > 0.97 { concerns.append("almost the whole view was found to be water") }
    if usablePairs * 2 < frames.count - 1 {
        concerns.append("only \(usablePairs) of \(frames.count - 1) frame pairs could be compared")
    }
    if !concerns.isEmpty {
        return WaterOutline(polygon: polygon, confidence: .weak,
                            message: concerns.joined(separator: "; "),
                            waterChange: water, bankChange: bank,
                            waterFraction: waterFraction, flowAgrees: flowAgrees)
    }
    return WaterOutline(
        polygon: polygon, confidence: .good,
        message: "water outlined (\(percent(waterFraction)) of the view; it changes about "
            + "\(Int((water / max(bank, floor)).rounded()))x as much as the banks)",
        waterChange: water, bankChange: bank,
        waterFraction: waterFraction, flowAgrees: flowAgrees)
}

/// Unit normal to the section pointing to the downstream point's side.
private func downstreamNormal(start: PlanePoint, end: PlanePoint,
                              downstream: PlanePoint?) -> PlaneVector? {
    guard let downstream, let along = try? (end - start).normalised() else { return nil }
    let normal = along.perpendicular
    let mid = PlanePoint(x: 0.5 * (start.x + end.x), z: 0.5 * (start.z + end.z))
    return (downstream - mid).dot(normal) >= 0 ? normal : -normal
}

/// Whether `point` lies inside `polygon` (even-odd rule) on the plane.
public func contains(polygon: [PlanePoint], point: PlanePoint) -> Bool {
    var inside = false
    var j = polygon.count - 1
    for i in polygon.indices {
        let a = polygon[i], b = polygon[j]
        if (a.z > point.z) != (b.z > point.z),
           point.x < (b.x - a.x) * (point.z - a.z) / (b.z - a.z) + a.x {
            inside.toggle()
        }
        j = i
    }
    return inside
}

// MARK: - comparing a patch between two frames

/// The patch around `pa` in frame `a`, compared with the patch around `pb` in
/// frame `b` (where the same plane point is now).
private struct PatchMatch {
    /// 1 - the best correlation within a pixel of `pb`: the bank's few pixels
    /// of parallax are forgiven, the water's change is not.
    let change: Double
    /// Where the pattern went, relative to `pb`, when it plainly moved and was
    /// found again.
    let shift: (u: Double, v: Double)?

    static let half = 3                 // 7x7 patch
    /// How far the direction search reaches, reduced pixels: a metre per
    /// second ten metres out, between frames 0.3 s apart.
    static let search = 8
    static let side = 2 * half + 1
    static let window = 2 * (half + search) + 1

    init?(from a: LumaFrame, at pa: PixelPoint, to b: LumaFrame, near pb: PixelPoint,
          searching: Bool = true) {
        let h = Self.half, s = Self.search, side = Self.side, window = Self.window
        // The template, normalised.  A flat patch carries no pattern to compare.
        var template = [Float](repeating: 0, count: side * side)
        var sum: Float = 0, sumSquares: Float = 0
        for j in 0..<side {
            for i in 0..<side {
                guard let value = a.sample(pa.u + Double(i - h), pa.v + Double(j - h)) else {
                    return nil
                }
                template[j * side + i] = value
                sum += value
                sumSquares += value * value
            }
        }
        let n = Float(side * side)
        let mean = sum / n
        let variance = sumSquares / n - mean * mean
        guard variance > 4 else { return nil }          // under 2 grey levels
        let scale = 1 / variance.squareRoot()
        for k in template.indices { template[k] = (template[k] - mean) * scale }

        // The search area in `b`, sampled once at `pb`'s sub-pixel offset.
        var area = [Float](repeating: 0, count: window * window)
        var inside = [Bool](repeating: false, count: window * window)
        let reach = h + s
        for j in 0..<window {
            for i in 0..<window {
                if let value = b.sample(pb.u + Double(i - reach), pb.v + Double(j - reach)) {
                    area[j * window + i] = value
                    inside[j * window + i] = true
                }
            }
        }

        func correlation(_ du: Int, _ dv: Int) -> Float? {
            var sum: Float = 0, sumSquares: Float = 0, cross: Float = 0
            for j in 0..<side {
                let row = (j + s + dv) * window + s + du
                for i in 0..<side {
                    guard inside[row + i] else { return nil }
                    let value = area[row + i]
                    sum += value
                    sumSquares += value * value
                    cross += value * template[j * side + i]
                }
            }
            let mean = sum / n
            let variance = sumSquares / n - mean * mean
            guard variance > 1e-3 else { return 0 }
            // The template is zero-mean, so its cross term needs no centring.
            return cross / (n * variance.squareRoot())
        }

        // Half a pixel either way is forgiven: a tracking error or a bank's
        // parallax moves the pattern that little, whereas water a tenth of a
        // metre per second ten metres out moves it a pixel or more.
        var nearest: Float = -1
        for dv in [-0.5, 0, 0.5] {
            for du in [-0.5, 0, 0.5] {
                if let r = Self.correlation(template, in: b, at: PixelPoint(u: pb.u + du, v: pb.v + dv)) {
                    nearest = max(nearest, r)
                }
            }
        }
        guard nearest > -1 else { return nil }
        change = Double(1 - nearest)

        // Only a patch that changed is searched for: the bank's direction
        // means nothing, and skipping it keeps the search cheap.
        var shift: (u: Double, v: Double)?
        if searching, change > 0.2 {
            // Coarse, every other pixel, then refined around the best.
            var best: Float = -1, bestU = 0, bestV = 0
            for dv in stride(from: -s, through: s, by: 2) {
                for du in stride(from: -s, through: s, by: 2) {
                    if let r = correlation(du, dv), r > best {
                        best = r; bestU = du; bestV = dv
                    }
                }
            }
            let (coarseU, coarseV) = (bestU, bestV)
            for dv in max(coarseV - 1, -s)...min(coarseV + 1, s) {
                for du in max(coarseU - 1, -s)...min(coarseU + 1, s) {
                    if let r = correlation(du, dv), r > best {
                        best = r; bestU = du; bestV = dv
                    }
                }
            }
            // Found again, and not merely at the edge of the search.
            if best >= 0.6, abs(bestU) < s, abs(bestV) < s, bestU != 0 || bestV != 0 {
                shift = (Double(bestU), Double(bestV))
            }
        }
        self.shift = shift
    }

    /// Correlation of the normalised `template` with `frame`'s patch centred at
    /// `centre`, sampled at that exact sub-pixel position; nil off the image.
    private static func correlation(_ template: [Float], in frame: LumaFrame,
                                    at centre: PixelPoint) -> Float? {
        let h = half, side = Self.side
        var sum: Float = 0, sumSquares: Float = 0, cross: Float = 0
        for j in 0..<side {
            for i in 0..<side {
                guard let value = frame.sample(centre.u + Double(i - h),
                                               centre.v + Double(j - h)) else { return nil }
                sum += value
                sumSquares += value * value
                cross += value * template[j * side + i]
            }
        }
        let n = Float(side * side)
        let variance = sumSquares / n - (sum / n) * (sum / n)
        guard variance > 1e-3 else { return 0 }
        return cross / (n * variance.squareRoot())
    }
}

// MARK: - the cell grid

private func neighbours(_ index: Int, _ cols: Int, _ rows: Int) -> [Int] {
    let r = index / cols, c = index % cols
    var out: [Int] = []
    for dr in -1...1 {
        for dc in -1...1 where dr != 0 || dc != 0 {
            let rr = r + dr, cc = c + dc
            if rr >= 0, rr < rows, cc >= 0, cc < cols { out.append(rr * cols + cc) }
        }
    }
    return out
}

private func dilate(_ mask: [Bool], _ cols: Int, _ rows: Int) -> [Bool] {
    mask.indices.map { i in mask[i] || neighbours(i, cols, rows).contains { mask[$0] } }
}

private func erode(_ mask: [Bool], _ cols: Int, _ rows: Int) -> [Bool] {
    // Cells past the grid's edge count as set, so a region reaching the edge
    // of the view is not eaten away from it.
    mask.indices.map { i in mask[i] && neighbours(i, cols, rows).allSatisfy { mask[$0] } }
}

/// Unset cells that cannot be reached from the grid's edge without crossing
/// a set one -- holes -- are set.
private func fillHoles(_ mask: [Bool], _ cols: Int, _ rows: Int) -> [Bool] {
    var outside = [Bool](repeating: false, count: mask.count)
    var queue: [Int] = []
    for i in mask.indices where !mask[i] {
        let r = i / cols, c = i % cols
        if r == 0 || c == 0 || r == rows - 1 || c == cols - 1 {
            outside[i] = true
            queue.append(i)
        }
    }
    while let i = queue.popLast() {
        let r = i / cols, c = i % cols
        for (rr, cc) in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        where rr >= 0 && rr < rows && cc >= 0 && cc < cols {
            let j = rr * cols + cc
            if !mask[j], !outside[j] {
                outside[j] = true
                queue.append(j)
            }
        }
    }
    return mask.indices.map { mask[$0] || !outside[$0] }
}

/// The 8-connected components of `mask` that include any of `seeds`.
private func components(_ mask: [Bool], _ cols: Int, _ rows: Int, touching seeds: Set<Int>) -> [Bool] {
    var kept = [Bool](repeating: false, count: mask.count)
    var queue = seeds.filter { mask[$0] }.sorted()
    for i in queue { kept[i] = true }
    while let i = queue.popLast() {
        for j in neighbours(i, cols, rows) where mask[j] && !kept[j] {
            kept[j] = true
            queue.append(j)
        }
    }
    return kept
}

/// The outer boundary of the largest set region, as cells in order around it
/// (Moore-neighbour tracing, with Jacob's stopping criterion).
private func traceBoundary(_ mask: [Bool], _ cols: Int, _ rows: Int) -> [(c: Int, r: Int)] {
    func on(_ c: Int, _ r: Int) -> Bool {
        c >= 0 && c < cols && r >= 0 && r < rows && mask[r * cols + c]
    }
    // The 8 neighbours, clockwise on screen (rows grow downward), from west.
    let steps = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
    func direction(_ dc: Int, _ dr: Int) -> Int {
        steps.firstIndex { $0.0 == dc && $0.1 == dr } ?? 0
    }
    // Start at the first set cell in raster order: its west neighbour is
    // unset, which is where the trace begins looking.
    guard let first = mask.firstIndex(of: true) else { return [] }
    let start = (c: first % cols, r: first / cols)
    var current = start
    var back = 0
    var boundary = [start]
    for _ in 0..<(8 * mask.count) {
        var moved = false
        for k in 1...8 {
            let d = (back + k) % 8
            let next = (c: current.c + steps[d].0, r: current.r + steps[d].1)
            guard on(next.c, next.r) else { continue }
            let previous = (back + k - 1) % 8
            let from = (c: current.c + steps[previous].0, r: current.r + steps[previous].1)
            current = next
            back = direction(from.c - next.c, from.r - next.r)
            moved = true
            break
        }
        if !moved { break }                                    // a lone cell
        if current == start && back == 0 { break }             // round again
        if current != start { boundary.append(current) }
    }
    return boundary
}

/// Douglas-Peucker on a closed polygon.
private func simplifyClosed(_ points: [PixelPoint], epsilon: Double) -> [PixelPoint] {
    guard points.count > 3 else { return points }
    // Split at the point furthest from the first, and simplify both halves.
    let first = points[0]
    let far = points.indices.max { a, b in
        squaredDistance(points[a], first) < squaredDistance(points[b], first)
    } ?? 0
    let one = douglasPeucker(Array(points[0...far]), epsilon)
    let two = douglasPeucker(Array(points[far...]) + [first], epsilon)
    return Array(one.dropLast()) + Array(two.dropLast())
}

private func douglasPeucker(_ points: [PixelPoint], _ epsilon: Double) -> [PixelPoint] {
    guard points.count > 2, let a = points.first, let b = points.last else { return points }
    var worst = 0.0, index = 0
    for i in 1..<(points.count - 1) {
        let d = segmentDistance(points[i], a, b)
        if d > worst { worst = d; index = i }
    }
    guard worst > epsilon else { return [a, b] }
    let left = douglasPeucker(Array(points[0...index]), epsilon)
    let right = douglasPeucker(Array(points[index...]), epsilon)
    return Array(left.dropLast()) + right
}

private func squaredDistance(_ a: PixelPoint, _ b: PixelPoint) -> Double {
    (a.u - b.u) * (a.u - b.u) + (a.v - b.v) * (a.v - b.v)
}

private func segmentDistance(_ p: PixelPoint, _ a: PixelPoint, _ b: PixelPoint) -> Double {
    let du = b.u - a.u, dv = b.v - a.v
    let length = du * du + dv * dv
    guard length > 0 else { return squaredDistance(p, a).squareRoot() }
    let t = max(0, min(1, ((p.u - a.u) * du + (p.v - a.v) * dv) / length))
    return squaredDistance(p, PixelPoint(u: a.u + t * du, v: a.v + t * dv)).squareRoot()
}

private func median(_ values: [Double]) -> Double {
    percentile(values, 0.5)
}

private func percentile(_ values: [Double], _ q: Double) -> Double {
    let sorted = values.sorted()
    guard !sorted.isEmpty else { return .nan }
    let position = q * Double(sorted.count - 1)
    let lo = Int(position.rounded(.down)), hi = Int(position.rounded(.up))
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - Double(lo))
}
