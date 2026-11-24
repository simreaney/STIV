//  FlowMapView.swift
//  The flow map: an arrow per station, as long as the water there is fast.
//
//  The desktop's `velocity_map.png` draws these arrows over the video frame.
//  Here they are drawn in plan -- seen from above, true to scale on the water
//  plane -- because a phone looks across a river at a shallow angle, and in its
//  own frame an arrow at the far bank would come out several times shorter
//  than the same speed at the near bank.  The plan is laid out as the user
//  stood: the near bank at the bottom, the far bank at the top, so the water
//  runs across the screen the way it ran across the camera's view.
//
//  The geometry -- where each arrow starts and ends, the key's speed -- is
//  `StivKit.FlowMap`; this file only draws it.

import StivKit
import SwiftUI
import UIKit

struct FlowMapView: View {
    let map: FlowMap

    var body: some View {
        Canvas { context, size in
            FlowMapDrawing(map: map, size: size).draw(in: context)
        }
        .accessibilityElement()
        .accessibilityLabel(accessibilitySummary)
    }

    private var accessibilitySummary: String {
        let speeds = map.arrows.map(\.velocityMs)
        guard let fastest = speeds.max(), let slowest = speeds.min() else {
            return "Flow map: no station produced a velocity."
        }
        return "Flow map: \(map.arrows.count) arrows across the section, from "
            + "\(decimals(slowest, 2)) to \(decimals(fastest, 2)) metres per second."
    }
}

/// The plan's layout on screen, and the drawing.
struct FlowMapDrawing {
    let map: FlowMap
    let size: CGSize

    static let arrowColour = Color(red: 0.92, green: 0.41, blue: 0.20)    // the desktop's orange
    static let failedColour = Color(red: 0.82, green: 0.23, blue: 0.23)
    static let sectionColour = Color(red: 0.16, green: 0.47, blue: 0.84)

    /// Plan axes: `along` runs from the near bank to the far bank, `across`
    /// to the user's right as they faced the far bank.
    private var origin: PlanePoint { map.stations.first ?? PlanePoint(x: 0, z: 0) }
    private var along: PlaneVector {
        if let first = map.stations.first, let last = map.stations.last,
           let unit = try? (last - first).normalised() {
            return unit
        }
        return map.flowDirection.perpendicular
    }
    private var across: PlaneVector { PlaneVector(x: -along.z, z: along.x) }

    private func local(_ p: PlanePoint) -> (x: Double, y: Double) {
        let d = p - origin
        return (d.dot(across), d.dot(along))
    }

    /// Scale and offset that fit every station and arrow tip, with room for
    /// the labels and the key.
    private var fit: (scale: Double, x0: Double, y0: Double) {
        let points = map.stations + map.arrows.map(\.tip)
        let xs = points.map { local($0).x }, ys = points.map { local($0).y }
        var (minX, maxX) = (xs.min() ?? -1, xs.max() ?? 1)
        var (minY, maxY) = (ys.min() ?? -1, ys.max() ?? 1)
        // Never narrower than a fifth of its height: a section with no arrows
        // is still drawn as a line, not as a point stretched across the view.
        let height = max(maxY - minY, 1e-6)
        if maxX - minX < 0.2 * height {
            let mid = 0.5 * (minX + maxX)
            (minX, maxX) = (mid - 0.1 * height, mid + 0.1 * height)
        }
        if maxY - minY < 1e-6 { (minY, maxY) = (minY - 1, maxY + 1) }
        let padX = 44.0, padTop = 26.0, padBottom = 44.0
        let scale = min((Double(size.width) - 2 * padX) / (maxX - minX),
                        (Double(size.height) - padTop - padBottom) / (maxY - minY))
        let usedW = (maxX - minX) * scale, usedH = (maxY - minY) * scale
        let x0 = (Double(size.width) - usedW) / 2 - minX * scale
        let y0 = padTop + (Double(size.height) - padTop - padBottom - usedH) / 2 + maxY * scale
        return (scale, x0, y0)
    }

    func point(_ p: PlanePoint) -> CGPoint {
        let (x, y) = local(p)
        let f = fit
        return CGPoint(x: f.x0 + x * f.scale, y: f.y0 - y * f.scale)
    }

    func draw(in context: GraphicsContext) {
        guard let near = map.stations.first, let far = map.stations.last else { return }
        let a = point(near), b = point(far)

        // The profile on the river: a light band from the section to the tips,
        // over each unbroken run of stations.
        for run in runs() where run.count >= 2 {
            var band = Path()
            band.addLines(run.map { point($0.base) } + run.reversed().map { point($0.tip) })
            band.closeSubpath()
            context.fill(band, with: .color(Self.arrowColour.opacity(0.16)))
            var envelope = Path()
            envelope.addLines(run.map { point($0.tip) })
            context.stroke(envelope, with: .color(Self.arrowColour.opacity(0.7)),
                           style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
        }

        var section = Path()
        section.move(to: a)
        section.addLine(to: b)
        context.stroke(section, with: .color(Self.sectionColour), lineWidth: 2)
        context.draw(Text("near bank").font(.caption2).foregroundStyle(.secondary),
                     at: CGPoint(x: a.x, y: a.y + 12))
        context.draw(Text("far bank").font(.caption2).foregroundStyle(.secondary),
                     at: CGPoint(x: b.x, y: b.y - 12))

        for arrow in map.arrows {
            drawArrow(in: context, from: point(arrow.base), to: point(arrow.tip),
                      hollow: arrow.interpolated)
        }
        for station in map.missing {
            let p = point(station.world)
            var cross = Path()
            cross.move(to: CGPoint(x: p.x - 4, y: p.y - 4))
            cross.addLine(to: CGPoint(x: p.x + 4, y: p.y + 4))
            cross.move(to: CGPoint(x: p.x - 4, y: p.y + 4))
            cross.addLine(to: CGPoint(x: p.x + 4, y: p.y - 4))
            context.stroke(cross, with: .color(Self.failedColour), lineWidth: 2)
        }

        let labelled = Set(map.labelledIndices())
        for arrow in map.arrows where labelled.contains(arrow.index) {
            let base = point(arrow.base), tip = point(arrow.tip)
            let dx = tip.x - base.x, dy = tip.y - base.y
            let length = max((dx * dx + dy * dy).squareRoot(), 1e-6)
            context.draw(Text(decimals(arrow.velocityMs, 2))
                            .font(.caption2.monospacedDigit().weight(.semibold)),
                         at: CGPoint(x: tip.x + 16 * dx / length, y: tip.y + 16 * dy / length))
        }

        drawKey(in: context)
    }

    /// Arrows in unbroken runs of consecutive stations.
    private func runs() -> [[FlowArrow]] {
        var out: [[FlowArrow]] = []
        for arrow in map.arrows {
            if let last = out.last?.last, last.index + 1 == arrow.index {
                out[out.count - 1].append(arrow)
            } else {
                out.append([arrow])
            }
        }
        return out
    }

    private func drawArrow(in context: GraphicsContext, from base: CGPoint, to tip: CGPoint,
                           hollow: Bool) {
        let dx = tip.x - base.x, dy = tip.y - base.y
        let length = (dx * dx + dy * dy).squareRoot()
        guard length > 0.5 else { return }
        let ux = dx / length, uy = dy / length
        // The head scales down with a short arrow, as the desktop's does.
        let head = min(9, 0.4 * length)
        let neck = CGPoint(x: tip.x - head * ux, y: tip.y - head * uy)
        var shaft = Path()
        shaft.move(to: base)
        shaft.addLine(to: neck)
        var triangle = Path()
        triangle.move(to: tip)
        triangle.addLine(to: CGPoint(x: neck.x - 0.5 * head * uy, y: neck.y + 0.5 * head * ux))
        triangle.addLine(to: CGPoint(x: neck.x + 0.5 * head * uy, y: neck.y - 0.5 * head * ux))
        triangle.closeSubpath()
        let colour = hollow ? Self.failedColour : Self.arrowColour
        context.stroke(shaft, with: .color(colour), lineWidth: hollow ? 1.5 : 2.5)
        if hollow {
            context.stroke(triangle, with: .color(colour), lineWidth: 1.5)
        } else {
            context.fill(triangle, with: .color(colour))
        }
    }

    /// The reference arrow, bottom left: how long an arrow `referenceSpeedMs`
    /// is, measured along the flow as the station arrows are.
    private func drawKey(in context: GraphicsContext) {
        guard map.metresPerMs > 0 else { return }
        let reference = map.referenceSpeedMs * map.metresPerMs
        let start = point(origin), end = point(origin + map.flowDirection * reference)
        let length = hypot(end.x - start.x, end.y - start.y)
        let y = Double(size.height) - 14
        drawArrow(in: context, from: CGPoint(x: 10, y: y), to: CGPoint(x: 10 + length, y: y),
                  hollow: false)
        context.draw(Text("\(decimals(map.referenceSpeedMs, map.referenceSpeedMs < 1 ? 2 : 0)) m/s")
                        .font(.caption2.monospacedDigit()),
                     at: CGPoint(x: 16 + length, y: y), anchor: .leading)
    }
}

/// The flow map as a PNG, for the export: `velocity_map.png`, the desktop's
/// name for the same picture.  On the main thread, which is where the export
/// runs; `nil` if it cannot be rendered.
func flowMapPNG(_ result: StivResult) -> Data? {
    MainActor.assumeIsolated {
        let finite = result.velocities.filter(\.isFinite)
        let stats = finite.isEmpty ? "no velocities passed quality control"
            : "mean \(decimals(finite.reduce(0, +) / Double(finite.count), 2)) m/s, "
              + "max \(decimals(finite.max() ?? 0, 2)) m/s"
        let picture = VStack(alignment: .leading, spacing: 6) {
            Text("Surface velocity  -  \(result.survey.section.name)   (\(stats))")
                .font(.headline)
            Text("Seen from above, true to scale. Arrow length is proportional to speed; "
                 + "hollow red arrows failed quality control and were interpolated; "
                 + "a cross is a station with no velocity.")
                .font(.caption).foregroundStyle(.secondary)
            FlowMapView(map: FlowMap(result: result))
        }
        .padding(20)
        .frame(width: 900, height: 640)
        .background(Color.white)
        .environment(\.colorScheme, .light)
        let renderer = ImageRenderer(content: picture)
        renderer.scale = 2
        return renderer.uiImage?.pngData()
    }
}
