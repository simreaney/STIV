//  ARPreview.swift
//  The camera feed, and drawing world geometry over it.
//
//  Two pixel spaces meet here and must not be confused.  The STI samples the
//  *captured image*, in the sensor's native landscape orientation, through
//  `StivKit`'s own projection.  Taps and overlays live in the *viewport*, in
//  SwiftUI points in whatever orientation the device is held, after ARKit's
//  display transform.  ARKit's `projectPoint`/`unprojectPoint` are the only sane
//  route between the world and the viewport, so this file uses them exclusively
//  and never does viewport arithmetic of its own.

import ARKit
import SceneKit
import StivKit
import SwiftUI
import UIKit

/// The live camera feed.
///
/// `ARSCNView` rather than `ARView`: nothing here is rendered in 3-D -- the
/// overlays are SwiftUI shapes on top -- so the lighter of the two views, with
/// an empty scene, is the honest choice.
struct ARPreview: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.scene = SCNScene()
        view.automaticallyUpdatesLighting = false
        view.rendersContinuously = true
        // The app runs the session itself; the view must not restart it.
        view.debugOptions = []
        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {}
}

/// The interface orientation, which `projectPoint` and `unprojectPoint` both
/// need and neither can infer.
///
/// Read from the window scene rather than from `UIDevice.orientation`: the
/// latter reports the physical attitude of the device, which is not the same
/// thing and is famously wrong when the phone is flat -- which, pointed down at
/// a river, it very nearly is.
func currentInterfaceOrientation() -> UIInterfaceOrientation {
    let scene = UIApplication.shared.connectedScenes
        .compactMap { $0 as? UIWindowScene }
        .first { $0.activationState == .foregroundActive }
        ?? UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first
    return scene?.interfaceOrientation ?? .portrait
}

/// World-to-viewport projection for one camera pose and one water plane, bundled
/// so an overlay can be drawn without every shape re-deriving it.
struct OverlayProjector {
    let camera: ARCamera
    let planeY: Double
    let orientation: UIInterfaceOrientation
    let viewportSize: CGSize

    func point(_ world: PlanePoint) -> CGPoint? {
        viewportPoint(world: world, planeY: planeY, camera: camera,
                      orientation: orientation, viewportSize: viewportSize)
    }

    /// Where a tap lands on the water surface, or `nil` if the ray misses.
    func world(at tap: CGPoint) -> PlanePoint? {
        worldPoint(viewportPoint: tap, camera: camera,
                   plane: WaterPlane(y: planeY),
                   orientation: orientation, viewportSize: viewportSize)
    }

    /// Whether a world point is in front of the camera and inside the viewport,
    /// so an overlay can dim what has drifted out of shot.
    func isVisible(_ world: PlanePoint) -> Bool {
        guard let p = point(world) else { return false }
        return p.x >= 0 && p.y >= 0 && p.x <= viewportSize.width && p.y <= viewportSize.height
    }
}

/// Draws the cross-section, the search lines and the flow arrow over the feed.
struct SectionOverlay: View {
    let projector: OverlayProjector
    let start: PlanePoint?
    let end: PlanePoint?
    let layout: SectionLayout?
    let downstream: PlanePoint?
    let waterPolygon: [PlanePoint]

    var body: some View {
        Canvas { context, _ in
            if !waterPolygon.isEmpty {
                var path = Path()
                let points = waterPolygon.compactMap { projector.point($0) }
                if points.count >= 3 {
                    path.addLines(points)
                    path.closeSubpath()
                    context.fill(path, with: .color(.blue.opacity(0.12)))
                    context.stroke(path, with: .color(.blue.opacity(0.5)),
                                   style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
                }
            }

            // Search lines first, so the section reads on top of them.
            if let layout {
                for line in layout.searchLines {
                    guard let a = projector.point(line.samples.first!),
                          let b = projector.point(line.samples.last!) else { continue }
                    var path = Path()
                    path.move(to: a)
                    path.addLine(to: b)
                    context.stroke(path, with: .color(.yellow.opacity(0.75)), lineWidth: 1.5)
                }
            }

            if let start, let end,
               let a = projector.point(start), let b = projector.point(end) {
                var path = Path()
                path.move(to: a)
                path.addLine(to: b)
                context.stroke(path, with: .color(.red), lineWidth: 3)
                marker(context, at: a, colour: .red)
                marker(context, at: b, colour: .red)
            } else if let start, let a = projector.point(start) {
                marker(context, at: a, colour: .red)
            }

            if let downstream, let p = projector.point(downstream) {
                marker(context, at: p, colour: .green)
                if let start, let end,
                   let a = projector.point(start), let b = projector.point(end) {
                    let midpoint = CGPoint(x: 0.5 * (a.x + b.x), y: 0.5 * (a.y + b.y))
                    var path = Path()
                    path.move(to: midpoint)
                    path.addLine(to: p)
                    context.stroke(path, with: .color(.green.opacity(0.7)),
                                   style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func marker(_ context: GraphicsContext, at point: CGPoint, colour: Color) {
        let radius: CGFloat = 9
        let rect = CGRect(x: point.x - radius, y: point.y - radius,
                          width: 2 * radius, height: 2 * radius)
        context.stroke(Path(ellipseIn: rect), with: .color(.white), lineWidth: 3)
        context.stroke(Path(ellipseIn: rect), with: .color(colour), lineWidth: 1.5)
    }
}

/// A centred reticle, for aiming at the waterline while the plane is fitted.
///
/// It matches `WaterPlaneFitter`'s sampling window, so what the user sees framed
/// is exactly what the fit is using.  A reticle that lied about that would be
/// worse than none.
struct DepthReticle: View {
    /// Must match `WaterPlaneFitter.add(reticleFraction:)`.
    let fraction: Double
    let active: Bool

    var body: some View {
        GeometryReader { geometry in
            let side = min(geometry.size.width, geometry.size.height) * fraction * 2
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(active ? Color.green : Color.white.opacity(0.7), lineWidth: 2)
                .frame(width: side, height: side)
                .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
        }
        .allowsHitTesting(false)
    }
}
