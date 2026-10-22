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
///
/// `viewportSize` must be the size of the view the feed is drawn in, and the
/// points it takes and returns must be in that view's coordinates -- which is
/// what `CameraFeed` hands its overlay and its taps, and nothing else does.
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
/// It draws `WaterPlaneFitter`'s own sampling region, mapped into the viewport
/// the way ARKit maps the camera image -- rotated, scaled to fill, cropped --
/// so what the user sees framed is exactly what the fit is using.  A reticle
/// that lied about that would be worse than none: the feed is magnified to
/// fill a screen far narrower than the sensor, so a box sized as a fraction of
/// the *screen* covers a different, smaller part of the scene than the same
/// fraction of the *image*.
struct DepthReticle: View {
    /// `ARCamera.imageResolution`; `nil` until the first frame arrives.
    let imageResolution: CGSize?
    /// The whole viewport the feed is drawn into.
    let viewportSize: CGSize
    let orientation: UIInterfaceOrientation
    let active: Bool

    var body: some View {
        if let rect {
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(active ? Color.green : Color.white.opacity(0.7), lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)
                .allowsHitTesting(false)
        }
    }

    /// The sampled square, in viewport points.
    var rect: CGRect? {
        guard let imageResolution, imageResolution.width > 0 else { return nil }
        let geometry = ViewportGeometry(imageSize: imageResolution, viewportSize: viewportSize,
                                        orientation: orientation)
        return geometry.viewportRect(image: WaterPlaneFitter.reticleRect(
            imageWidth: imageResolution.width, imageHeight: imageResolution.height))
    }
}

// MARK: - one coordinate space for the feed

/// A layer that ignores the safe area and hands its content the size it really
/// has: the whole screen.
///
/// `ARSCNView` draws the feed edge to edge, and ARKit's `projectPoint` and
/// `unprojectPoint` map the world into *that* view.  A `GeometryReader` that
/// respects the safe area reports a smaller size -- 59 points shorter at the
/// top alone on a phone with a Dynamic Island -- and an overlay projected into
/// it lands that far from the feed, and a tap lands on the wrong point of the
/// river.  Ignoring the safe area *on the reader itself* makes the size, the
/// origin and every child's local coordinates the feed's own.
struct FullScreenLayer<Content: View>: View {
    @ViewBuilder let content: (CGSize) -> Content

    var body: some View {
        GeometryReader { geometry in
            content(geometry.size)
                .frame(width: geometry.size.width, height: geometry.size.height)
        }
        .ignoresSafeArea()
    }
}

/// The camera feed, with world-anchored overlays drawn and taps read in the
/// feed's own full-screen coordinates.  Controls go on top of this, in a layer
/// of their own that does respect the safe area.
struct CameraFeed<Overlay: View>: View {
    let session: ARSession
    /// A tap on the feed: where, and in a viewport of what size -- the two
    /// things `OverlayProjector` needs to turn it into a world point.
    var onTap: ((CGPoint, CGSize) -> Void)?
    @ViewBuilder let overlay: (CGSize) -> Overlay

    init(session: ARSession,
         onTap: ((CGPoint, CGSize) -> Void)? = nil,
         @ViewBuilder overlay: @escaping (CGSize) -> Overlay) {
        self.session = session
        self.onTap = onTap
        self.overlay = overlay
    }

    var body: some View {
        FullScreenLayer { size in
            ZStack {
                ARPreview(session: session)
                overlay(size)
                if let onTap {
                    // A transparent tap target over the whole feed, reading
                    // locations in its own -- that is, the feed's -- space.
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture(coordinateSpace: .local) { onTap($0, size) }
                }
            }
        }
    }
}

extension CameraFeed where Overlay == EmptyView {
    init(session: ARSession) {
        self.init(session: session) { _ in EmptyView() }
    }
}

// MARK: - shared banners and cards

/// What the user should know about the session right now: tracking trouble
/// and heat.  Shown wherever the camera is live.
struct StatusBanners: View {
    @EnvironmentObject private var model: SurveyModel
    @EnvironmentObject private var capture: ARCaptureSession

    var body: some View {
        VStack(spacing: 6) {
            TrackingBanner(advice: capture.trackingState.advice)
            if let warning = model.thermalWarning {
                Label(warning, systemImage: "thermometer.sun.fill")
                    .font(.footnote)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.red.opacity(0.85), in: Capsule())
                    .foregroundStyle(.white)
            }
        }
    }
}

/// The camera is not running, why, and what to do about it.
struct CaptureFailureCard: View {
    @EnvironmentObject private var model: SurveyModel
    let failure: CaptureFailure

    var body: some View {
        OverlayCard {
            VStack(alignment: .leading, spacing: 12) {
                Label("The camera is not running", systemImage: "video.slash.fill")
                    .font(.headline)
                Text(failure.message)
                    .font(.footnote)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 10) {
                    if failure.needsCameraPermission,
                       let settings = URL(string: UIApplication.openSettingsURLString) {
                        PrimaryButton(title: "Open Settings", systemImage: "gear") {
                            UIApplication.shared.open(settings)
                        }
                    }
                    PrimaryButton(title: "Try again", systemImage: "arrow.clockwise") {
                        model.retryCapture()
                    }
                }
            }
        }
        .padding(16)
    }
}

/// Back to the water level, from a screen past it.
///
/// Confirmed first when there is a cross-section to lose: every tap was a ray
/// that met the current level, and the same rays meet a new level elsewhere.
struct RemeasureButton: View {
    @EnvironmentObject private var model: SurveyModel
    @State private var confirming = false

    var body: some View {
        Button {
            if model.sectionStart == nil { model.remeasureWaterLevel() } else { confirming = true }
        } label: {
            Label("Re-measure water level", systemImage: "ruler")
        }
        .confirmationDialog("Measure the water level again?", isPresented: $confirming,
                            titleVisibility: .visible) {
            Button("Re-measure", role: .destructive) { model.remeasureWaterLevel() }
        } message: {
            Text("The cross-section will need tapping again: each tap was placed on the "
               + "current water level.")
        }
    }
}
