//  SectionScreen.swift
//  Tapping the cross-section, and seeing it in metres immediately.
//
//  Each tap is a ray through the camera intrinsics intersected with the water
//  plane, so the width appears in metres the moment the second tap lands -- no
//  tape, no control points, no altitude.  That immediacy is the point: a width
//  that reads 2 m when the river is 20 m wide tells the user the water level is
//  wrong *before* they record.

import ARKit
import StivKit
import SwiftUI

struct SectionScreen: View {
    @EnvironmentObject private var model: SurveyModel
    @EnvironmentObject private var capture: ARCaptureSession

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                ARPreview(session: capture.session).ignoresSafeArea()

                if let camera = capture.camera, let plane = model.plane {
                    let projector = OverlayProjector(
                        camera: camera, planeY: plane.y,
                        orientation: currentInterfaceOrientation(),
                        viewportSize: geometry.size)

                    SectionOverlay(projector: projector,
                                   start: model.sectionStart,
                                   end: model.sectionEnd,
                                   layout: model.layout,
                                   downstream: model.downstream,
                                   waterPolygon: model.waterPolygon)
                        .ignoresSafeArea()

                    // A transparent tap target over the whole feed.  The
                    // unprojection can fail -- a tap above the horizon hits no
                    // plane -- and saying so is better than silently ignoring it.
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture { location in
                            if let world = projector.world(at: location) {
                                model.handleTap(world: world)
                                model.rebuildLayout()
                            } else {
                                model.tapMissed()
                            }
                        }
                }

                VStack {
                    TrackingBanner(advice: capture.trackingState.advice)
                    Spacer()
                    controls
                }
                .padding(16)
            }
        }
    }

    private var controls: some View {
        OverlayCard {
            VStack(alignment: .leading, spacing: 10) {
                Text(model.pickTarget.prompt)
                    .font(.subheadline)
                    .fixedSize(horizontal: false, vertical: true)

                if let hint = model.tapHint {
                    Label(hint, systemImage: "hand.tap")
                        .font(.caption).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if let width = model.sectionWidthM {
                    HStack(spacing: 18) {
                        Metric(label: "Section", value: "\(decimals(width, 2)) m")
                        if let plane = model.plane, let camera = capture.camera {
                            let projection = camera.projection(onto: plane)
                            Metric(label: "Camera",
                                   value: "\(decimals(projection.cameraHeightM, 2)) m up")
                        }
                        if let layout = model.layout, let far = layout.searchLines.last {
                            Metric(label: "Far station",
                                   value: "\(decimals(far.rangeM, 1)) m away")
                        }
                    }
                }

                if let layout = model.layout {
                    scaleNote(layout: layout)
                    if !layout.stationsOutsideFrame.isEmpty {
                        Label("\(layout.stationsOutsideFrame.count) search line(s) fall "
                            + "outside the frame. Step back, shorten the search lines, or "
                            + "move the section.", systemImage: "viewfinder.trianglebadge.exclamationmark")
                            .font(.caption).foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                if let error = model.layoutError {
                    Text(error).font(.caption).foregroundStyle(.red)
                }

                HStack(spacing: 10) {
                    Button {
                        model.undoLastTap()
                    } label: {
                        Label("Undo", systemImage: "arrow.uturn.backward")
                            .frame(maxWidth: .infinity).padding(.vertical, 12)
                    }
                    .background(.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 12))

                    PrimaryButton(title: "Settings", systemImage: "slider.horizontal.3",
                                  enabled: model.layout != nil) {
                        model.go(to: .settings)
                    }
                }
            }
        }
    }

    /// The far-field scale warning, shown while it can still be acted on.
    ///
    /// The term is `2 * eps * R / h` -- see `StivKit.Uncertainty` -- so it is
    /// governed by range over camera height and the fix is physical. Saying it
    /// here, before recording, is worth more than reporting it afterwards.
    private func scaleNote(layout: SectionLayout) -> some View {
        let worst = layout.searchLines.map {
            relativeScaleUncertainty(rangeM: $0.rangeM,
                                     cameraHeightM: layout.cameraHeightM,
                                     settings: model.survey.scale)
        }.max() ?? 0
        let threshold = model.survey.quality.maxRelativeScaleUncertainty
        return Group {
            if worst > threshold {
                Label("Scale at the far side is uncertain to about \(percent(worst)). "
                    + "Raising the camera is the only thing that helps much -- this term "
                    + "scales with range ÷ height.",
                      systemImage: "arrow.up.circle")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Label("Scale uncertainty across the section: up to \(percent(worst)).",
                      systemImage: "checkmark.circle")
                    .font(.caption).foregroundStyle(.green)
            }
        }
    }

    private struct Metric: View {
        let label: String
        let value: String

        var body: some View {
            VStack(alignment: .leading, spacing: 1) {
                Text(label).font(.caption2).foregroundStyle(.secondary)
                Text(value).font(.callout.monospacedDigit().weight(.semibold))
            }
        }
    }
}
