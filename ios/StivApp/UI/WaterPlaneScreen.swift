//  WaterPlaneScreen.swift
//  Measuring the one number the whole scale rests on.
//
//  Aim at the **waterline** -- where the water meets the bank -- not at the
//  water.  Infrared is absorbed by water, so LiDAR returns from open surface are
//  sparse and low-confidence; the waterline is at water level by definition, it
//  is texture the sensor handles well, and from a bank it is the nearest thing
//  in the scene.  Close in -- a flume, a culvert, a footbridge -- the sensor
//  reaches the surface itself, and then the switch below says so, because the
//  report should not present an inferred level as a measured one.

import ARKit
import StivKit
import SwiftUI

struct WaterPlaneScreen: View {
    @EnvironmentObject private var model: SurveyModel
    @EnvironmentObject private var capture: ARCaptureSession
    @State private var onWaterSurface = false

    private var hasDepth: Bool { model.availability == .available }

    var body: some View {
        ZStack {
            ARPreview(session: capture.session).ignoresSafeArea()
            DepthReticle(fraction: 0.125, active: capture.planeSampleCount > 0)
                .ignoresSafeArea()

            VStack {
                TrackingBanner(advice: capture.trackingState.advice)
                    .padding(.top, 8)
                Spacer()
                if hasDepth { measuredControls } else { manualControls }
            }
            .padding(16)
        }
    }

    private var measuredControls: some View {
        OverlayCard {
            VStack(alignment: .leading, spacing: 12) {
                Label("Water level", systemImage: "ruler")
                    .font(.headline)

                Text(capture.planeGuidance.isEmpty
                     ? "Aim at the waterline within about 5 m and sweep along it."
                     : capture.planeGuidance)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                ProgressView(
                    value: Double(min(capture.planeSampleCount,
                                      WaterPlaneFitter.minimumSamples)),
                    total: Double(WaterPlaneFitter.minimumSamples))

                Toggle(isOn: $onWaterSurface) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Aiming at the water surface itself").font(.subheadline)
                        Text("Turn this on only at close range, where the sensor reaches "
                           + "the water. Otherwise aim at the waterline.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }

                if !capture.depthAvailable {
                    Label("No depth in this frame yet.", systemImage: "clock")
                        .font(.caption).foregroundStyle(.orange)
                }

                PrimaryButton(title: "Use this water level", systemImage: "checkmark",
                              enabled: capture.planeSampleCount
                                        >= WaterPlaneFitter.minimumSamples) {
                    model.acceptMeasuredPlane(onWaterSurface: onWaterSurface)
                }
            }
        }
    }

    /// The no-LiDAR path.  The geometry is identical -- gravity still supplies
    /// the plane's normal, and ARKit knows the camera's own world height -- so
    /// only the provenance of the water level changes, from a measurement to an
    /// estimate.  Which is exactly what the reported uncertainty then says.
    private var manualControls: some View {
        OverlayCard {
            VStack(alignment: .leading, spacing: 12) {
                Label("Camera height above the water", systemImage: "arrow.up.and.down")
                    .font(.headline)
                Text("This device has no LiDAR scanner, so the water level has to be "
                   + "entered. Hold the phone where you will record from and estimate its "
                   + "height above the water surface.")
                    .font(.footnote).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack {
                    Text("\(decimals(model.manualCameraHeightM, 2)) m")
                        .font(.title2.monospacedDigit())
                    Slider(value: $model.manualCameraHeightM, in: 0.3...12, step: 0.05)
                }

                Text("Scale error from this estimate is about "
                   + "\(percent(0.10 / max(model.manualCameraHeightM, 0.1))), and it "
                   + "shrinks the higher you stand.")
                    .font(.caption2).foregroundStyle(.secondary)

                PrimaryButton(title: "Use this height", systemImage: "checkmark",
                              enabled: capture.camera != nil) {
                    model.acceptManualPlane()
                }
            }
        }
    }
}
