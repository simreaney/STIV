//  RecordScreen.swift
//  Thirty seconds of holding still, with the feedback that makes it worthwhile.
//
//  Camera motion is compensated from the AR pose, so a shaking hand does not
//  ruin the measurement outright.  What it does ruin is the *pose*: excessive
//  motion pushes ARKit's tracking into a limited state, and a limited state
//  degrades the metric scale that every velocity is multiplied by.  Hence the
//  live tracking banner and the residual read-out -- they are the difference
//  between a run that can be trusted and one that merely finished.

import ARKit
import StivKit
import SwiftUI

struct RecordScreen: View {
    @EnvironmentObject private var model: SurveyModel
    @EnvironmentObject private var capture: ARCaptureSession

    private var isRecording: Bool { capture.phase == .recording }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                ARPreview(session: capture.session).ignoresSafeArea()

                if let camera = capture.camera, let plane = model.plane {
                    SectionOverlay(
                        projector: OverlayProjector(
                            camera: camera, planeY: plane.y,
                            orientation: currentInterfaceOrientation(),
                            viewportSize: geometry.size),
                        start: model.sectionStart,
                        end: model.sectionEnd,
                        layout: model.layout,
                        downstream: model.downstream,
                        waterPolygon: model.waterPolygon)
                        .ignoresSafeArea()
                }

                VStack {
                    TrackingBanner(advice: capture.trackingState.advice)
                    Spacer()
                    controls
                }
                .padding(16)
            }
        }
        .onChange(of: capture.recordingComplete) {
            if capture.recordingComplete { model.stopRecordingAndAnalyse() }
        }
    }

    private var controls: some View {
        OverlayCard {
            VStack(alignment: .leading, spacing: 12) {
                if isRecording {
                    progress
                } else {
                    briefing
                }

                HStack(spacing: 10) {
                    if !isRecording {
                        Button {
                            model.go(to: .section)
                        } label: {
                            Label("Back", systemImage: "chevron.left")
                                .frame(maxWidth: .infinity).padding(.vertical, 12)
                        }
                        .background(.white.opacity(0.14),
                                    in: RoundedRectangle(cornerRadius: 12))
                    }

                    PrimaryButton(
                        title: isRecording ? "Stop and analyse" : "Start recording",
                        systemImage: isRecording ? "stop.circle" : "record.circle",
                        enabled: model.layout != nil
                    ) {
                        if isRecording {
                            model.stopRecordingAndAnalyse()
                        } else {
                            model.startRecording()
                        }
                    }
                }
            }
        }
    }

    private var briefing: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Hold as still as you can", systemImage: "hand.raised")
                .font(.headline)
            Text("Brace the phone against a rail or a post if there is one. "
               + "\(Int(model.survey.capture.durationSeconds)) seconds, "
               + "\(model.survey.section.stationCount) stations.")
                .font(.footnote).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if !capture.depthAvailable && model.availability == .available {
                Label("No depth in view: the camera-motion cross-check needs something "
                    + "within 5 m -- some bank in frame -- or it stays off.",
                      systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var progress: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "record.circle").foregroundStyle(.red)
                Text("\(decimals(capture.recordedSeconds, 1)) s of "
                   + "\(Int(model.survey.capture.durationSeconds)) s")
                    .font(.headline.monospacedDigit())
                Spacer()
                Text("\(capture.recordedRows) rows")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            ProgressView(value: min(capture.recordedSeconds,
                                    model.survey.capture.durationSeconds),
                         total: model.survey.capture.durationSeconds)

            // The pose residual, live.  Near zero means the AR pose explains the
            // image motion and the STI is following the river rather than the
            // sensor; growing means the opposite, while there is still time to
            // stop and start again.
            let residuals = capture.motionReport.residualPx.filter(\.isFinite)
            if let latest = residuals.last {
                HStack(spacing: 14) {
                    Gauge(label: "Pose residual", value: "\(decimals(latest, 2)) px",
                          good: latest < 1.5)
                    Gauge(label: "Patches",
                          value: "\(capture.motionReport.patchCount)",
                          good: capture.motionReport.patchCount >= 12)
                    if let shift = capture.motionReport.shiftPx.last {
                        Gauge(label: "Frame shift", value: "\(decimals(shift, 0)) px",
                              good: shift < 100)
                    }
                }
            } else if capture.motionReport.enabled {
                Text("Cross-checking the pose against \(capture.motionReport.patchCount) "
                   + "static patches.")
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                Text("Camera motion is compensated from the AR pose; no near-field patches "
                   + "in view to cross-check it against.")
                    .font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private struct Gauge: View {
        let label: String
        let value: String
        let good: Bool

        var body: some View {
            VStack(alignment: .leading, spacing: 1) {
                Text(label).font(.caption2).foregroundStyle(.secondary)
                Text(value)
                    .font(.callout.monospacedDigit().weight(.semibold))
                    .foregroundStyle(good ? .green : .orange)
            }
        }
    }
}
