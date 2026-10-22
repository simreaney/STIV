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
        ZStack {
            CameraFeed(session: capture.session) { size in
                if let camera = capture.camera, let plane = model.plane {
                    SectionOverlay(
                        projector: OverlayProjector(
                            camera: camera, planeY: plane.y,
                            orientation: currentInterfaceOrientation(),
                            viewportSize: size),
                        start: model.sectionStart,
                        end: model.sectionEnd,
                        layout: model.layout,
                        downstream: model.downstream,
                        waterPolygon: model.waterPolygon)
                }
            }

            VStack {
                StatusBanners()
                Spacer()
                controls
            }
            .padding(16)
        }
        .onChange(of: capture.recordingComplete) {
            if capture.recordingComplete { model.stopRecordingAndAnalyse() }
        }
    }

    private var controls: some View {
        OverlayCard {
            VStack(alignment: .leading, spacing: 12) {
                if isRecording || model.isFinishing {
                    progress
                } else {
                    briefing
                }

                HStack(spacing: 10) {
                    if !isRecording && !model.isFinishing {
                        Button {
                            model.go(to: .section)
                        } label: {
                            Label("Back", systemImage: "chevron.left")
                                .frame(maxWidth: .infinity).padding(.vertical, 12)
                        }
                        .background(.white.opacity(0.14),
                                    in: RoundedRectangle(cornerRadius: 12))
                    }

                    // Disabled while the last recording is being handed over:
                    // a Start in that window would begin a run whose results
                    // the previous run's completion could then overwrite.
                    PrimaryButton(
                        title: model.isFinishing ? "Finishing…"
                             : isRecording ? "Stop and analyse" : "Start recording",
                        systemImage: isRecording ? "stop.circle" : "record.circle",
                        enabled: model.layout != nil && !model.isFinishing
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
            Text("Aim so the whole cross-section is on screen. Brace the phone against a "
               + "rail or a post if there is one. "
               + "\(Int(model.survey.capture.durationSeconds)) seconds, "
               + "\(model.survey.section.stationCount) stations.")
                .font(.footnote).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let error = model.recordingError {
                Label(error, systemImage: "exclamationmark.octagon")
                    .font(.caption).foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let estimate = model.recordingEstimate, let notice = estimate.notice {
                Label(notice, systemImage: "memorychip")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !capture.depthAvailable && model.availability == .available {
                Label("No depth in view: the camera-motion cross-check needs something "
                    + "within 5 m -- some bank in frame -- or it stays off.",
                      systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            RemeasureButton()
                .font(.footnote)
        }
    }

    /// The duration this recording will actually run for: the budget may have
    /// shortened it.
    private var targetSeconds: Double {
        model.recordingSeconds ?? model.survey.capture.durationSeconds
    }

    private var progress: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "record.circle").foregroundStyle(.red)
                Text("\(decimals(capture.recordedSeconds, 1)) s of "
                   + "\(Int(targetSeconds)) s")
                    .font(.headline.monospacedDigit())
                Spacer()
                Text("\(capture.recordedRows) rows")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            ProgressView(value: min(capture.recordedSeconds, targetSeconds),
                         total: max(targetSeconds, 1))

            if let notice = model.recordingNotice {
                Label(notice, systemImage: "memorychip")
                    .font(.caption2).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // The pose residual, live, under the correction actually being
            // applied.  Near zero means the AR pose explains the image motion
            // and the STI is following the river rather than the sensor;
            // growing means the opposite, while there is still time to stop and
            // start again.  A check that wanted more correction than the cap
            // allows shows its whole residual and says so.
            let check = capture.poseCheck
            if check.state != .unchecked {
                HStack(spacing: 14) {
                    Gauge(label: "Pose residual",
                          value: check.residualPx.isFinite
                              ? "\(decimals(check.residualPx, 2)) px" : "--",
                          good: check.residualPx.isFinite && check.residualPx < 1.5
                              && (check.state == .applied || check.state == .negligible))
                    Gauge(label: "Correction", value: correctionText(check),
                          good: check.state == .applied || check.state == .negligible)
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

    private func correctionText(_ check: PoseCheckStatus) -> String {
        switch check.state {
        case .applied:   return "\(decimals(check.correctionPx, 1)) px"
        case .negligible: return "none needed"
        case .holding:   return "held"
        case .rejected:  return "over cap"
        case .released:  return "released"
        case .unchecked: return "--"
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