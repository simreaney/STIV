//  SettingsScreen.swift
//  The knobs, with the one that matters first.
//
//  `frameStep` sets `St` and therefore where the measured velocities land on the
//  `tan phi` axis, which section 2.3 of the paper shows is accurate between 0.1
//  and 7 and best near 1.  Everything else here is a refinement.  Defaults come
//  from `stiv/config.py` and the fussier ones stay folded away, as they do in
//  the notebook wizard.

import StivKit
import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject private var model: SurveyModel
    @State private var showAdvanced = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Expected velocity")
                            Spacer()
                            Text("\(decimals(model.survey.sti.velocityRangeMs.lowerBound, 1))"
                               + " to "
                               + "\(decimals(model.survey.sti.velocityRangeMs.upperBound, 1))"
                               + " m/s")
                                .foregroundStyle(.secondary).monospacedDigit()
                        }
                        Slider(value: upperVelocity, in: 0.3...12, step: 0.1)
                        Text("This is the search range of Eq. (14), and the mechanism the "
                           + "paper uses to throw away vectors corrupted by obstacles. Too "
                           + "wide admits noise; too narrow biases the answer toward your "
                           + "guess.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                } header: {
                    Text("What you expect")
                }

                Section {
                    Stepper(value: $model.survey.capture.frameStep, in: 1...30) {
                        HStack {
                            Text("Frame step")
                            Spacer()
                            Text("\(model.survey.capture.frameStep)")
                                .foregroundStyle(.secondary).monospacedDigit()
                        }
                    }
                    HStack {
                        Text("Suggested").foregroundStyle(.secondary)
                        Spacer()
                        Button("\(model.suggestedFrameStep)") {
                            model.survey.capture.frameStep = model.suggestedFrameStep
                        }
                        .buttonStyle(.bordered)
                    }
                    if let tan = medianTanPhi {
                        HStack {
                            Text("tan φ at that velocity").foregroundStyle(.secondary)
                            Spacer()
                            Text(decimals(tan, 2))
                                .monospacedDigit()
                                .foregroundStyle(tan > 0.1 && tan < 7 ? .green : .orange)
                        }
                    }
                } header: {
                    Text("Time scale")
                } footer: {
                    Text("The recording keeps every frame, so this can be changed on the "
                       + "results screen afterwards without recording again.")
                }

                Section("Sampling") {
                    Stepper(value: $model.survey.section.stationCount, in: 3...80) {
                        HStack {
                            Text("Stations across the section")
                            Spacer()
                            Text("\(model.survey.section.stationCount)")
                                .foregroundStyle(.secondary).monospacedDigit()
                        }
                    }
                    VStack(alignment: .leading) {
                        HStack {
                            Text("Search line length")
                            Spacer()
                            Text("\(decimals(model.survey.sti.searchLineLengthM, 1)) m")
                                .foregroundStyle(.secondary).monospacedDigit()
                        }
                        Slider(value: $model.survey.sti.searchLineLengthM,
                               in: 0.5...20, step: 0.5)
                    }
                    VStack(alignment: .leading) {
                        HStack {
                            Text("Recording length")
                            Spacer()
                            Text("\(Int(model.survey.capture.durationSeconds)) s")
                                .foregroundStyle(.secondary).monospacedDigit()
                        }
                        Slider(value: $model.survey.capture.durationSeconds,
                               in: 5...120, step: 5)
                    }
                }

                Section {
                    DisclosureGroup("Orientation analysis", isExpanded: $showAdvanced) {
                        Stepper(value: windowPx, in: 5...81, step: 2) {
                            HStack {
                                Text("Window")
                                Spacer()
                                Text("\(model.survey.sti.windowPx) px")
                                    .foregroundStyle(.secondary).monospacedDigit()
                            }
                        }
                        Picker("Window shape", selection: $model.survey.sti.windowType) {
                            Text("Box (as the paper)").tag(WindowType.box)
                            Text("Gaussian").tag(WindowType.gaussian)
                        }
                        Picker("Gradient stencil", selection: $model.survey.sti.gradientOrder) {
                            Text("2-point").tag(GradientOrder.two)
                            Text("4-point (Eq. 12)").tag(GradientOrder.four)
                            Text("6-point (Eq. 13)").tag(GradientOrder.six)
                        }
                        Picker("Angle estimator", selection: $model.survey.sti.estimator) {
                            Text("Eq. (14) weighted mean").tag(AngleEstimator.eq14)
                            Text("Histogram peak").tag(AngleEstimator.peak)
                        }
                        VStack(alignment: .leading) {
                            HStack {
                                Text("STI smoothing σ")
                                Spacer()
                                Text(decimals(model.survey.sti.smoothSigma, 1))
                                    .foregroundStyle(.secondary).monospacedDigit()
                            }
                            Slider(value: $model.survey.sti.smoothSigma, in: 0...4, step: 0.25)
                            Text("Section 2.3: smoothing markedly improves accuracy on "
                               + "smooth patterns.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        Toggle("Remove static background",
                               isOn: $model.survey.sti.removeStaticBackground)
                    }
                } footer: {
                    Text("Defaults match the desktop package, which follows the 2007 paper: "
                       + "a 31 px window with 50% overlap and the four-point stencil.")
                }
            }
            .navigationTitle("Settings")
            .safeAreaInset(edge: .bottom) {
                PrimaryButton(title: "Record", systemImage: "record.circle") {
                    model.rebuildLayout()
                    model.go(to: .record)
                }
                .padding(16)
                .background(.bar)
            }
            .onChange(of: model.survey.section.stationCount) { model.rebuildLayout() }
            .onChange(of: model.survey.sti.searchLineLengthM) { model.rebuildLayout() }
        }
    }

    /// `tan phi = U St / Sx` at the expected velocity and the current geometry:
    /// the number the frame step is really being chosen against.
    private var medianTanPhi: Double? {
        guard let layout = model.layout,
              let line = layout.searchLines[safe: layout.searchLines.count / 2]
        else { return nil }
        let velocity = 0.5 * (model.survey.sti.velocityRangeMs.lowerBound
                              + model.survey.sti.velocityRangeMs.upperBound)
        return abs(velocity) * model.survey.capture.stSecondsPerRow / line.sxMetresPerColumn
    }

    /// The upper bound moves; the small negative lower bound stays, so a
    /// genuinely still reach is not biased upward.
    private var upperVelocity: Binding<Double> {
        Binding(
            get: { model.survey.sti.velocityRangeMs.upperBound },
            set: { model.survey.sti.velocityRangeMs = -0.5 ... max($0, 0.3) })
    }

    /// The structure-tensor window must stay odd, which a stepper of step 2
    /// preserves only if it starts odd.
    private var windowPx: Binding<Int> {
        Binding(
            get: { model.survey.sti.windowPx },
            set: { model.survey.sti.windowPx = $0 % 2 == 0 ? $0 + 1 : $0 })
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
