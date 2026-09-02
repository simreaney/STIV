//  ResultsScreen.swift
//  The profile, its uncertainty, and the reasons not to believe parts of it.
//
//  Same content as the desktop's `velocity_profile.png`, `velocity.csv` and
//  `report.txt`, with one addition the desktop cannot offer: the frame step is
//  live.  The recording holds every frame, so moving that control re-runs the
//  whole analysis in a second or two -- `tan phi` can be walked into the sweet
//  spot of Fig. 7 while looking at the answer, instead of by re-recording.

import Charts
import StivKit
import SwiftUI
import UIKit

struct ResultsScreen: View {
    @EnvironmentObject private var model: SurveyModel
    @State private var shareItems: [URL] = []
    @State private var showingShare = false
    @State private var showingStis = false

    var body: some View {
        NavigationStack {
            Group {
                if let result = model.result {
                    content(result)
                } else if model.isAnalysing {
                    ProgressView("Analysing…")
                } else {
                    ContentUnavailableView(
                        "No result",
                        systemImage: "waveform.slash",
                        description: Text(model.analysisError ?? "Nothing was recorded."))
                }
            }
            .navigationTitle("Results")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("New") { model.go(to: .waterPlane) }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        shareItems = model.exportFiles()
                        showingShare = !shareItems.isEmpty
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .disabled(model.result == nil)
                }
            }
            .sheet(isPresented: $showingShare) { ShareSheet(items: shareItems) }
            .sheet(isPresented: $showingStis) {
                if let result = model.result { StiSheet(result: result) }
            }
        }
    }

    private func content(_ result: StivResult) -> some View {
        List {
            Section {
                VelocityProfileChart(result: result)
                    .frame(height: 220)
                CoherencyChart(result: result)
                    .frame(height: 90)
            } header: {
                Text("Surface velocity")
            } footer: {
                Text("The band is ±1σ: the spread of the Eq. (14) angle histogram "
                   + "propagated through Eq. (16), combined with the scale uncertainty at "
                   + "each station's range.")
            }

            Section("Summary") {
                ForEach(summaryRows(result), id: \.label) { row in
                    HStack {
                        Text(row.label).foregroundStyle(.secondary)
                        Spacer()
                        Text(row.value).monospacedDigit()
                    }
                    .font(.callout)
                }
            }

            if model.canRetune {
                Section {
                    Stepper(value: frameStep, in: 1...30) {
                        HStack {
                            Text("Frame step")
                            Spacer()
                            Text("\(model.survey.capture.frameStep)")
                                .foregroundStyle(.secondary).monospacedDigit()
                        }
                    }
                    if let suggested = result.suggestedFrameStep,
                       suggested != model.survey.capture.frameStep {
                        Button("Use suggested step (\(suggested))") {
                            model.setFrameStep(suggested)
                        }
                    }
                    if model.isAnalysing { ProgressView() }
                } header: {
                    Text("Retune")
                } footer: {
                    Text("Accuracy is best when tan φ is near 1 (Fig. 7). Changing the "
                       + "frame step re-analyses the recording you already have.")
                }
            }

            if !result.warnings.isEmpty {
                Section("Warnings") {
                    ForEach(result.warnings.indices, id: \.self) { index in
                        Label(result.warnings[index],
                              systemImage: "exclamationmark.triangle")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    }
                }
            }

            Section("Stations") {
                Button {
                    showingStis = true
                } label: {
                    Label("Space-time images", systemImage: "square.grid.3x3")
                }
                ForEach(result.stations, id: \.index) { station in
                    StationRow(station: station)
                }
            }

            if result.stations.contains(where: { !$0.flags.isEmpty }) {
                Section("Flags") {
                    ForEach(usedFlags(result), id: \.self) { flag in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(flag.rawValue).font(.caption.monospaced())
                            Text(flag.explanation).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    private var frameStep: Binding<Int> {
        Binding(get: { model.survey.capture.frameStep }, set: { model.setFrameStep($0) })
    }

    private func usedFlags(_ result: StivResult) -> [StationFlag] {
        var seen: [StationFlag] = []
        for station in result.stations {
            for flag in station.flags where !seen.contains(flag) { seen.append(flag) }
        }
        return seen
    }

    private func summaryRows(_ result: StivResult) -> [(label: String, value: String)] {
        let finite = result.velocities.filter(\.isFinite).sorted()
        var rows: [(label: String, value: String)] = [
            ("Section width", "\(decimals(result.widthM, 2)) m"),
            ("Stations", "\(result.stations.filter(\.ok).count) clean "
                       + "of \(result.stations.count)"),
            ("Duration", "\(decimals(result.durationSeconds, 1)) s, \(result.rowCount) rows"),
            ("St", "\(decimals(result.stSecondsPerRow * 1000, 1)) ms/row"),
            ("Camera height", "\(decimals(result.cameraHeightM, 2)) m"),
        ]
        if !finite.isEmpty {
            rows.insert(("Mean velocity",
                         "\(decimals(finite.reduce(0, +) / Double(finite.count), 3)) m/s"), at: 0)
            rows.insert(("Max velocity", "\(decimals(finite[finite.count - 1], 3)) m/s"), at: 1)
        }
        let tans = result.stations.map { abs($0.tanPhi) }.filter(\.isFinite).sorted()
        if !tans.isEmpty {
            rows.append(("Median tan φ", decimals(tans[tans.count / 2], 2)))
        }
        return rows
    }
}

// MARK: - charts

/// Velocity across the section with its uncertainty band, and the failed
/// stations marked.
///
/// Interpolated stations are drawn hollow.  The desktop plot does the same, and
/// for the same reason: a filled circle over a station that produced no data
/// would be a lie that reads as a measurement.
struct VelocityProfileChart: View {
    let result: StivResult

    var body: some View {
        Chart {
            ForEach(result.stations, id: \.index) { station in
                if station.velocityMs.isFinite, station.velocityStdMs.isFinite {
                    AreaMark(
                        x: .value("Distance", station.stationM),
                        yStart: .value("Low", station.velocityMs - station.velocityStdMs),
                        yEnd: .value("High", station.velocityMs + station.velocityStdMs))
                        .foregroundStyle(.blue.opacity(0.18))
                }
            }
            ForEach(result.stations, id: \.index) { station in
                if station.velocityMs.isFinite {
                    LineMark(x: .value("Distance", station.stationM),
                             y: .value("Velocity", station.velocityMs))
                        .foregroundStyle(.blue)
                    PointMark(x: .value("Distance", station.stationM),
                              y: .value("Velocity", station.velocityMs))
                        .symbol(station.flags.contains(.interpolated) ? .circle : .square)
                        .foregroundStyle(station.ok ? .blue
                                         : (station.failed ? .red : .orange))
                }
            }
            RuleMark(y: .value("Zero", 0.0))
                .foregroundStyle(.gray.opacity(0.4))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
        }
        .chartXAxisLabel("Distance along section (m)")
        .chartYAxisLabel("Surface velocity (m/s)")
    }
}

/// Coherency below the profile, as in the desktop figure.
///
/// It is the single most diagnostic number per station: `(λ₁−λ₂)/(λ₁+λ₂)` is 1
/// for a clean streak and 0 for texture-free water, so a profile that looks
/// plausible over a floor of low coherency is not a measurement.
struct CoherencyChart: View {
    let result: StivResult

    var body: some View {
        Chart {
            ForEach(result.stations, id: \.index) { station in
                BarMark(x: .value("Distance", station.stationM),
                        y: .value("Coherency", station.coherency))
                    .foregroundStyle(station.coherency > 0.3 ? .green : .orange)
            }
            RuleMark(y: .value("Threshold", result.survey.quality.minMeanCoherency))
                .foregroundStyle(.red.opacity(0.6))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
        }
        .chartYScale(domain: 0...1)
        .chartYAxisLabel("Coherency")
    }
}

// MARK: - rows and sheets

struct StationRow: View {
    let station: StationResult

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("\(decimals(station.stationM, 2)) m")
                .font(.callout.monospacedDigit())
                .frame(width: 74, alignment: .leading)

            VStack(alignment: .leading, spacing: 2) {
                Text(station.velocityMs.isFinite
                     ? "\(decimals(station.velocityMs, 3)) ± "
                       + "\(decimals(station.velocityStdMs, 3)) m/s"
                     : "no data")
                    .font(.callout.monospacedDigit())
                if !station.flags.isEmpty {
                    Text(station.flags.map(\.rawValue).joined(separator: ", "))
                        .font(.caption2.monospaced())
                        .foregroundStyle(station.failed ? .red : .orange)
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text("C \(decimals(station.coherency, 2))")
                Text("tan φ \(decimals(station.tanPhi, 2))")
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
        }
    }
}

/// The contact sheet of every space-time image, as `sti_panel.png`.
///
/// Worth looking at even when the numbers are fine: the streaks either are there
/// or they are not, and no summary statistic conveys that as directly as seeing
/// them.
struct StiSheet: View {
    let result: StivResult
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 10)],
                          spacing: 12) {
                    ForEach(result.stis.indices, id: \.self) { index in
                        let sti = result.stis[index]
                        VStack(spacing: 4) {
                            if let image = stiImage(sti.data) {
                                Image(uiImage: image)
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .background(.black)
                            }
                            Text("\(decimals(result.stations[safe: index]?.stationM ?? 0, 1)) m")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(12)
            }
            .navigationTitle("Space-time images")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    /// Render an STI as a greyscale image: space across, time down, as Fig. 4.
    private func stiImage(_ matrix: Matrix) -> UIImage? {
        let bytes = displayBytes(matrix)
        guard !bytes.isEmpty else { return nil }
        let provider = CGDataProvider(data: Data(bytes) as CFData)
        guard let provider,
              let cgImage = CGImage(
                width: matrix.cols, height: matrix.rows,
                bitsPerComponent: 8, bitsPerPixel: 8,
                bytesPerRow: matrix.cols,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [URL]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
