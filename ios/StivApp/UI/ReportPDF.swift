//  ReportPDF.swift
//  report.pdf: the run as one document, to keep, print or hand in.
//
//  The phone's counterpart of `stiv/pdf_report.py`, in the same order:
//
//  1. Summary -- the headline velocities, what was measured and how, every
//     warning, and how to read the figures.
//  2. Figures -- the flow map, the velocity profile with its coherency, and the
//     space-time images: the results screen's own views, drawn as vectors.
//  3. Stations -- the velocity table.
//  4. Configuration -- `survey.yaml` in full.
//
//  A4 pages, drawn with `UIGraphicsPDFRenderer`; the SwiftUI figures go in
//  through `ImageRenderer`, which draws vector paths and text into a PDF
//  context rather than a bitmap.

import StivKit
import SwiftUI
import UIKit

/// Fixed colours: a PDF is paper, whatever the phone's appearance setting --
/// the system's adaptive colours would print white text on a white page in
/// dark mode.
private enum Ink {
    static let primary = UIColor(white: 0.07, alpha: 1)
    static let secondary = UIColor(white: 0.36, alpha: 1)
    static let rule = UIColor(white: 0.80, alpha: 1)
    static let tile = UIColor(red: 0.945, green: 0.941, blue: 0.925, alpha: 1)
    static let failed = UIColor(red: 0.82, green: 0.23, blue: 0.23, alpha: 1)
}

enum ReportPDF {
    /// A4, in points.
    static let page = CGRect(x: 0, y: 0, width: 595.28, height: 841.89)
    static let margin: CGFloat = 54

    /// The report for `result`.  `configYAML` is `survey.yaml`, appended in
    /// full when given.  Main thread: the figures are SwiftUI views.
    static func make(result: StivResult, configYAML: String?, created: Date = Date()) -> Data {
        MainActor.assumeIsolated {
            let name = result.survey.section.name
            let format = UIGraphicsPDFRendererFormat()
            format.documentInfo = [
                kCGPDFContextTitle as String: "STIV report - \(name)",
                kCGPDFContextSubject as String: "River surface velocity by space-time image velocimetry",
                kCGPDFContextCreator as String: "STIV for iPhone \(appVersion)",
            ]
            let footer = "STIV report  -  \(name)  -  STIV for iPhone \(appVersion)"
            return UIGraphicsPDFRenderer(bounds: page, format: format).pdfData { context in
                let pages = Pages(context: context, footer: footer)
                summary(pages, result: result, created: created)
                figures(pages, result: result)
                stations(pages, result: result)
                if let configYAML { configuration(pages, yaml: configYAML) }
            }
        }
    }

    static var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
    }

    // MARK: - the sections

    private static func summary(_ pages: Pages, result: StivResult, created: Date) {
        pages.begin("Surface velocity report")
        let stamp = DateFormatter()
        stamp.dateStyle = .long
        stamp.timeStyle = .short
        pages.text("\(result.survey.section.name)   -   \(stamp.string(from: created))",
                   font: .systemFont(ofSize: 9), colour: Ink.secondary)
        pages.gap(8)

        let finite = result.velocities.filter(\.isFinite)
        var tiles: [(String, String, String)] = []
        if finite.isEmpty {
            tiles.append(("Surface velocity", "none", "no station passed quality control"))
        } else {
            let mean = finite.reduce(0, +) / Double(finite.count)
            tiles.append(("Mean surface velocity", "\(decimals(mean, 2)) m/s",
                          "over \(finite.count) stations"))
            tiles.append(("Max surface velocity", "\(decimals(finite.max() ?? 0, 2)) m/s", ""))
        }
        tiles.append(("Stations passing QC",
                      "\(result.stations.filter { !$0.failed }.count) / \(result.stations.count)", ""))
        pages.tiles(tiles)

        pages.heading("The survey")
        let s = result.survey
        pages.pair("Recorded", "\(decimals(result.durationSeconds, 1)) s, \(result.rowCount) rows "
                   + "(St = \(decimals(result.stSecondsPerRow * 1000, 1)) ms/row, "
                   + "frame step \(s.capture.frameStep))")
        pages.pair("Scale", result.rectifierDescription)
        pages.pair("Camera height", "\(decimals(result.cameraHeightM, 2)) m above the water")
        pages.pair("Cross-section", "\(decimals(result.widthM, 2)) m wide, \(result.stations.count) "
                   + "stations, \(decimals(s.sti.searchLineLengthM, 1)) m search lines")
        pages.pair("Expected velocity", "\(decimals(s.sti.velocityRangeMs.lowerBound, 1)) to "
                   + "\(decimals(s.sti.velocityRangeMs.upperBound, 1)) m/s")
        pages.pair("Flow direction", "\(decimals(result.flowBearingDegrees, 1)) deg in the world frame")
        pages.pair("Camera motion", result.motion.summary())
        pages.pair("Software", "STIV for iPhone \(appVersion); analysis took "
                   + "\(decimals(result.elapsedSeconds, 1)) s")

        pages.heading(result.warnings.isEmpty ? "Warnings: none" : "Warnings")
        for warning in result.warnings {
            pages.bullet(warning)
        }

        pages.heading("Run report")
        // The summary less its warnings, which are listed above.
        let report = result.summary().components(separatedBy: "\n\nwarning:")[0]
        for line in report.split(separator: "\n", omittingEmptySubsequences: false) {
            pages.text(String(line), font: .monospacedSystemFont(ofSize: 7.5, weight: .regular),
                       colour: Ink.secondary)
        }

        pages.heading("Reading the figures")
        for line in [
            "Flow map: seen from above, true to scale, with the near bank at the bottom. Each arrow "
                + "is a station, its length proportional to the surface velocity; the key gives the "
                + "scale. Hollow red arrows failed quality control and were interpolated; a cross is "
                + "a station with no velocity.",
            "Velocity profile: the surface velocity across the section with its ±1σ band, and the "
                + "coherency of each station below -- 1 for a clean streak pattern, 0 for none.",
            "Space-time images: the streaks the velocities were measured from. Their angle is the "
                + "speed; no streaks, no measurement.",
        ] {
            pages.bullet(line, colour: Ink.secondary)
        }
    }

    private static func figures(_ pages: Pages, result: StivResult) {
        let width = page.width - 2 * margin

        pages.begin("Flow map")
        pages.figure(FlowMapView(map: FlowMap(result: result)), height: width * 0.8)

        pages.begin("Surface velocity across the section")
        pages.figure(VelocityProfileChart(result: result).padding(4), height: 300)
        pages.gap(10)
        pages.figure(CoherencyChart(result: result).padding(4), height: 130)

        let previews = result.stiPreviews
        guard !previews.isEmpty else { return }
        pages.begin("Space-time images")
        pages.text("Space across, time down. Each is labelled with its station and velocity.",
                   font: .systemFont(ofSize: 9), colour: Ink.secondary)
        pages.gap(6)
        // Up to 24, evenly spread across the section, as `sti_panel.png`.
        let count = min(previews.count, 24)
        let picks = (0..<count).map {
            Int((Double($0) * Double(previews.count - 1) / Double(max(count - 1, 1))).rounded())
        }
        let columns = 6
        let gap: CGFloat = 8
        let cell = (width - gap * CGFloat(columns - 1)) / CGFloat(columns)
        let height = picks.map { index -> CGFloat in
            let p = previews[index]
            return min(cell * 1.4, cell * CGFloat(p.rows) / CGFloat(max(p.cols, 1)))
        }.max() ?? cell
        for rowStart in stride(from: 0, to: picks.count, by: columns) {
            pages.ensureRoom(height + 30)
            for (column, index) in picks[rowStart..<min(rowStart + columns, picks.count)].enumerated() {
                let preview = previews[index]
                let station = result.stations.first { $0.index == preview.stationIndex }
                let x = margin + CGFloat(column) * (cell + gap)
                let h = min(cell * 1.4, cell * CGFloat(preview.rows) / CGFloat(max(preview.cols, 1)))
                stiImage(preview)?.draw(in: CGRect(x: x, y: pages.y, width: cell, height: h))
                let label = station.map { st in
                    "\(decimals(st.stationM, 1)) m\n"
                        + (st.velocityMs.isFinite ? "\(decimals(st.velocityMs, 2)) m/s" : "no value")
                } ?? ""
                (label as NSString).draw(
                    in: CGRect(x: x, y: pages.y + h + 2, width: cell, height: 22),
                    withAttributes: [.font: UIFont.monospacedDigitSystemFont(ofSize: 7, weight: .regular),
                                     .foregroundColor: (station?.failed ?? false) ? Ink.failed : Ink.primary])
            }
            pages.gap(height + 30)
        }
    }

    private static func stations(_ pages: Pages, result: StivResult) {
        pages.begin("Velocity by station")
        pages.text("Distance is measured along the cross-section from its start. ± is the station's "
                   + "1σ spread, its range the distance from the phone. Flags explain a station "
                   + "that failed or was changed by quality control.",
                   font: .systemFont(ofSize: 9), colour: Ink.secondary)
        pages.gap(6)
        let mono = UIFont.monospacedSystemFont(ofSize: 8, weight: .regular)
        func column(_ s: String, _ width: Int) -> String {
            String(repeating: " ", count: max(0, width - s.count)) + s
        }
        func number(_ x: Double, _ places: Int) -> String { x.isFinite ? decimals(x, places) : "-" }
        pages.text(column("station", 8) + column("velocity", 10) + column("±", 8)
                   + column("coherency", 10) + column("tan phi", 9) + column("range", 8) + "  flags",
                   font: .monospacedSystemFont(ofSize: 8, weight: .bold))
        pages.text(column("(m)", 8) + column("(m/s)", 10) + column("(m/s)", 8)
                   + String(repeating: " ", count: 19) + column("(m)", 8),
                   font: mono, colour: Ink.secondary)
        for st in result.stations {
            pages.text(column(decimals(st.stationM, 2), 8)
                       + column(number(st.velocityMs, 3), 10)
                       + column(number(st.velocityStdMs, 3), 8)
                       + column(number(st.coherency, 2), 10)
                       + column(number(st.tanPhi, 2), 9)
                       + column(number(st.rangeM, 1), 8)
                       + "  " + st.flags.map(\.pythonName).joined(separator: ", "),
                       font: mono, colour: st.failed ? Ink.failed : Ink.primary)
        }
    }

    private static func configuration(_ pages: Pages, yaml: String) {
        pages.begin("Configuration")
        pages.text("The survey as survey.yaml: the settings this run used, the section as ground "
                   + "control points, and the scale. Fill in video.path with footage of the same "
                   + "section and `stiv run survey.yaml` repeats the analysis on a computer.",
                   font: .systemFont(ofSize: 9), colour: Ink.secondary)
        pages.gap(6)
        for line in yaml.split(separator: "\n", omittingEmptySubsequences: false) {
            pages.text(String(line), font: .monospacedSystemFont(ofSize: 7.5, weight: .regular))
        }
    }
}

// MARK: - laying out pages

/// Flowing content down A4 pages, starting a new one whenever it runs out.
private final class Pages {
    let context: UIGraphicsPDFRendererContext
    let footer: String
    var y: CGFloat = 0
    private var title = ""
    private var pageOfTitle = 0

    private let left = ReportPDF.margin
    private var width: CGFloat { ReportPDF.page.width - 2 * ReportPDF.margin }
    private var bottom: CGFloat { ReportPDF.page.height - ReportPDF.margin }

    init(context: UIGraphicsPDFRendererContext, footer: String) {
        self.context = context
        self.footer = footer
    }

    /// A new page under `title`; one that runs over says "(continued)".
    func begin(_ title: String) {
        self.title = title
        pageOfTitle = 0
        newPage()
    }

    private func newPage() {
        context.beginPage()
        pageOfTitle += 1
        y = ReportPDF.margin
        let heading = pageOfTitle == 1 ? title : "\(title) (continued)"
        draw(heading, font: .systemFont(ofSize: 15, weight: .bold), colour: Ink.primary)
        y += 4
        let rule = UIBezierPath()
        rule.move(to: CGPoint(x: left, y: y))
        rule.addLine(to: CGPoint(x: left + width, y: y))
        Ink.rule.setStroke()
        rule.lineWidth = 0.6
        rule.stroke()
        y += 10
        (footer as NSString).draw(
            at: CGPoint(x: left, y: ReportPDF.page.height - ReportPDF.margin * 0.6),
            withAttributes: [.font: UIFont.systemFont(ofSize: 7), .foregroundColor: Ink.secondary])
    }

    func ensureRoom(_ height: CGFloat) {
        if y + height > bottom { newPage() }
    }

    private func height(of text: String, font: UIFont, width: CGFloat) -> CGFloat {
        ceil((text as NSString).boundingRect(
            with: CGSize(width: width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: font], context: nil).height)
    }

    private func draw(_ text: String, font: UIFont, colour: UIColor, indent: CGFloat = 0) {
        let w = width - indent
        let h = height(of: text.isEmpty ? " " : text, font: font, width: w)
        ensureRoom(h)
        (text as NSString).draw(
            with: CGRect(x: left + indent, y: y, width: w, height: h),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: font, .foregroundColor: colour], context: nil)
        y += h
    }

    func text(_ text: String, font: UIFont = .systemFont(ofSize: 9.5), colour: UIColor = Ink.primary,
              indent: CGFloat = 0) {
        draw(text, font: font, colour: colour, indent: indent)
    }

    func bullet(_ text: String, colour: UIColor = Ink.primary) {
        let font = UIFont.systemFont(ofSize: 9.5)
        ensureRoom(height(of: text, font: font, width: width - 12))
        ("•" as NSString).draw(at: CGPoint(x: left, y: y),
                               withAttributes: [.font: font, .foregroundColor: colour])
        draw(text, font: font, colour: colour, indent: 12)
        y += 3
    }

    func heading(_ text: String) {
        ensureRoom(40)
        y += 12
        draw(text, font: .systemFont(ofSize: 11.5, weight: .bold), colour: Ink.primary)
        y += 3
    }

    func gap(_ points: CGFloat) { y += points }

    /// A label and its value, the value wrapped in a column of its own.
    func pair(_ label: String, _ value: String) {
        let font = UIFont.systemFont(ofSize: 9.5)
        let column: CGFloat = 120
        let h = height(of: value, font: font, width: width - column)
        ensureRoom(h)
        (label as NSString).draw(at: CGPoint(x: left, y: y),
                                 withAttributes: [.font: font, .foregroundColor: Ink.secondary])
        (value as NSString).draw(
            with: CGRect(x: left + column, y: y, width: width - column, height: h),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: font, .foregroundColor: Ink.primary], context: nil)
        y += h + 1
    }

    /// Headline numbers in a row of boxes.
    func tiles(_ items: [(label: String, value: String, sub: String)]) {
        let height: CGFloat = 58, gap: CGFloat = 8
        ensureRoom(height + 8)
        let w = (width - gap * CGFloat(items.count - 1)) / CGFloat(max(items.count, 1))
        for (k, item) in items.enumerated() {
            let box = CGRect(x: left + CGFloat(k) * (w + gap), y: y, width: w, height: height)
            let path = UIBezierPath(roundedRect: box, cornerRadius: 6)
            Ink.tile.setFill()
            path.fill()
            Ink.rule.setStroke()
            path.lineWidth = 0.5
            path.stroke()
            (item.label.uppercased() as NSString).draw(
                at: CGPoint(x: box.minX + 9, y: box.minY + 8),
                withAttributes: [.font: UIFont.systemFont(ofSize: 6.5, weight: .medium),
                                 .foregroundColor: Ink.secondary])
            (item.value as NSString).draw(
                at: CGPoint(x: box.minX + 9, y: box.minY + 20),
                withAttributes: [.font: UIFont.systemFont(ofSize: 17, weight: .bold),
                                 .foregroundColor: Ink.primary])
            (item.sub as NSString).draw(
                at: CGPoint(x: box.minX + 9, y: box.minY + 42),
                withAttributes: [.font: UIFont.systemFont(ofSize: 6.5),
                                 .foregroundColor: Ink.secondary])
        }
        y += height + 8
    }

    /// A SwiftUI view, the full text width and `height` tall, as vectors.
    ///
    /// `ImageRenderer` draws in Core Graphics' own orientation, origin at the
    /// bottom left; the page's context is UIKit's, origin at the top left, so
    /// the box is flipped to meet it.
    func figure<V: View>(_ view: V, height: CGFloat) {
        ensureRoom(height)
        let box = CGRect(x: left, y: y, width: width, height: height)
        let cg = context.cgContext
        let content = view.frame(width: box.width, height: box.height)
            .environment(\.colorScheme, .light)
        // The report is built on the main thread (see `ReportPDF.make`), where
        // SwiftUI rendering has to happen.
        MainActor.assumeIsolated {
            ImageRenderer(content: content).render { _, render in
                cg.saveGState()
                cg.translateBy(x: box.minX, y: box.maxY)
                cg.scaleBy(x: 1, y: -1)
                render(cg)
                cg.restoreGState()
            }
        }
        y += height
    }
}
