//  ViewportTests.swift
//  The screen the user taps and the image the STI samples.
//
//  The camera feed is drawn edge to edge, under the status bar and the Dynamic
//  Island.  Everything drawn over it, every tap read through it, and the size
//  handed to ARKit to project into it must use that same full-screen space.
//  A view that respects the safe area is 93 points shorter on an iPhone 17
//  Pro, and an overlay projected into it lands tens of points from the feed --
//  with taps hitting the wrong patch of river to match.
//
//  The overlay itself projects through `ARCamera`, which cannot be built
//  outside ARKit; `OnDeviceTests` checks that on a phone.  What is checked here
//  is the layout that feeds it, and the app's own aspect-fill mapping.

import StivKit
import SwiftUI
import UIKit
import XCTest
@testable import STIV

final class ViewportTests: XCTestCase {
    private let image = CGSize(width: 1920, height: 1440)
    private let portrait = CGSize(width: 393, height: 852)
    private let landscape = CGSize(width: 852, height: 393)

    // MARK: - the layout

    /// Host a `FullScreenLayer` in a real window and report the size it hands
    /// its content and where that content sits on screen.
    private func hostFullScreenLayer() throws -> (window: UIWindow, size: CGSize?, frame: CGRect?) {
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }.first, "the test host has no window scene")
        let window = UIWindow(windowScene: scene)
        var size: CGSize?
        var frame: CGRect?
        let probe = FullScreenLayer { layerSize in
            GeometryReader { geometry in
                Color.clear.onAppear {
                    size = layerSize
                    frame = geometry.frame(in: .global)
                }
            }
        }
        window.rootViewController = UIHostingController(rootView: probe)
        window.makeKeyAndVisible()
        let laidOut = expectation(for: NSPredicate { _, _ in frame != nil }, evaluatedWith: nil)
        wait(for: [laidOut], timeout: 5)
        return (window, size, frame)
    }

    func testTheFeedLayerIsTheWholeScreenNotTheSafeArea() throws {
        let (window, size, frame) = try hostFullScreenLayer()
        defer { window.isHidden = true }
        XCTAssertGreaterThan(window.safeAreaInsets.top + window.safeAreaInsets.bottom, 0,
                             "test is vacuous: this simulator has no safe-area insets")
        XCTAssertEqual(size, window.bounds.size,
                       "the projector would be given the safe area, not the feed")
        XCTAssertEqual(frame, window.bounds,
                       "taps would be read in a space offset from the feed's")
    }

    // MARK: - the aspect-fill mapping

    func testImageAndViewportRoundTripAcrossTheWholeScreen() {
        for (orientation, size) in [(UIInterfaceOrientation.portrait, portrait),
                                    (.landscapeRight, landscape), (.landscapeLeft, landscape),
                                    (.portraitUpsideDown, portrait)] {
            let geometry = ViewportGeometry(imageSize: image, viewportSize: size,
                                            orientation: orientation)
            for q in [CGPoint(x: 0, y: 0), CGPoint(x: size.width, y: size.height),
                      CGPoint(x: 37.5, y: 611.25), CGPoint(x: size.width / 2, y: 12)] {
                let back = geometry.viewportPoint(image: geometry.imagePoint(viewport: q))
                XCTAssertEqual(back.x, q.x, accuracy: 1e-9, "\(orientation) \(q)")
                XCTAssertEqual(back.y, q.y, accuracy: 1e-9, "\(orientation) \(q)")
            }
            // The image centre is the screen centre.
            let centre = geometry.viewportPoint(image: PixelPoint(u: 960, v: 720))
            XCTAssertEqual(centre.x, size.width / 2, accuracy: 1e-9)
            XCTAssertEqual(centre.y, size.height / 2, accuracy: 1e-9)
        }
    }

    /// Which way each orientation turns the image.  A round trip cannot see
    /// a rotation the wrong way round -- it undoes its own mistake -- so each
    /// direction is checked outright.
    ///
    /// The sensor's native orientation is landscape-right, shown unturned.
    /// Portrait is a quarter turn clockwise from there, so the image's `u`
    /// axis runs down the screen and its `v` axis to the left; landscape-left
    /// is a half turn; upside-down portrait the remaining quarter.  On a phone,
    /// `OnDeviceTests.testTheReticlesViewportMappingAgreesWithARKit` holds
    /// this mapping to ARKit's own.
    func testEachOrientationTurnsTheImageTheWayARKitDoes() {
        let cases: [(UIInterfaceOrientation, CGSize, u: CGVector, v: CGVector, origin: (left: Bool, top: Bool))] = [
            (.landscapeRight, landscape, CGVector(dx: 1, dy: 0), CGVector(dx: 0, dy: 1), (true, true)),
            (.portrait, portrait, CGVector(dx: 0, dy: 1), CGVector(dx: -1, dy: 0), (false, true)),
            (.landscapeLeft, landscape, CGVector(dx: -1, dy: 0), CGVector(dx: 0, dy: -1), (false, false)),
            (.portraitUpsideDown, portrait, CGVector(dx: 0, dy: -1), CGVector(dx: 1, dy: 0), (true, false)),
        ]
        for (orientation, size, u, v, origin) in cases {
            let geometry = ViewportGeometry(imageSize: image, viewportSize: size,
                                            orientation: orientation)
            let centre = geometry.viewportPoint(image: PixelPoint(u: 960, v: 720))
            let alongU = geometry.viewportPoint(image: PixelPoint(u: 1060, v: 720))
            let alongV = geometry.viewportPoint(image: PixelPoint(u: 960, v: 820))
            func direction(_ p: CGPoint) -> CGVector {
                let d = CGVector(dx: p.x - centre.x, dy: p.y - centre.y)
                let n = hypot(d.dx, d.dy)
                return CGVector(dx: (d.dx / n).rounded(), dy: (d.dy / n).rounded())
            }
            XCTAssertEqual(direction(alongU), u, "\(orientation): image u runs the wrong way")
            XCTAssertEqual(direction(alongV), v, "\(orientation): image v runs the wrong way")

            // And the image's top-left corner lands in the corner it should.
            let corner = geometry.viewportPoint(image: PixelPoint(u: 0, v: 0))
            XCTAssertEqual(corner.x < size.width / 2, origin.left, "\(orientation)")
            XCTAssertEqual(corner.y < size.height / 2, origin.top, "\(orientation)")
        }
    }

    /// The feed *fills* the screen: the sensor's long side is cropped, so every
    /// corner of the screen shows some part of the image and the image is
    /// magnified by the larger of the two ratios.
    func testTheFeedFillsTheScreenAndCropsTheRest() {
        let geometry = ViewportGeometry(imageSize: image, viewportSize: portrait,
                                        orientation: .portrait)
        XCTAssertEqual(geometry.scale, 852.0 / 1920.0, accuracy: 1e-12)
        for corner in [CGPoint(x: 0, y: 0), CGPoint(x: 393, y: 852)] {
            let p = geometry.imagePoint(viewport: corner)
            XCTAssertTrue((0...1920).contains(p.u) && (0...1440).contains(p.v),
                          "screen corner \(corner) shows no image: \(p)")
        }
        // In portrait the image's u axis runs down the screen and is shown in
        // full; its v axis runs across and is cropped symmetrically.
        let top = geometry.imagePoint(viewport: CGPoint(x: 196.5, y: 0))
        let left = geometry.imagePoint(viewport: CGPoint(x: 0, y: 426))
        let right = geometry.imagePoint(viewport: CGPoint(x: 393, y: 426))
        XCTAssertEqual(top.u, 0, accuracy: 1e-9)
        XCTAssertEqual(left.v + right.v, 1440, accuracy: 1e-9, "the crop is not centred")
        XCTAssertGreaterThan(abs(left.v - right.v), 0)
        XCTAssertLessThan(abs(left.v - right.v), 1440, "portrait must crop the image's v axis")
    }

    /// What the bug looked like: a point projected for the safe area's
    /// viewport but drawn on the full-screen canvas, against where the feed
    /// shows it.
    func testASafeAreaViewportMisplacesTheOverlay() {
        let safeArea = CGSize(width: 393, height: 852 - 59 - 34)
        let full = ViewportGeometry(imageSize: image, viewportSize: portrait,
                                    orientation: .portrait)
        let wrong = ViewportGeometry(imageSize: image, viewportSize: safeArea,
                                     orientation: .portrait)
        let point = PixelPoint(u: 1200, v: 700)
        let shown = full.viewportPoint(image: point)
        let drawn = wrong.viewportPoint(image: point)
        XCTAssertGreaterThan(hypot(shown.x - drawn.x, shown.y - drawn.y), 30,
                             "a safe-area viewport would be close enough not to matter")
    }
}