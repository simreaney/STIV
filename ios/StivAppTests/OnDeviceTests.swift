//  OnDeviceTests.swift
//  The checks that need a real ARKit session, and so a real iPhone.
//
//  Everything else in this target runs in the simulator by standing in for
//  ARKit.  These are the places where standing in proves nothing, because the
//  question *is* what ARKit does: whether its projection follows the same axis
//  conventions as StivKit's, and whether its buffers arrive in the formats the
//  readers assume.  They skip in the simulator, which has no world tracking.
//
//  Run them with `ios/tools/test.sh device`.  The first run asks for camera
//  access; after that, point the phone at something with texture -- a desk,
//  a floor -- and move it slowly so tracking can start.  The LiDAR test also
//  needs a surface within a couple of metres.

import ARKit
import XCTest
@testable import STIV

final class OnDeviceTests: XCTestCase {
    private var session: ARSession?

    override func setUpWithError() throws {
        try XCTSkipUnless(ARWorldTrackingConfiguration.isSupported,
                          "needs a real iPhone: the simulator has no world tracking")
    }

    override func tearDown() {
        session?.pause()
        session = nil
    }

    /// Start a session configured as the app configures it, and wait for a
    /// frame that satisfies `condition`.
    private func frame(
        semantics: ARConfiguration.FrameSemantics = [],
        timeout: TimeInterval = 45,
        what: String,
        where condition: @escaping (ARFrame) -> Bool
    ) throws -> ARFrame {
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        configuration.frameSemantics = semantics
        let session = ARSession()
        self.session = session
        session.run(configuration)

        let arrived = expectation(for: NSPredicate { _, _ in
            session.currentFrame.map(condition) ?? false
        }, evaluatedWith: nil)
        let outcome = XCTWaiter.wait(for: [arrived], timeout: timeout)
        guard outcome == .completed, let frame = session.currentFrame else {
            throw XCTSkip("no \(what) within \(Int(timeout)) s -- move the phone slowly over "
                          + "a textured surface and run again")
        }
        return frame
    }

    /// The one check the unit tests cannot make: StivKit's hand-written
    /// world-to-image projection against ARKit's own.  A sign or a transpose
    /// wrong here mirrors every measurement while every other test passes.
    func testTheManualProjectionAgreesWithARKit() throws {
        let frame = try frame(what: "normal tracking") { $0.camera.trackingState.isNormal }
        let worst = verifyProjectionParity(frame: frame)
        XCTAssertFalse(worst.isNaN, "no test point landed in the frame")
        XCTAssertLessThan(worst, 1.0, "projection disagrees with ARKit by \(worst) px")
    }

    /// The depth reticle is drawn through the app's own aspect-fill mapping
    /// rather than through ARKit, because it maps a region of the *image*, not
    /// a point in the world.  This pins that mapping to ARKit's viewport
    /// projection, in each orientation the app supports.
    func testTheReticlesViewportMappingAgreesWithARKit() throws {
        let frame = try frame(what: "normal tracking") { $0.camera.trackingState.isNormal }
        let worst = verifyViewportParity(frame: frame)
        XCTAssertFalse(worst.isNaN, "no test point landed in the frame")
        XCTAssertLessThan(worst, 1.0, "viewport mapping disagrees with ARKit by \(worst) pt")
    }

    /// `LumaPlane` reads plane 0 of the captured image as full-range luma.
    func testTheCapturedImageIsTheBiPlanarYCbCrTheReaderExpects() throws {
        let frame = try frame(what: "camera frame") { _ in true }
        let image = frame.capturedImage
        let format = CVPixelBufferGetPixelFormatType(image)
        XCTAssertEqual(CVPixelBufferGetPlaneCount(image), 2)
        XCTAssertTrue([kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                       kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange].contains(format),
                      "unexpected capturedImage format \(format)")
        let size = withLumaPlane(of: image) { ($0.width, $0.height) }
        XCTAssertEqual(size?.0, Int(frame.camera.imageResolution.width))
        XCTAssertEqual(size?.1, Int(frame.camera.imageResolution.height))
    }

    /// Apple documents neither the depth map's format nor its size, and the
    /// back-projection assumes it shares the captured image's aspect ratio.
    func testLidarDepthArrivesInTheFormatTheReaderExpects() throws {
        try XCTSkipUnless(ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth),
                          "no LiDAR on this device")
        let frame = try frame(semantics: .sceneDepth, what: "LiDAR depth") {
            $0.sceneDepth != nil && $0.camera.trackingState.isNormal
        }
        let depth = try XCTUnwrap(frame.sceneDepth)
        let found = withDepthPlane(depthMap: depth.depthMap,
                                   confidenceMap: depth.confidenceMap) { plane -> Int in
            let resolution = frame.camera.imageResolution
            XCTAssertEqual(Double(plane.width) / Double(plane.height),
                           Double(resolution.width / resolution.height), accuracy: 0.01,
                           "depth map and image no longer share an aspect ratio")
            var accepted = 0
            for y in 0..<plane.height {
                for x in 0..<plane.width where plane.highConfidenceDepth(x: x, y: y) != nil {
                    accepted += 1
                }
            }
            return accepted
        }
        let accepted = try XCTUnwrap(found, "depth map is not Float32 any more")
        XCTAssertGreaterThan(accepted, 0, "no high-confidence returns: point at something nearby")
    }
}
