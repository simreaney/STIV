//  FrameBufferTests.swift
//  Reading ARKit's pixel buffers: the luma plane the STI is sampled from, and
//  the depth map the water level is measured from.
//
//  Both are read through raw pointers, and both have a stride that is not the
//  width.  Getting that wrong does not crash -- it reads the neighbouring row's
//  pixels, or padding -- so it is tested with buffers whose padding holds a
//  value no correct read can return.

import CoreVideo
import XCTest
@testable import STIV

final class LumaPlaneTests: XCTestCase {
    /// 101 px wide, so the 64-byte row alignment pads every row to 128.
    private let width = 101
    private let height = 37

    /// A ramp small enough never to wrap, so bilinear interpolation of it is
    /// exact and any sample can be checked against the formula.
    private func ramp(_ u: Int, _ v: Int) -> UInt8 { UInt8(u + 2 * v) }

    private func rampBuffer() -> CVPixelBuffer {
        let buffer = makeYCbCrBuffer(width: width, height: height)
        writeLuma(buffer, padding: 255) { ramp($0, $1) }
        return buffer
    }

    func testTheStrideIsHonouredRatherThanAssumedToBeTheWidth() throws {
        let buffer = rampBuffer()
        XCTAssertGreaterThan(CVPixelBufferGetBytesPerRowOfPlane(buffer, 0), width,
                             "test is vacuous: the buffer was not padded")

        let checked = try XCTUnwrap(withLumaPlane(of: buffer) { luma -> Int in
            XCTAssertEqual(luma.width, width)
            XCTAssertEqual(luma.height, height)
            var n = 0
            for v in 0..<height {
                for u in 0..<width {
                    XCTAssertEqual(luma.sample(u: Double(u), v: Double(v)),
                                   Float(ramp(u, v)), "pixel (\(u), \(v))")
                    n += 1
                }
            }
            return n
        })
        XCTAssertEqual(checked, width * height)
    }

    func testSamplingBetweenPixelsIsBilinear() throws {
        let buffer = rampBuffer()
        try XCTUnwrap(withLumaPlane(of: buffer) { luma in
            // u + 2v is linear, so bilinear interpolation reproduces it exactly.
            for (u, v) in [(10.25, 5.5), (0.5, 0.5), (99.75, 35.1), (50.0, 17.9)] {
                let sample = luma.sample(u: u, v: v).map(Double.init) ?? .nan
                XCTAssertEqual(sample, u + 2 * v, accuracy: 1e-3, "(\(u), \(v))")
            }
        })
    }

    /// Off the edge must read as *missing*, not as black: the STI fills a
    /// missing sample from its column's time-mean, but a zero would be taken
    /// as a genuinely dark pixel and draw a streak.
    func testSamplesOffTheEdgeAreMissingNotDark() throws {
        let buffer = rampBuffer()
        try XCTUnwrap(withLumaPlane(of: buffer) { luma in
            XCTAssertNil(luma.sample(u: -0.01, v: 3))
            XCTAssertNil(luma.sample(u: 3, v: -0.01))
            XCTAssertNil(luma.sample(u: Double(width - 1) + 0.01, v: 3))
            XCTAssertNil(luma.sample(u: 3, v: Double(height - 1) + 0.01))
            // The last pixel itself is still inside.
            XCTAssertEqual(luma.sample(u: Double(width - 1), v: Double(height - 1)),
                           Float(ramp(width - 1, height - 1)))
        })
    }

    func testPatchVarianceSeparatesTextureFromFlatGround() throws {
        let flat = makeYCbCrBuffer(width: 64, height: 64)
        writeLuma(flat) { _, _ in 90 }
        let checkerboard = makeYCbCrBuffer(width: 64, height: 64)
        writeLuma(checkerboard) { u, v in (u + v) % 2 == 0 ? 0 : 100 }

        let flatVariance = try XCTUnwrap(withLumaPlane(of: flat) {
            $0.patchVariance(centreU: 32, centreV: 32, half: 10)
        })
        let texturedVariance = try XCTUnwrap(withLumaPlane(of: checkerboard) {
            $0.patchVariance(centreU: 32, centreV: 32, half: 10)
        })
        XCTAssertEqual(flatVariance, 0, accuracy: 1e-9)
        // Half 0, half 100: variance 50^2, less a hair for the odd pixel count.
        XCTAssertEqual(texturedVariance, 2500, accuracy: 10)
    }
}

final class DepthPlaneTests: XCTestCase {
    func testOnlyHighConfidenceFinitePositiveDepthsAreAccepted() throws {
        // Columns: 0 low, 1 medium, 2 high confidence.  Rows vary the depth.
        let depths: [Float] = [1.5, .nan, 0, -1, .infinity, 2.25]
        let (depth, confidence) = makeDepthBuffers(width: 3, height: depths.count) { x, y in
            (depths[y], UInt8(x))
        }
        try XCTUnwrap(withDepthPlane(depthMap: depth, confidenceMap: confidence) { plane in
            XCTAssertEqual(plane.width, 3)
            XCTAssertEqual(plane.height, depths.count)
            for y in depths.indices {
                XCTAssertNil(plane.highConfidenceDepth(x: 0, y: y), "low confidence, row \(y)")
                XCTAssertNil(plane.highConfidenceDepth(x: 1, y: y), "medium confidence, row \(y)")
            }
            XCTAssertEqual(plane.highConfidenceDepth(x: 2, y: 0), 1.5)
            XCTAssertNil(plane.highConfidenceDepth(x: 2, y: 1), "NaN")
            XCTAssertNil(plane.highConfidenceDepth(x: 2, y: 2), "zero")
            XCTAssertNil(plane.highConfidenceDepth(x: 2, y: 3), "negative")
            XCTAssertNil(plane.highConfidenceDepth(x: 2, y: 4), "infinite")
            XCTAssertEqual(plane.highConfidenceDepth(x: 2, y: 5), 2.25)
            XCTAssertNil(plane.highConfidenceDepth(x: 3, y: 0), "off the right edge")
            XCTAssertNil(plane.highConfidenceDepth(x: 2, y: depths.count), "off the bottom")
        })
    }

    /// Each row is 8 floats -- 32 bytes -- padded to 64.  A reader that stepped
    /// by `width * 4` would read row 1 from the middle of row 0's padding.
    func testRowsAreAddressedByStride() throws {
        let (depth, confidence) = makeDepthBuffers(width: 8, height: 6) { x, y in
            (Float(1 + y) + Float(x) / 10, 2)
        }
        XCTAssertGreaterThan(CVPixelBufferGetBytesPerRow(depth), 8 * 4,
                             "test is vacuous: the depth buffer was not padded")
        try XCTUnwrap(withDepthPlane(depthMap: depth, confidenceMap: confidence) { plane in
            for y in 0..<6 {
                for x in 0..<8 {
                    XCTAssertEqual(plane.highConfidenceDepth(x: x, y: y) ?? .nan,
                                   Float(1 + y) + Float(x) / 10, accuracy: 1e-6)
                }
            }
        })
    }

    func testWithoutAConfidenceMapEveryFiniteDepthIsTrusted() throws {
        let (depth, _) = makeDepthBuffers(width: 4, height: 4) { _, _ in (3.0, 0) }
        let value = withDepthPlane(depthMap: depth, confidenceMap: nil) {
            $0.highConfidenceDepth(x: 1, y: 1)
        }
        XCTAssertEqual(value, 3.0)
    }

    /// Apple documents neither the depth map's format nor its size.  If it ever
    /// stops being Float32, reading it as Float32 would be garbage: refuse.
    func testADepthMapInAnUnexpectedFormatIsRefused() {
        let (_, confidence) = makeDepthBuffers(width: 4, height: 4) { _, _ in (1, 2) }
        XCTAssertNil(withDepthPlane(depthMap: confidence, confidenceMap: nil) { _ in true })
    }
}
