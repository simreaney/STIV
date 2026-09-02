// swift-tools-version:5.9
import PackageDescription

// StivKit is deliberately dependency-free and imports neither ARKit nor UIKit:
// the numerics are the part that must be verifiable off-device and comparable,
// number for number, against the Python package in the repository root.
let package = Package(
    name: "StivKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "StivKit", targets: ["StivKit"]),
    ],
    targets: [
        .target(name: "StivKit"),
        .testTarget(
            name: "StivKitTests",
            dependencies: ["StivKit"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
