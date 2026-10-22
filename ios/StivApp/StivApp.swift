//  StivApp.swift
//  Space-Time Image Velocimetry on an iPhone.
//
//  The desktop package in this repository needs a drone flight, a measured
//  altitude or four surveyed ground control points, and a laptop.  Everything
//  the method actually requires is in one hand: a camera for g(x, t), a
//  gravity-aligned metric pose, and a LiDAR range to fix the water level.
//
//  Walk to the bank, measure the water level, tap the two ends of the
//  cross-section, hold still for thirty seconds, read the velocity profile.

import SwiftUI

@main
struct StivApp: App {
    @StateObject private var model = SurveyModel()

    var body: some Scene {
        WindowGroup {
            SurveyFlowView()
                .environmentObject(model)
                // The capture session is published separately: SwiftUI does not
                // propagate changes through a nested ObservableObject, so a view
                // that only held `model` would never see a frame arrive.
                .environmentObject(model.capture)
                // A river surface is what is being measured; a dark UI keeps the
                // screen readable in the sun and out of the way of the feed.
                .preferredColorScheme(.dark)
        }
    }
}

/// One screen per decision, in the order each one becomes answerable.
struct SurveyFlowView: View {
    @EnvironmentObject private var model: SurveyModel
    @EnvironmentObject private var capture: ARCaptureSession

    var body: some View {
        ZStack {
            switch model.step {
            case .welcome:    WelcomeScreen()
            case .waterPlane: WaterPlaneScreen()
            case .section:    SectionScreen()
            case .settings:   SettingsScreen()
            case .record:     RecordScreen()
            case .results:    ResultsScreen()
            }

            // A camera that will not start -- refused permission, a sensor
            // fault -- otherwise looks like a black feed asking the user to
            // move the phone slowly.  Say what is wrong, and offer the way out.
            if model.step.usesCamera, let failure = capture.failure {
                Color.black.opacity(0.55).ignoresSafeArea()
                CaptureFailureCard(failure: failure)
            }
        }
        .animation(.default, value: model.step)
    }
}

// MARK: - shared chrome

/// A dark card over the camera feed, for prompts and controls.
struct OverlayCard<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(14)
            .background(.black.opacity(0.72), in: RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14)
                .strokeBorder(.white.opacity(0.12), lineWidth: 1))
    }
}

/// The tracking-state banner.
///
/// Shown wherever the session is live, because every number this app produces
/// depends on world tracking being healthy, and the failure is otherwise silent:
/// a drifting pose still yields a confident velocity.
struct TrackingBanner: View {
    let advice: String?

    var body: some View {
        if let advice {
            Label(advice, systemImage: "exclamationmark.triangle.fill")
                .font(.footnote)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.orange.opacity(0.9), in: Capsule())
                .foregroundStyle(.black)
        }
    }
}

struct PrimaryButton: View {
    let title: String
    var systemImage: String?
    var enabled = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                if let systemImage { Image(systemName: systemImage) }
                Text(title).fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
        }
        .background(enabled ? Color.accentColor : Color.gray.opacity(0.4),
                    in: RoundedRectangle(cornerRadius: 12))
        .foregroundStyle(.white)
        .disabled(!enabled)
    }
}
