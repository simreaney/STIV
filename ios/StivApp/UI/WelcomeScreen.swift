//  WelcomeScreen.swift
//  What this measures, what it needs, and what it cannot do.
//
//  The limitations here are not boilerplate.  STIV reads the motion of a
//  *pattern* on the water surface: a rippled reach measures well and a glassy
//  pool cannot be measured at all, by any implementation, ever.  Better to say
//  so before someone spends a morning on it.

import SwiftUI

struct WelcomeScreen: View {
    @EnvironmentObject private var model: SurveyModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("STIV").font(.largeTitle.bold())
                    Text("River surface velocity from video and LiDAR")
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 14) {
                    Step(number: 1, title: "Measure the water level",
                         detail: "Aim at the waterline within about 5 m. LiDAR fixes the "
                               + "height of the water surface; gravity supplies the rest, "
                               + "which is what carries the scale out to the far bank.")
                    Step(number: 2, title: "Tap the cross-section",
                         detail: "Near bank, far bank, then a point downstream. The width "
                               + "appears in metres straight away.")
                    Step(number: 3, title: "Hold still and record",
                         detail: "Thirty seconds. Camera motion is compensated from the "
                               + "AR pose, but a braced phone is still better than a "
                               + "hand-held one.")
                }

                DisclosureGroup("What this needs to work") {
                    VStack(alignment: .leading, spacing: 10) {
                        Bullet("**Visible texture on the water.** Ripples, boils, foam, "
                             + "floating debris. A mirror-calm pool cannot be measured -- "
                             + "that is a limit of the method, not of this app.")
                        Bullet("**A view of the flow direction.** STIV measures one "
                             + "component, along the search line, so the line must run "
                             + "with the current.")
                        Bullet("**Height helps more than anything.** Scale error at the far "
                             + "bank grows in proportion to range ÷ camera height. A "
                             + "bridge parapet beats a riverbank; standing beats crouching.")
                        Bullet("**Surface velocity is not discharge.** Converting one to "
                             + "the other needs depths and a velocity index, and this "
                             + "version does not do it.")
                    }
                    .padding(.top, 8)
                }
                .font(.callout)

                if model.availability != .available {
                    OverlayCard {
                        Label {
                            Text(model.availability.explanation).font(.footnote)
                        } icon: {
                            Image(systemName: "info.circle.fill")
                        }
                    }
                }

                PrimaryButton(title: "Start", systemImage: "camera.viewfinder",
                              enabled: model.availability != .noWorldTracking) {
                    model.begin()
                }
            }
            .padding(22)
        }
    }

    private struct Step: View {
        let number: Int
        let title: String
        let detail: String

        var body: some View {
            HStack(alignment: .top, spacing: 12) {
                Text("\(number)")
                    .font(.headline)
                    .frame(width: 28, height: 28)
                    .background(Color.accentColor, in: Circle())
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.headline)
                    Text(detail).font(.footnote).foregroundStyle(.secondary)
                }
            }
        }
    }

    private struct Bullet: View {
        let markdown: String
        init(_ markdown: String) { self.markdown = markdown }

        var body: some View {
            HStack(alignment: .top, spacing: 8) {
                Text("•")
                Text(.init(markdown))
            }
            .font(.footnote)
        }
    }
}
