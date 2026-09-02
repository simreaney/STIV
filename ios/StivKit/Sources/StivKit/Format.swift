//  Format.swift
//  Number formatting for human-readable summaries.
//
//  `String(format:)` with `%d` and a Swift `Int` is a varargs width mismatch on
//  64-bit -- it happens to work on little-endian hardware and is wrong anyway --
//  so integers are interpolated and only fixed-decimal Doubles go through
//  printf, via this one helper.

import Foundation

/// A Double at a fixed number of decimal places, or `n/a` when it is not a
/// number.  Reports say "n/a" rather than "nan": a gauging summary is read by
/// hydrologists, not by C programmers.
public func decimals(_ v: Double, _ places: Int) -> String {
    guard v.isFinite else { return "n/a" }
    return String(format: "%.\(places)f", v)
}

/// A Double as a percentage at a fixed number of decimal places.
public func percent(_ v: Double, _ places: Int = 0) -> String {
    guard v.isFinite else { return "n/a" }
    return String(format: "%.\(places)f%%", v * 100)
}
