//! IEEE 754 binary16: the codes a half-precision vector is stored as.
//!
//! Round to nearest even. An embedding is a unit vector of a thousand
//! dimensions, so its components sit around 0.03 and the nearest
//! representable value is within 0.05% of every one of them; values below
//! 2^-14 become subnormals and below 2^-25 become zero, which for a cosine
//! index is a rounding of a component that contributes nothing.

use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};

/// The binary16 code nearest to `value`, ties to even.
pub fn encode(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        // Infinity stays infinity; a NaN stays a NaN.
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    let rebased = exponent - 127 + 15;
    if rebased >= 0x1f {
        return sign | 0x7c00;
    }
    if rebased <= 0 {
        // Subnormal in binary16, or too small for it.
        if rebased < -10 {
            return sign;
        }
        let full = mantissa | 0x0080_0000;
        let shift = (14 - rebased) as u32;
        let halfway = 1u32 << (shift - 1);
        let rounded = full + halfway - 1 + ((full >> shift) & 1);
        return sign | (rounded >> shift) as u16;
    }
    // A carry out of the mantissa lands in the exponent, which is exactly
    // rounding up to the next binade.
    let rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    let code = ((rebased as u32) << 10) + (rounded >> 13);
    sign | code.min(0x7c00) as u16
}

/// The value a binary16 code stands for.
pub fn decode(code: u16) -> f32 {
    let sign = u32::from(code & 0x8000) << 16;
    let exponent = u32::from((code >> 10) & 0x1f);
    let mantissa = u32::from(code & 0x03ff);
    if exponent == 0 {
        let magnitude = mantissa as f32 / 16_777_216.0;
        return if sign == 0 { magnitude } else { -magnitude };
    }
    let bits = if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

/// Widens `codes` into `values`, which must be as long, with the CPU's own
/// conversion instruction where it has one: the values [`decode`] gives, which
/// a test checks over every code.
pub fn widen(codes: &[u16], values: &mut [f32]) {
    codes
        .reinterpret_cast::<half::f16>()
        .convert_to_f32_slice(values);
}

#[cfg(test)]
mod tests {
    use super::{decode, encode, widen};

    /// Every code that is a number comes back as itself.
    #[test]
    fn every_finite_code_round_trips() {
        for code in 0..=u16::MAX {
            if (code >> 10) & 0x1f == 0x1f {
                continue;
            }
            assert_eq!(encode(decode(code)), code, "{code:#06x}");
        }
    }

    /// The CPU conversion `widen` uses gives what `decode` does for every
    /// code; NaN payloads are not compared, as nothing stores one.
    #[test]
    fn the_cpu_conversion_matches_decode() {
        let codes: Vec<u16> = (0..=u16::MAX).collect();
        let mut widened = vec![0.0; codes.len()];
        widen(&codes, &mut widened);
        for (code, value) in codes.iter().zip(widened) {
            let expected = decode(*code);
            assert!(
                value.to_bits() == expected.to_bits() || (value.is_nan() && expected.is_nan()),
                "{code:#06x}: {value} vs {expected}"
            );
        }
    }

    /// Rounding is to the nearest code, ties to even, and saturates.
    #[test]
    fn rounding_is_to_the_nearest_code() {
        assert_eq!(encode(1.0), 0x3c00);
        assert_eq!(encode(-2.0), 0xc000);
        assert_eq!(encode(65_504.0), 0x7bff);
        assert_eq!(encode(1e9), 0x7c00, "too large is infinity");
        assert_eq!(encode(1e-9), 0x0000, "too small is zero");
        // Halfway between 1.0 and the next code rounds to the even one.
        assert_eq!(encode(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(encode(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        // What an embedding's components look like: within half a unit in
        // the last place, which is 2^-11 of the value.
        for value in [0.031_25_f32, -0.0172, 0.117, 3.1e-5] {
            let back = decode(encode(value));
            assert!((back - value).abs() <= value.abs() * 2f32.powi(-11) + 2f32.powi(-25));
        }
    }
}
