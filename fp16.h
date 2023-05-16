#pragma once

// https://en.wikipedia.org/wiki/Half-precision_floating-point_format

typedef begin_packed struct _fp16_u_ {
    uint16_t bytes;
} end_packed _fp16_t_;

#undef fp16_t
#define fp16_t _fp16_t_

static_assertion(sizeof(fp16_t) == 2);

#define fp16x(hex)       ((fp16_t){ .bytes = hex })

#define FP16_NAN         fp16x(0x7FFF) // not a number one of many possible values
#define FP16_PINF        fp16x(0x7C00) // positive infinity
#define FP16_NINF        fp16x(0xFC00) // negative infinity
#define F16_DECIMAL_DIG  5             // # of decimal digits of rounding precision
#define F16_DIG          4             // # of decimal digits of precision
#define F16_EPSILON      fp16x(0x1400) // 9.7656250E-04 smallest such that 1.0 + F16_EPSILON != 1.0
#define F16_HAS_SUBNORM  1             // type does support subnormal numbers
#define F16_GUARD        0
#define F16_MANT_DIG     10            // # of bits in mantissa
#define F16_MAX          fp16x(0x7BFF) // 6.5504000E+04
#define F16_MAX_10_EXP   8             // max decimal exponent
#define F16_MAX_EXP      15            // max binary exponent
#define F16_MIN          fp16x(0x0400) // 6.1035156E-05 min normalized positive value
#define F16_MIN_10_EXP   (-8)          // min decimal exponent
#define F16_MIN_EXP      (-14)         // min binary exponent
#define F16_NORMALIZE    0
#define F16_RADIX        2             // exponent radix
#define F16_TRUE_MIN     fp16x(0x0001) // 2.9802322E-08 subnormal pow(2,-24) ~5.96E-8

inline bool fp16_isnan(fp16_t v) { return ((v.bytes >> 10) & 0x1F) == 0x1F && (v.bytes & 0x3FF) != 0; }

inline bool fp16_isfinite(fp16_t v) { return ((v.bytes >> 10) & 0x1F) != 0x1F; }

inline fp32_t fp16to32(fp16_t half) {
    uint16_t v = half.bytes;
    uint32_t sign = (v >> 15) & 0x00000001;
    uint32_t exponent = (v >> 10) & 0x0000001f;
    uint32_t mantissa = v & 0x000003ff;
    uint32_t result;
    if (exponent == 0 && mantissa == 0) {  // zero
        result = sign << 31;
    } else if (exponent == 0 && mantissa != 0) {  // denormalized
        exponent = 127 - 15;
        while ((mantissa & 0x00000400) == 0) {  // left shift mantissa
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= 0x000003ff;
        result = (sign << 31) | (exponent << 23) | (mantissa << 13);
    } else if (exponent == 0x1f && mantissa == 0) {  // infinity
        result = (sign << 31) | (exponent << 23);
    } else if (exponent == 0x1f && mantissa != 0) {  // NaN
        result = (sign << 31) | (exponent << 23) | (mantissa << 13);
    } else {  // normalized
        exponent = exponent + (127 - 15);
        mantissa = mantissa << 13;
        result = (sign << 31) | (exponent << 23) | mantissa;
    }
    fp32_t f;
    memcpy(&f, &result, sizeof(fp32_t));
    return f;
}

inline fp16_t fp32to16(fp32_t v) {
    uint32_t bits = *(uint32_t*)&v;
    uint32_t sign = bits >> 31;
    uint32_t exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;
    uint32_t result;
    if (exponent == 0) {  // zero or subnormal
        result = (sign << 15) | ((mantissa >> 13) & 0x7FFF);
    } else if (exponent == 0xFF) {  // NaN or infinity
        if (mantissa != 0) {  // NaN
            result = (sign << 15) | 0x7C00;
        } else {  // infinity
            result = (sign << 15) | 0x7C00 | ((1 << 10) - 1);
        }
    } else {  // normalized
        exponent = exponent + 15 - 127;
        if (exponent >= 31) {  // overflow
            result = (sign << 15) | 0x7C00;
        } else if (exponent <= 0) {  // underflow
            if (14 - exponent > 24) {  // round to zero
                result = sign << 15;
            } else {  // denormalized
                if (exponent < -10) { // too small for subnormal
                    mantissa = 0;
                } else {
                    mantissa |= 0x800000;  // add hidden bit
                    mantissa >>= 1 - exponent;
                }
                result = (sign << 15) | ((mantissa >> 13) & 0x7FFF);
            }
        } else {
            if (exponent > 15) { // too big to represent in fp16_t
                result = (sign << 15) | 0x7C00;
            } else {
                mantissa |= 0x800000;  // add hidden bit
                if (exponent == 15 && (mantissa & 0x3FF) != 0) {
                    // round to even
                    mantissa += (mantissa & 0x4000) >> 14;
                }
                result = (sign << 15) |
                        ((exponent << 10) & 0x7C00) |
                         (mantissa >> 13);
            }
        }
    }
    return (fp16_t){ .bytes = (uint16_t)result};
}

inline fp16_t fp16_add(fp16_t x, fp16_t y) {
    return fp32to16(fp16to32(x) + fp16to32(y));
}

inline fp16_t fp16_sub(fp16_t x, fp16_t y) {
    return fp32to16(fp16to32(x) - fp16to32(y));
}
inline fp16_t fp16_mul(fp16_t x, fp16_t y) {
    return fp32to16(fp16to32(x) * fp16to32(y));
}
inline fp16_t fp16_div(fp16_t x, fp16_t y) {
    return fp32to16(fp16to32(x) / fp16to32(y));
}

inline int fp16_compare(fp16_t x, fp16_t y) {
    fp16_t diff = fp16_sub(x, y);
    return (diff.bytes & 0x8000) ? -1 : (diff.bytes == 0) ? 0 : +1;
}

inline bool fp16_equ(fp16_t x, fp16_t y) { return fp16_compare(x, y) == 0; }
inline bool fp16_leq(fp16_t x, fp16_t y) { return fp16_compare(x, y) <= 0; }
inline bool fp16_les(fp16_t x, fp16_t y) { return fp16_compare(x, y) <  0; }
inline bool fp16_gtr(fp16_t x, fp16_t y) { return fp16_compare(x, y) >  0; }
inline bool fp16_gte(fp16_t x, fp16_t y) { return fp16_compare(x, y) >= 0; }
inline bool fp16_neq(fp16_t x, fp16_t y) { return fp16_compare(x, y) != 0; }

#include <float.h>

#ifdef RT_IMPLEMENTATION

void fp16_test() {
#if 0
    #define dump_f16(label, v) traceln("%-35s 0x%04X %.7E", label, v.bytes, fp16to32(v))
#else
    #define dump_f16(label, v) (void)(v); // unused
#endif
    fp16_t nan   = FP16_NAN;
    fp16_t inf_p = FP16_PINF;
    fp16_t inf_n = FP16_NINF;
    assertion(fp16_isnan(nan));
    assertion(!fp16_isfinite(inf_p));
    assertion(!fp16_isfinite(inf_n));
    assertion(fp16_isfinite(F16_MAX));
    assertion(fp16_isfinite(F16_MIN));
    assertion(fp16_isfinite(F16_EPSILON));
    assertion(fp16_isfinite(F16_TRUE_MIN));
    assertion(!fp16_isfinite(fp32to16(fp16to32(F16_MAX) + fp16to32(F16_MIN))));
    fp16_t one = fp32to16(1.0f); // 0x3C00
    assertion(fp16_equ(one, fp16x(0x3C00)));
    fp16_t one_plus_epsilon = fp16_add(one, F16_EPSILON);
    assert(fp16_neq(one_plus_epsilon, one));
    fp16_t one_plus_min = fp16_add(one, F16_MIN);
    assert(fp16_equ(one_plus_min, one));
    fp16_t smallest_positive_subnormal_number   = fp16x(0x0001); // fp32to16(0.000000059604645f);
    fp16_t largest_subnormal_number             = fp16x(0x03ff); // fp32to16(0.000060975552f);
    fp16_t smallest_positive_normal_number      = fp16x(0x0400); // fp32to16(0.00006103515625f);
    fp16_t nearest_value_to_one_third           = fp16x(0x3555); // fp32to16(0.33325195f);
    fp16_t largest_less_than_one                = fp16x(0x3BFF); // fp32to16(0.99951172f);
    fp16_t smallest_number_larger_than_one      = fp16x(0x3c01); // fp32to16(1.00097656f);
    fp16_t largest_normal_number                = fp16x(0x7BFF); // fp32to16(65504.0f);
    fp16_t largest_normal_number_negative       = fp16x(0xFBFF); // fp32to16(-65504.0f);
    dump_f16("smallest_positive_subnormal_number", smallest_positive_subnormal_number);
    dump_f16("largest_subnormal_number          ", largest_subnormal_number);
    dump_f16("smallest_positive_normal_number   ", smallest_positive_normal_number);
    dump_f16("nearest_value_to_one_third        ", nearest_value_to_one_third);
    dump_f16("largest_less_than_one             ", largest_less_than_one);
    dump_f16("smallest_number_larger_than_one   ", smallest_number_larger_than_one);
    dump_f16("largest_normal_number             ", largest_normal_number);
    dump_f16("largest_normal_number_negative    ", largest_normal_number_negative);
    dump_f16("F16_EPSILON                       ", F16_EPSILON);
    dump_f16("one                               ", one);
}

#endif
