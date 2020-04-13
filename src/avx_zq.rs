use std::arch::x86_64::*;
use std::ops::{ Add, Mul, Sub, Neg, AddAssign, MulAssign, SubAssign, DivAssign };
use std::cmp::{ PartialEq, Eq };
use std::fmt::{ Debug };

use super::zq;
use super::zq::{ ZqElement, ONE };
use super::util::create_array;
use super::util;
use super::avx_util;
use super::avx_util::{ constant_f32, constant_i32, constant_zero, constant_u32 };

pub const Q: i32 = zq::Q as i32;

/// Vectors of 8 elements of the field Zq = Z/qZ for q = Q = 7681. Addition
/// and multiplication are done component-wise.
#[derive(Clone, Copy)]
pub struct ZqVector8
{
    // 8 x 32bit integer, in positive representation (i.e. in 0..Q-1)
    data: __m256i
}

const NEG_Q: i32 = -Q;
const Q_DEC: i32 = Q - 1;
const Q_INV: f32 = 1. / (Q as f32);

// Works only if product <= 7680^2, but this is sufficient to reduce the products mod q.
unsafe fn mod_q(product: __m256i) -> __m256i
{
    // use floating point arithmetic with correction
    // in order to perform modulo using multiplication
    // (this requires the upper bits of a 26 + 24 bit product)
    // the floating point arithmetic gives us at least
    // the upper bits of a 24 + 24 bit product, so correction
    // is required (see test_modulo_q)
    let product_float: __m256 = _mm256_cvtepi32_ps(product); 
    let quotient: __m256 = _mm256_mul_ps(product_float, constant_f32::<Q_INV>());
    let rounded_quotient: __m256i = _mm256_cvttps_epi32(quotient);
    let rest: __m256i = _mm256_sub_epi32(product, _mm256_mullo_epi32(rounded_quotient, constant_i32::<Q>()));
    // apply correction: rest is now in -7681 to 2 * 7680 - 1
    let too_small = _mm256_cmpgt_epi32(constant_zero(), rest);
    let too_big = _mm256_cmpgt_epi32(rest, constant_i32::<Q_DEC>());
    let correction = _mm256_or_si256(_mm256_and_si256(too_small, constant_i32::<Q>()), _mm256_and_si256(too_big, constant_i32::<NEG_Q>()));
    return _mm256_add_epi32(rest, correction);
}

impl ZqVector8 
{
    pub fn zero() -> ZqVector8
    {
        ZqVector8 {
            data: unsafe { _mm256_setzero_si256() }
        }
    }

    pub fn broadcast(x: ZqElement) -> ZqVector8
    {
        ZqVector8 {
            data: unsafe { _mm256_set1_epi32(x.representative_pos() as i32) }
        }
    }

    pub fn as_array(&self) -> [ZqElement; 8]
    {
        let data = unsafe { [
            _mm256_extract_epi32(self.data, 0), 
            _mm256_extract_epi32(self.data, 1), 
            _mm256_extract_epi32(self.data, 2), 
            _mm256_extract_epi32(self.data, 3), 
            _mm256_extract_epi32(self.data, 4), 
            _mm256_extract_epi32(self.data, 5), 
            _mm256_extract_epi32(self.data, 6), 
            _mm256_extract_epi32(self.data, 7)
        ] };
        return util::create_array(|i| ZqElement::from_perfect(data[i] as i16));
    }
}

pub fn transpose_vectorized_matrix<const COL_COUNT: usize, const VEC_COUNT: usize>(value: [ZqVector8; VEC_COUNT]) -> [ZqVector8; VEC_COUNT]
{
    let transposed = unsafe {
        avx_util::transpose_vectorized_matrix::<COL_COUNT, VEC_COUNT>(create_array(|i| value[i].data))
    };
    create_array(|i| ZqVector8 { data: transposed[i] })
}

impl<'a> From<&'a [ZqElement]> for ZqVector8
{
    fn from(value: &'a [ZqElement]) -> ZqVector8
    {
        assert_eq!(8, value.len());
        return ZqVector8::from(
            create_array(|i| value[i])
        );
    }
}

impl<'a> From<&'a [i16]> for ZqVector8
{
    fn from(value: &'a [i16]) -> ZqVector8
    {
        assert_eq!(8, value.len());
        return ZqVector8::from(
            create_array(|i| value[i])
        );
    }
}

impl From<[ZqElement; 8]> for ZqVector8
{
    fn from(value: [ZqElement; 8]) -> ZqVector8
    {
        let data = create_array(|i| value[i].representative_pos() as i32);
        return ZqVector8 {
            data: unsafe { avx_util::compose::<8, 1>(data)[0] }
        };
    }
}

impl From<[i16; 8]> for ZqVector8
{
    #[inline(always)]
    fn from(value: [i16; 8]) -> ZqVector8
    {
        let data = create_array(|i| value[i] as i32);
        return ZqVector8 {
            data: unsafe { mod_q(avx_util::compose::<8, 1>(data)[0]) }
        };
    }
}

impl Eq for ZqVector8 {}

impl PartialEq for ZqVector8
{
    fn eq(&self, rhs: &ZqVector8) -> bool
    {
        unsafe {
            avx_util::eq(self.data, rhs.data)
        }
    }
}

impl Debug for ZqVector8
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result 
    {
        let data = self.as_array();
        write!(f, "({}, {}, {}, {}, {}, {}, {}, {})", data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
    }
}

impl AddAssign<ZqVector8> for ZqVector8
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: ZqVector8)
    {
        unsafe {
            let sum = _mm256_add_epi32(self.data, rhs.data);
            let too_great = _mm256_cmpgt_epi32(sum, constant_i32::<Q_DEC>());
            self.data = _mm256_add_epi32(sum, _mm256_and_si256(too_great, constant_i32::<NEG_Q>()));
        }
    }
}

impl SubAssign<ZqVector8> for ZqVector8
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: ZqVector8)
    {
        unsafe {
            let difference = _mm256_sub_epi32(self.data, rhs.data);
            let too_small = _mm256_cmpgt_epi32(constant_zero(), difference);
            self.data = _mm256_add_epi32(difference, _mm256_and_si256(too_small, constant_i32::<Q>()));
        }
    }
}

impl MulAssign<ZqVector8> for ZqVector8
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: ZqVector8)
    {
        unsafe {
            let product: __m256i = _mm256_mullo_epi32(self.data, rhs.data);
            self.data = mod_q(product);
        }
    }
}

impl MulAssign<ZqElement> for ZqVector8
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: ZqElement)
    {
        let factor = unsafe { _mm256_set1_epi32(rhs.representative_pos() as i32) };
        *self *= ZqVector8 {
            data: factor
        };
    }
}

// We only support to divide the vector by a scalar, since a component-wise vector-vector
// multiplication is super inefficient with avx.
impl DivAssign<ZqElement> for ZqVector8
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: ZqElement)
    {
        let inverse = ONE / rhs;
        let factor = unsafe { _mm256_set1_epi32(inverse.representative_pos() as i32) };
        *self *= ZqVector8 {
            data: factor
        };
    }
}

impl Neg for ZqVector8
{
    type Output = ZqVector8;

    #[inline(always)]
    fn neg(mut self) -> Self::Output
    {
        unsafe {
            self.data = _mm256_sub_epi32(self.data, constant_i32::<Q>());
        }
        return self;
    }
}

impl Add<ZqVector8> for ZqVector8
{
    type Output = ZqVector8;

    #[inline(always)]
    fn add(mut self, rhs: ZqVector8) -> Self::Output
    {
        self += rhs;
        return self;
    }
}

impl Sub<ZqVector8> for ZqVector8
{
    type Output = ZqVector8;

    #[inline(always)]
    fn sub(mut self, rhs: ZqVector8) -> Self::Output
    {
        self -= rhs;
        return self;
    }
}

impl Mul<ZqVector8> for ZqVector8
{
    type Output = ZqVector8;

    #[inline(always)]
    fn mul(mut self, rhs: ZqVector8) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

#[derive(Clone, Copy)]
pub struct CompressedZq8<const D: u16>
{
    pub data: __m256i
}

impl ZqVector8
{

    pub fn compress<const D: u16>(self) -> CompressedZq8<D>
    {
        // this floating point approach always leads to the right result:
        // for each x, n, |0.5 - (x * n / 7681) mod 1| >= |0.5 - (x * 1 / 7681) mod 1|
        // >= |0.5 - (3840 / 7681) mod 1| >= 6.509569066531773E-5 
        // > (error in floating point representation of 1/7681) * 7681
        unsafe {
            let representation_pos_float = _mm256_cvtepi32_ps(self.data);
            let factor = constant_f32::<{(1 << D) as f32 / Q as f32}>();
            let unrounded_result = _mm256_mul_ps(representation_pos_float, factor);
            let rounded_result = _mm256_cvtps_epi32(unrounded_result);
            let result = _mm256_and_si256(constant_u32::<{(1 << D) as u32 - 1}>(), rounded_result);
            CompressedZq8 {
                data: result
            }
        }
    }
    
    // Returns the element y of Zq for which
    // y.representative_pos() is nearest to 2^d/q * x 
    pub fn decompress<const D: u16>(x: CompressedZq8<D>) -> ZqVector8
    {
        unsafe {
            let factor = constant_f32::<{Q as f32 / (1 << D) as f32}>();
            let data_float = _mm256_cvtepi32_ps(x.data);
            let rounded = _mm256_cvtps_epi32(_mm256_mul_ps(data_float, factor));
            ZqVector8 {
                data: rounded
            }
        }
    }
}

#[test]
fn test_from() {
    let v: ZqVector8 = ZqVector8::from([(-3) * 7681, 4 * 7681 + 625, 1, 0, -7680, 2 * 7681 + 3000, -1, 2 * 7681 + 6000]);
    let w: ZqVector8 = ZqVector8::from([0, 625, 1, 0, 1, 3000, 7680, 6000]);
    assert_eq!(v, w);
}

#[test]
fn test_add_sub() {
    let mut v: ZqVector8 = ZqVector8::from([3567, 132, 6113, 5432, -314, 543, 0, -321]);
    let w: ZqVector8 = ZqVector8::from([-5609, 12, 2386, -2728, -64, 12, -8000, -12]);
    let sum = ZqVector8::from([-2042, 144, 818, 2704, -378, 555, -319, -333]);
    let difference = ZqVector8::from([1495, 120, 3727, 479, -250, 531, 319, -309]);
    v += w;
    assert_eq!(sum, v);
    v -= w;
    v -= w;
    assert_eq!(difference, v);
}

#[test]
fn test_mul() {
    let mut v: ZqVector8 = ZqVector8::from([3567, 132, 6113, 5432, -314, 543, 0, -321]);
    let w: ZqVector8 = ZqVector8::from([-5609, 12, 2386, -2728, -64, 12, -8000, -12]);
    let expected = ZqVector8::from([-5979, 1584, 7080, -1847, 4734, 6516, 0, 3852]);
    v *= w;
    assert_eq!(expected, v);
}

#[test]
fn test_modulo_q() {
    // Check that the hacky mod_q procedure really works for all values in (Q - 1)^2
    unsafe {
        for x in 0..(Q * Q / 8 + 1) {
            let value = avx_util::compose::<8, 1>(util::create_array(|i| x + i as i32))[0];
            let expected = avx_util::compose::<8, 1>(util::create_array(|i| (x + i as i32) % Q))[0];
            assert!(avx_util::eq(expected, mod_q(value)));
        }
    }
}
