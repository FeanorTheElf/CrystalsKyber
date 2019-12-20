use std::arch::x86_64::*;
use std::ops::{ Add, Mul, Sub, Neg, AddAssign, MulAssign, SubAssign };
use std::cmp::{ PartialEq, Eq };
use std::fmt::{ Debug };

use super::zq::Zq;

#[cfg(test)]
use super::zq::ZERO;

pub const Q: i32 = 7681;

#[derive(Clone, Copy)]
pub struct Zq8
{
    // 8 x 32bit integer
    data: __m256i
}

unsafe fn avx_q() -> __m256i { _mm256_set1_epi32 (Q) }
unsafe fn avx_negative_q() -> __m256i { _mm256_set1_epi32 (-Q) }
unsafe fn avx_q_minus_one() -> __m256i { _mm256_set1_epi32 (Q-1) }
const ONE_OVER_Q: f32 = 1. / 7681.;
unsafe fn avx_one_over_q() -> __m256 { _mm256_set1_ps(ONE_OVER_Q) }
unsafe fn avx_zero() -> __m256i { _mm256_setzero_si256() }

macro_rules! impl_get {
    ($($ident:ident: $index:literal),*) => {
        $(
            fn $ident(&self) -> Zq
            {
                Zq::from( unsafe { _mm256_extract_epi32(self.data, $index) } as i16)
            }
        )*
    };
}

macro_rules! impl_set {
    ($($ident:ident: $index:literal),*) => {
        $(
            fn $ident(&mut self, value: Zq)
            {
                unsafe {
                    self.data = _mm256_blend_epi32(self.data, _mm256_set1_epi32(value.representative_pos() as i32), 1 << $index);
                }    
            }
        )*
    };
}

unsafe fn mod_q(product: __m256i) -> __m256i
{
    let product_float: __m256 = _mm256_cvtepi32_ps(product); 
    let quotient: __m256 = _mm256_mul_ps(product_float, avx_one_over_q());
    let rounded_quotient: __m256i = _mm256_cvttps_epi32(quotient);
    let rest: __m256i = _mm256_sub_epi32(product, _mm256_mullo_epi32(rounded_quotient, avx_q()));
    // apply correction: rest is now in -7681 to 2 * 7680 - 1
    let too_small = _mm256_cmpgt_epi32(avx_zero(), rest);
    let too_big = _mm256_cmpgt_epi32(rest, avx_q_minus_one());
    let correction = _mm256_or_si256(_mm256_and_si256(too_small, avx_q()), _mm256_and_si256(too_big, avx_negative_q()));
    return _mm256_add_epi32(rest, correction);
}

impl Zq8 
{
    pub fn zero() -> Zq8
    {
        Zq8 {
            data: unsafe { _mm256_setzero_si256() }
        }
    }

    impl_get!(get_0: 0, get_1: 1, get_2: 2, get_3: 3, get_4: 4, get_5: 5, get_6: 6, get_7: 7);
    impl_set!(set_0: 0, set_1: 1, set_2: 2, set_3: 3, set_4: 4, set_5: 5, set_6: 6, set_7: 7);
}

impl From<[Zq; 8]> for Zq8
{
    fn from(value: [Zq; 8]) -> Zq8
    {
        Zq8 {
            data: unsafe { _mm256_setr_epi32(
                value[0].representative_pos() as i32, 
                value[1].representative_pos() as i32, 
                value[2].representative_pos() as i32, 
                value[3].representative_pos() as i32, 
                value[4].representative_pos() as i32, 
                value[5].representative_pos() as i32, 
                value[6].representative_pos() as i32, 
                value[7].representative_pos() as i32) 
            }
        }
    }
}

impl From<[i16; 8]> for Zq8
{
    fn from(value: [i16; 8]) -> Zq8
    {
        unsafe {
            let items = _mm256_setr_epi32(value[0] as i32, 
                value[1] as i32, 
                value[2] as i32, 
                value[3] as i32, 
                value[4] as i32, 
                value[5] as i32, 
                value[6] as i32, 
                value[7] as i32);
            return Zq8 {
                data: mod_q(items)
            };
        }
    }
}

impl Eq for Zq8 {}

impl PartialEq for Zq8
{
    fn eq(&self, rhs: &Zq8) -> bool
    {
        unsafe {
            let equality: __m256i = _mm256_cmpeq_epi32(self.data, rhs.data);
            let bitmask: i32 = _mm256_movemask_epi8(equality);
            return bitmask == !0;
        }
    }
}

impl Debug for Zq8
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result 
    {
        write!(f, "({}, {}, {}, {}, {}, {}, {}, {})", self.get_0(), self.get_1(), self.get_2(), 
            self.get_3(), self.get_4(), self.get_5(), self.get_6(), self.get_7())
    }
}

impl<'a> AddAssign<&'a Zq8> for Zq8
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Zq8)
    {
        unsafe {
            let sum = _mm256_add_epi32(self.data, rhs.data);
            let too_great = _mm256_cmpgt_epi32(sum, avx_q_minus_one());
            self.data = _mm256_add_epi32(sum, _mm256_and_si256(too_great, avx_negative_q()));
        }
    }
}

impl<'a> SubAssign<&'a Zq8> for Zq8
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Zq8)
    {
        unsafe {
            let difference = _mm256_sub_epi32(self.data, rhs.data);
            let too_small = _mm256_cmpgt_epi32(avx_zero(), difference);
            self.data = _mm256_add_epi32(difference, _mm256_and_si256(too_small, avx_q()));
        }
    }
}

impl<'a> MulAssign<&'a Zq8> for Zq8
{
    #[no_mangle]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a Zq8)
    {
        // use floating point arithmetic with correction
        // in order to perform modulo using multiplication
        // (this requires the upper bits of a 26 + 24 bit product)
        // the floating point arithmetic gives us at least
        // the upper bits of a 24 + 24 bit product, so correction
        // is required (see test_modulo_q)
        unsafe {
            let product: __m256i = _mm256_mullo_epi32(self.data, rhs.data);
            self.data = mod_q(product);
        }
    }
}

impl Neg for Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn neg(mut self) -> Self::Output
    {
        unsafe {
            self.data = _mm256_sub_epi32(self.data, avx_q());
        }
        return self;
    }
}

impl<'a> Add<&'a Zq8> for Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn add(mut self, rhs: &'a Zq8) -> Self::Output
    {
        self += rhs;
        return self;
    }
}

impl<'a> Add<Zq8> for &'a Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn add(self, mut rhs: Zq8) -> Self::Output
    {
        rhs += self;
        return rhs;
    }
}

impl<'a> Sub<&'a Zq8> for Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn sub(mut self, rhs: &'a Zq8) -> Self::Output
    {
        self -= rhs;
        return self;
    }
}

impl<'a> Sub<Zq8> for &'a Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn sub(self, mut rhs: Zq8) -> Self::Output
    {
        rhs -= self;
        return -rhs;
    }
}

impl<'a> Mul<&'a Zq8> for Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn mul(mut self, rhs: &'a Zq8) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

impl<'a> Mul<Zq8> for &'a Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn mul(self, mut rhs: Zq8) -> Self::Output
    {
        rhs *= self;
        return rhs;
    }
}

#[test]
fn test_get_set() {
    let mut vector = Zq8::zero();
    assert_eq!(ZERO, vector.get_0());
    assert_eq!(ZERO, vector.get_1());
    assert_eq!(ZERO, vector.get_2());
    assert_eq!(ZERO, vector.get_3());
    assert_eq!(ZERO, vector.get_4());
    assert_eq!(ZERO, vector.get_5());
    assert_eq!(ZERO, vector.get_6());
    assert_eq!(ZERO, vector.get_7());

    vector.set_1(Zq::from(1_i16));
    vector.set_2(Zq::from(2_i16));
    vector.set_3(Zq::from(3_i16));
    vector.set_4(Zq::from(4_i16));
    vector.set_5(Zq::from(5_i16));
    vector.set_6(Zq::from(6_i16));
    vector.set_7(Zq::from(7_i16));

    assert_eq!(0, vector.get_0().representative_pos());
    assert_eq!(1, vector.get_1().representative_pos());
    assert_eq!(2, vector.get_2().representative_pos());
    assert_eq!(3, vector.get_3().representative_pos());
    assert_eq!(4, vector.get_4().representative_pos());
    assert_eq!(5, vector.get_5().representative_pos());
    assert_eq!(6, vector.get_6().representative_pos());
    assert_eq!(7, vector.get_7().representative_pos());
}

#[test]
fn test_from() {
    let v: Zq8 = Zq8::from([(-3) * 7681, 4 * 7681 + 625, 1, 0, -7680, 2 * 7681 + 3000, -1, 2 * 7681 + 6000]);
    let w: Zq8 = Zq8::from([0, 625, 1, 0, 1, 3000, 7680, 6000]);
    assert_eq!(v, w);
}

#[test]
fn test_add_sub() {
    let mut v: Zq8 = Zq8::from([3567, 132, 6113, 5432, -314, 543, 0, -321]);
    let w: Zq8 = Zq8::from([-5609, 12, 2386, -2728, -64, 12, -8000, -12]);
    let sum = Zq8::from([-2042, 144, 818, 2704, -378, 555, -319, -333]);
    let difference = Zq8::from([1495, 120, 3727, 479, -250, 531, 319, -309]);
    v += &w;
    assert_eq!(sum, v);
    v -= &w;
    v -= &w;
    assert_eq!(difference, v);
}

#[test]
fn test_mul() {
    let mut v: Zq8 = Zq8::from([3567, 132, 6113, 5432, -314, 543, 0, -321]);
    let w: Zq8 = Zq8::from([-5609, 12, 2386, -2728, -64, 12, -8000, -12]);
    let expected = Zq8::from([-5979, 1584, 7080, -1847, 4734, 6516, 0, 3852]);
    v *= &w;
    assert_eq!(expected, v);
}

#[test]
fn test_modulo_q() {
    for x in 0..7680*7680 {
        let p: i32 = x as i32;
        let f: f32 = p as f32;
        let q: f32 = f * ONE_OVER_Q;
        let r: i32 = q.floor() as i32;
        let mut m: i32 = p - r * 7681;
        if m < 0 {
            m += 7681;
        } else if m >= 7681 {
            m -= 7681;
        }
        assert!((x as i32) % 7681 == m, "Expected {}, got {} at for {}", (x as i32) % 7681, m, x);
    }
}