use std::arch::x86_64::*;
use std::ops::{ Add, Mul, Sub, Neg, AddAssign, MulAssign, SubAssign, DivAssign };
use std::cmp::{ PartialEq, Eq };
use std::fmt::{ Debug };

use super::zq::{ Zq, ONE };
use super::util::create_array;
use super::avx_util;
use super::avx_util::{ constant_f32, constant_i32, constant_zero, constant_u32 };

#[cfg(test)]
use super::zq::ZERO;

pub const Q: i32 = 7681;

#[derive(Clone, Copy)]
pub struct Zq8
{
    // 8 x 32bit integer, in positive representation (i.e. in 0..Q-1)
    data: __m256i
}

const NEG_Q: i32 = -Q;
const Q_DEC: i32 = Q - 1;
const Q_INV: f32 = 1. / 7681.;

macro_rules! impl_get {
    ($($ident:ident: $index:literal),*) => {
        $(
            pub fn $ident(&self) -> Zq
            {
                Zq::from( unsafe { _mm256_extract_epi32(self.data, $index) } as i16)
            }
        )*
    };
}

macro_rules! impl_set {
    ($($ident:ident: $index:literal),*) => {
        $(
            pub fn $ident(&mut self, value: Zq)
            {
                unsafe {
                    self.data = _mm256_blend_epi32(self.data, _mm256_set1_epi32(value.representative_pos() as i32), 1 << $index);
                }    
            }
        )*
    };
}

// works only if product <= 7680*7680
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

    pub fn sum_horizontal(&self) -> Zq
    {
        unsafe {
            let mut sum: i32 = avx_util::horizontal_sum(self.data);
            if sum >= 4 * Q {
                sum -= 4 * Q;
            }
            if sum >= 2 * Q {
                sum -= 2 * Q;
            }
            if sum >= Q {
                sum -= Q;
            }
            return Zq::from_perfect(sum as i16);
        }
    }

    pub fn shift_left(self, amount: usize) -> Zq8
    {
        Zq8 {
            data: unsafe { avx_util::shift_left(amount, self.data) }
        }
    }

    pub fn broadcast(x: Zq) -> Zq8
    {
        Zq8 {
            data: unsafe { _mm256_set1_epi32(x.representative_pos() as i32) }
        }
    }

}

pub fn transpose_vectorized_matrix<const COL_COUNT: usize, const VEC_COUNT: usize>(value: [Zq8; VEC_COUNT]) -> [Zq8; VEC_COUNT]
{
    let transposed = unsafe {
        avx_util::transpose_vectorized_matrix::<COL_COUNT, VEC_COUNT>(create_array(|i| value[i].data))
    };
    create_array(|i| Zq8 { data: transposed[i] })
}

impl<'a> From<&'a [Zq]> for Zq8
{
    fn from(value: &'a [Zq]) -> Zq8
    {
        assert_eq!(8, value.len());
        return Zq8::from(
            create_array(|i: usize| value[i].representative_pos())
        );
    }
}

impl<'a> From<&'a [i16]> for Zq8
{
    fn from(value: &'a [i16]) -> Zq8
    {
        assert_eq!(8, value.len());
        return Zq8::from(
            create_array(|i: usize| value[i])
        );
    }
}

impl From<[Zq; 8]> for Zq8
{
    fn from(value: [Zq; 8]) -> Zq8
    {
        return Zq8::from(
            create_array(|i: usize| value[i].representative_pos())
        );
    }
}

impl From<[i16; 8]> for Zq8
{
    #[inline(always)]
    fn from(value: [i16; 8]) -> Zq8
    {
        let f = |i: usize| value[i] as i32;
        let data = create_array!(f(0, 1, 2, 3, 4, 5, 6, 7));
        return Zq8 {
            data: unsafe { mod_q(avx_util::compose::<8, 1>(data)[0]) }
        };
    }
}

impl Eq for Zq8 {}

impl PartialEq for Zq8
{
    fn eq(&self, rhs: &Zq8) -> bool
    {
        unsafe {
            avx_util::eq(self.data, rhs.data)
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

impl AddAssign<Zq8> for Zq8
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Zq8)
    {
        unsafe {
            let sum = _mm256_add_epi32(self.data, rhs.data);
            let too_great = _mm256_cmpgt_epi32(sum, constant_i32::<Q_DEC>());
            self.data = _mm256_add_epi32(sum, _mm256_and_si256(too_great, constant_i32::<NEG_Q>()));
        }
    }
}

impl SubAssign<Zq8> for Zq8
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Zq8)
    {
        unsafe {
            let difference = _mm256_sub_epi32(self.data, rhs.data);
            let too_small = _mm256_cmpgt_epi32(constant_zero(), difference);
            self.data = _mm256_add_epi32(difference, _mm256_and_si256(too_small, constant_i32::<Q>()));
        }
    }
}

impl MulAssign<Zq8> for Zq8
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq8)
    {
        unsafe {
            let product: __m256i = _mm256_mullo_epi32(self.data, rhs.data);
            self.data = mod_q(product);
        }
    }
}

impl MulAssign<Zq> for Zq8
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq)
    {
        let factor = unsafe { _mm256_set1_epi32(rhs.representative_pos() as i32) };
        *self *= Zq8 {
            data: factor
        };
    }
}

impl DivAssign<Zq> for Zq8
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: Zq)
    {
        let inverse = ONE / rhs;
        let factor = unsafe { _mm256_set1_epi32(inverse.representative_pos() as i32) };
        *self *= Zq8 {
            data: factor
        };
    }
}

impl Neg for Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn neg(mut self) -> Self::Output
    {
        unsafe {
            self.data = _mm256_sub_epi32(self.data, constant_i32::<Q>());
        }
        return self;
    }
}

impl Add<Zq8> for Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn add(mut self, rhs: Zq8) -> Self::Output
    {
        self += rhs;
        return self;
    }
}

impl Sub<Zq8> for Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn sub(mut self, rhs: Zq8) -> Self::Output
    {
        self -= rhs;
        return self;
    }
}

impl Mul<Zq8> for Zq8
{
    type Output = Zq8;

    #[inline(always)]
    fn mul(mut self, rhs: Zq8) -> Self::Output
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

impl Zq8
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
    pub fn decompress<const D: u16>(x: CompressedZq8<D>) -> Zq8
    {
        unsafe {
            let factor = constant_f32::<{Q as f32 / (1 << D) as f32}>();
            let data_float = _mm256_cvtepi32_ps(x.data);
            let rounded = _mm256_cvtps_epi32(_mm256_mul_ps(data_float, factor));
            Zq8 {
                data: rounded
            }
        }
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
    v += w;
    assert_eq!(sum, v);
    v -= w;
    v -= w;
    assert_eq!(difference, v);
}

#[test]
fn test_mul() {
    let mut v: Zq8 = Zq8::from([3567, 132, 6113, 5432, -314, 543, 0, -321]);
    let w: Zq8 = Zq8::from([-5609, 12, 2386, -2728, -64, 12, -8000, -12]);
    let expected = Zq8::from([-5979, 1584, 7080, -1847, 4734, 6516, 0, 3852]);
    v *= w;
    assert_eq!(expected, v);
}

#[test]
fn test_shift_left() {
    let v: Zq8 = Zq8::from([0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(Zq8::from([1, 2, 3, 4, 5, 6, 7, 0]), v.shift_left(1));
}

#[test]
fn test_modulo_q() {
    for x in 0..7680*7680 {
        let p: i32 = x as i32;
        let f: f32 = p as f32;
        let q: f32 = f * Q_INV;
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

#[bench]
fn bench_add_mul(bencher: &mut test::Bencher)
{
    let data: [i16; 32] = core::hint::black_box([1145, 6716, 88, 5957, 3742, 3441, 2663, 1301, 159, 4074, 2945, 6671, 1392, 3999, 
        2394, 7624, 2420, 4199, 2762, 4206, 4471, 1582, 3870, 5363, 4246, 1800, 4568, 2081, 5642, 1115, 1242, 704]);
    let elements: [Zq8; 4] = [Zq8::from(&data[0..8]), Zq8::from(&data[8..16]), Zq8::from(&data[16..24]), Zq8::from(&data[24..32])];
    bencher.iter(|| {
        let mut result = Zq8::zero();
        for i in 0..4 {
            for j in 0..4 {
                result += elements[i] * elements[j];
                result += elements[i] * elements[j].shift_left(1);
                result += elements[i] * elements[j].shift_left(2);
                result += elements[i] * elements[j].shift_left(3);
                result += elements[i] * elements[j].shift_left(4);
                result += elements[i] * elements[j].shift_left(5);
                result += elements[i] * elements[j].shift_left(6);
                result += elements[i] * elements[j].shift_left(7);
            }
        }
        assert_eq!(Zq::from(4050), result.sum_horizontal());
    });
}