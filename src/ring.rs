use super::zq::*;

use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, SubAssign };
use std::cmp::Eq;
use std::convert::From;

use super::encoding;
use super::util;

///Degree of the ring extension
pub const N: usize = 256;

/// Elements of the ring Rq := Zq[X] / (X^N + 1)
pub trait RqElementCoefficientRepr: Eq + Clone + 
    for<'a> From<&'a [i16]> + From<[ZqElement; 256]> +
    for<'a> Add<&'a Self, Output = Self> + 
    for<'a> Sub<&'a Self, Output = Self> + 
    Mul<ZqElement, Output = Self> +
    for<'a> AddAssign<&'a Self> + 
    for<'a> SubAssign<&'a Self> + 
    MulAssign<ZqElement>
{
    type ChineseRemainderRepr: RqElementChineseRemainderRepr<CoefficientRepr = Self>;

    fn get_zero() -> Self;
    fn to_chinese_remainder_repr(self) -> Self::ChineseRemainderRepr;
    fn compress<const D: u16>(&self) -> CompressedRq<D>;
    fn decompress<const D: u16>(x: &CompressedRq<D>) -> Self;
}

pub trait RqElementChineseRemainderRepr: Eq + Clone + encoding::Encodable +
    for<'a> From<&'a [i16]> + From<[ZqElement; 256]> +
    for<'a> Add<&'a Self, Output = Self> + 
    for<'a> Sub<&'a Self, Output = Self> + 
    for<'a> Mul<&'a Self, Output = Self> + 
    for<'a> AddAssign<&'a Self> + 
    for<'a> SubAssign<&'a Self> +
    for<'a> MulAssign<&'a Self>
{
    type CoefficientRepr: RqElementCoefficientRepr<ChineseRemainderRepr = Self>;

    fn get_zero() -> Self;
    fn to_coefficient_repr(self) -> Self::CoefficientRepr;
    fn value_at_zeta(&self, zeta_index: usize) -> ZqElement;
    fn mul_scalar(&mut self, x: ZqElement);
    /// More efficient but semantically equivalent to `self += a * b`
    fn add_product(&mut self, a: &Self, b: &Self);
}

/// (Lossful) compression of a ring element using D bits. Therefore, the error
/// of encoding and decoding is at most Q/2^d
#[derive(Clone)]
pub struct CompressedRq<const D: u16>
{
    pub data: [CompressedZq<D>; N]
}

impl<const D: u16> encoding::Encodable for CompressedRq<D>
{
    fn encode<T: encoding::Encoder>(&self, encoder: &mut T)
    {
        for i in 0..N {
            self.data[i].encode(encoder);
        }
    }

    fn decode<T: encoding::Decoder>(data: &mut T) -> Self
    {
        CompressedRq {
            data: util::create_array(|_i| CompressedZq::decode(data))
        }
    }
}

impl<const D: u16> std::fmt::Debug for CompressedRq<D>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
    {
        write!(f, "[")?;
        for i in 0..N-1 {
            write!(f, "{}", self.data[i].data)?;
            write!(f, ",")?;
        }
        write!(f, "{}", self.data[N - 1].data)?;
        write!(f, "] | 0..{}", 1 << D)?;
        return Ok(());
    }
}

// The case D = 1 is a special one, as we use this to represent the plaintext.
// Therefore, we support to convert this from and to a byte array
impl CompressedRq<1>
{
    pub fn get_data(&self) -> [u8; 32]
    {
        let mut result: [u8; 32] = [0; 32];
        for i in 0..N {
            result[i/8] |= self.data[i].get_bit() << (i % 8);
        }
        return result;
    }

    pub fn from_data(m: [u8; 32]) -> CompressedRq<1>
    {
        let mut result: [CompressedZq<1>; N] = [CompressedZq::zero(); N];
        for i in 0..N {
            result[i] = CompressedZq::from_bit(m[i/8] >> (i % 8));
        }
        return CompressedRq {
            data: result
        };
    }
}