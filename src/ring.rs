use super::zq::*;

use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, SubAssign };
use std::cmp::Eq;
use std::convert::From;

use super::base64;
use super::util;

///Degree of the ring extension
pub const N: usize = 256;

/// Describes types that are algebraic ring extensions of Zq and have a chinese remainder representation, i.e.
/// the ring extension has degree N and Zq contains N-th roots of unity. Then the ring elements have
/// a representation in memory as vectors with component-wise addition and multiplication.
pub trait Ring: Eq + Clone + 
             for<'a> From<&'a [i16]> +
             for<'a> Add<&'a Self, Output = Self> + 
             for<'a> Sub<&'a Self, Output = Self> + 
             Mul<Zq, Output = Self> +
             for<'a> AddAssign<&'a Self> + 
             for<'a> SubAssign<&'a Self> + 
             MulAssign<Zq>
{
    type NTTDomain: RingNTTDomain<StdRepr = Self>;

    fn zero() -> Self;
    fn ntt(self) -> Self::NTTDomain;
    fn compress<const D : u16>(&self) -> CompressedRq<D>;
    fn decompress<const D : u16>(x: &CompressedRq<D>) -> Self;
}

pub trait RingNTTDomain: Eq + Clone + base64::Encodable +
                        for<'a> Add<&'a Self, Output = Self> + 
                        for<'a> Sub<&'a Self, Output = Self> + 
                        for<'a> Mul<&'a Self, Output = Self> + 
                        for<'a> AddAssign<&'a Self> + 
                        for<'a> SubAssign<&'a Self> +
                        for<'a> MulAssign<&'a Self>
{
    type StdRepr: Ring<NTTDomain = Self>;

    fn zero() -> Self;
    fn inv_ntt(self) -> Self::StdRepr;
    fn mul_scalar(&mut self, x: Zq);

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

impl<const D: u16> base64::Encodable for CompressedRq<D>
{
    fn encode(&self, encoder: &mut base64::Encoder)
    {
        for i in 0..N {
            self.data[i].encode(encoder);
        }
    }

    fn decode(data: &mut base64::Decoder) -> base64::Result<Self>
    {
        Ok(CompressedRq {
            data: util::try_create_array(|_i| CompressedZq::decode(data))?
        })
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