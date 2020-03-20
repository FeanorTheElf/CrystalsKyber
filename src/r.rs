use super::zq::*;

use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, SubAssign };
use std::cmp::Eq;
use std::convert::From;

use super::base64;
use super::util;

pub const N: usize = 256;

pub trait RingElement: Eq + Clone + 
             for<'a> From<&'a [i16]> +
             for<'a> Add<&'a Self, Output = Self> + 
             for<'a> Sub<&'a Self, Output = Self> + 
             Mul<Zq, Output = Self> +
             for<'a> AddAssign<&'a Self> + 
             for<'a> SubAssign<&'a Self> + 
             MulAssign<Zq>
{
    type FourierRepr: RingFourierRepr<StdRepr = Self>;

    fn zero() -> Self;
    fn dft(self) -> Self::FourierRepr;
    fn compress<const D : u16>(&self) -> CompressedR<D>;
    fn decompress<const D : u16>(x: &CompressedR<D>) -> Self;
}

pub trait RingFourierRepr: Eq + Clone +
                        for<'a> Add<&'a Self, Output = Self> + 
                        for<'a> Sub<&'a Self, Output = Self> + 
                        for<'a> Mul<&'a Self, Output = Self> + 
                        for<'a> AddAssign<&'a Self> + 
                        for<'a> SubAssign<&'a Self> +
                        for<'a> MulAssign<&'a Self>
{
    type StdRepr: RingElement<FourierRepr = Self>;

    fn zero() -> Self;
    fn inv_dft(self) -> Self::StdRepr;
    fn mul_scalar(&mut self, x: Zq);
    fn add_product(&mut self, a: &Self, b: &Self);
    fn encode(&self, encoder: &mut base64::Encoder);
    fn decode(data: &mut base64::Decoder) -> Self;
}

#[derive(Clone)]
pub struct CompressedR<const D: u16>
{
    pub data: [CompressedZq<D>; N]
}

impl<const D: u16> CompressedR<D>
{
    pub fn encode(&self, encoder: &mut base64::Encoder)
    {
        for i in 0..N {
            self.data[i].encode(encoder);
        }
    }

    pub fn decode(data: &mut base64::Decoder) -> Self
    {
        CompressedR {
            data: util::create_array(|_i| CompressedZq::decode(data))
        }
    }
}

impl<const D: u16> std::fmt::Debug for CompressedR<D>
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

impl CompressedR<1>
{
    pub fn get_data(&self) -> [u8; 32]
    {
        let mut result: [u8; 32] = [0; 32];
        for i in 0..N {
            result[i/8] |= self.data[i].get_bit() << (i % 8);
        }
        return result;
    }

    pub fn from_data(m: [u8; 32]) -> CompressedR<1>
    {
        let mut result: [CompressedZq<1>; N] = [CompressedZq::zero(); N];
        for i in 0..N {
            result[i] = CompressedZq::from_bit(m[i/8] >> (i % 8));
        }
        return CompressedR {
            data: result
        };
    }
}