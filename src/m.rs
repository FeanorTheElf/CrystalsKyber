use super::zq::*;
use super::r::*;
use super::util;
use super::base64;

use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, SubAssign };
use std::convert::From;

pub const DIM: usize = 3;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Module<T: RingElement>
{
    data: [T::FourierRepr; DIM]
}

impl<T: RingElement> Module<T>
{
    pub fn encode(&self, encoder: &mut base64::Encoder)
    {
        for element in &self.data {
            element.encode(encoder);
        }
    }

    pub fn decode(data: &mut base64::Decoder) -> Self
    {
        Module {
            data: util::create_array(|_i| T::FourierRepr::decode(data))
        }
    }
}

impl<'a, T: RingElement> Add<&'a Module<T>> for Module<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn add(mut self, rhs: &'a Module<T>) -> Module<T>
    {
        self += rhs;
        return self;
    }
}

impl<'a, T: RingElement> Add<Module<T>> for &'a Module<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn add(self, mut rhs: Module<T>) -> Module<T>
    {
        rhs += self;
        return rhs;
    }
}

impl<'a, T: RingElement> Sub<&'a Module<T>> for Module<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn sub(mut self, rhs: &'a Module<T>) -> Module<T>
    {
        self -= rhs;
        return self;
    }
}

impl<'a, T: RingElement> Sub<Module<T>> for &'a Module<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn sub(self, mut rhs: Module<T>) -> Module<T>
    {
        rhs -= self;
        rhs *= NEG_ONE;
        return rhs;
    }
}

impl<'a, T: RingElement> Mul<&'a Module<T>> for &'a Module<T>
{
    type Output = T::FourierRepr;

    #[inline(always)]
    fn mul(self, rhs: &'a Module<T>) -> T::FourierRepr
    {
        let mut result = T::FourierRepr::zero();
        for i in 0..DIM {
            result.add_product(&self.data[i], &rhs.data[i]);
        }
        return result;
    }
}

impl<'a, T: RingElement> Mul<&'a T::FourierRepr> for Module<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn mul(mut self, rhs: &'a T::FourierRepr) -> Module<T>
    {
        self *= rhs;
        return self;
    }
}

impl<'a, T: RingElement> AddAssign<&'a Module<T>> for Module<T>
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Module<T>) 
    {
        for i in 0..DIM {
            self.data[i] += &rhs.data[i];
        }
    }
}

impl<'a, T: RingElement> SubAssign<&'a Module<T>> for Module<T>
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Module<T>) 
    {
        for i in 0..DIM {
            self.data[i] -= &rhs.data[i];
        }
    }
}

impl<'a, T: RingElement> MulAssign<&'a T::FourierRepr> for Module<T>
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a T::FourierRepr) 
    {
        for i in 0..DIM {
            self.data[i] *= rhs;
        }
    }
}

impl<T: RingElement> MulAssign<Zq> for Module<T>
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq) 
    {
        for i in 0..DIM {
            self.data[i].mul_scalar(rhs);
        }
    }
}

impl<'a, T: RingElement> From<&'a [T::FourierRepr]> for Module<T>
{
    #[inline(always)]
    fn from(data: &'a [T::FourierRepr]) -> Module<T>
    {
        assert_eq!(DIM, data.len());
        Self::from(util::create_array(|i| data[i].clone()))
    }
}

impl<T: RingElement> From<[T::FourierRepr; DIM]> for Module<T>
{
    #[inline(always)]
    fn from(data: [T::FourierRepr; DIM]) -> Module<T>
    {
        Module {
            data: data
        }
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct Mat<T: RingElement>
{
    rows: [Module<T>; DIM]
}

#[derive(PartialEq, Eq, Clone)]
pub struct TransposedMat<'a, T: RingElement>
{
    data: &'a Mat<T>
}

impl<T: RingElement> Mat<T>
{
    pub fn transpose<'a>(&'a self) -> TransposedMat<'a, T>
    {
        TransposedMat {
            data: self
        }
    }
}

impl<'a, T: RingElement> TransposedMat<'a, T>
{
    pub fn transpose(&'a self) -> &'a Mat<T>
    {
        self.data
    }
}

impl<'a, T: RingElement> Mul<&'a Module<T>> for &'a Mat<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn mul(self, rhs: &'a Module<T>) -> Module<T> 
    {
        Module {
            data: [&self.rows[0] * rhs, &self.rows[1] * rhs, &self.rows[2] * rhs]
        }
    }
}

impl<'a, T: RingElement> Mul<&'a Module<T>> for TransposedMat<'a, T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn mul(self, rhs: &'a Module<T>) -> Module<T>
    {
        let mut result: [T::FourierRepr; DIM] = util::create_array(|_i| T::FourierRepr::zero());
        for row in 0..DIM {
            for col in 0..DIM {
                result[row].add_product(&self.data.rows[col].data[row], &rhs.data[col]);
            }
        }
        return Module {
            data: result
        };
    }
}

impl<T: RingElement> From<[[T::FourierRepr; DIM]; DIM]> for Mat<T>
{
    #[inline(always)]
    fn from(data: [[T::FourierRepr; DIM]; DIM]) -> Mat<T>
    {
        let [fst, snd, trd] = data;
        Mat {
            rows: [Module::from(fst), Module::from(snd), Module::from(trd)]
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressedM<const D: u16>
{
    data: [CompressedR<D>; DIM]
}

impl<const D: u16> CompressedM<D>
{
    pub fn encode(&self, encoder: &mut base64::Encoder)
    {
        for element in &self.data {
            element.encode(encoder);
        }
    }

    pub fn decode(data: &mut base64::Decoder) -> Self
    {
        CompressedM {
            data: util::create_array(|_i| CompressedR::decode(data))
        }
    }
}

impl<T: RingElement> Module<T>
{
    pub fn compress<const D: u16>(self) -> CompressedM<D>
    {
        let [fst, snd, trd] = self.data;
        CompressedM {
            data: [RingFourierRepr::inv_dft(fst).compress(), 
                RingFourierRepr::inv_dft(snd).compress(), 
                RingFourierRepr::inv_dft(trd).compress()]
        }
    }

    pub fn decompress<const D: u16>(x: &CompressedM<D>) -> Module<T>
    {
        Module {
            data: [T::dft(T::decompress(&x.data[0])),
                T::dft(T::decompress(&x.data[1])), 
                T::dft(T::decompress(&x.data[2]))]
        }
    }
}