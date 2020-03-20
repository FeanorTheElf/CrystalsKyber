use super::zq::*;
use super::ring::*;
use super::util;
use super::base64;
use super::base64::Encodable;

use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, SubAssign };
use std::convert::From;

pub const DIM: usize = 3;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Module<T: Ring>
{
    data: [T::FourierRepr; DIM]
}

impl<T: Ring> Module<T>
{
    pub fn encode(&self, encoder: &mut base64::Encoder)
    {
        for element in &self.data {
            element.encode(encoder);
        }
    }

    pub fn decode(data: &mut base64::Decoder) -> base64::Result<Self>
    {
        Ok(Module {
            data: util::try_create_array(|_i| T::FourierRepr::decode(data))?
        })
    }
}

impl<'a, T: Ring> Add<&'a Module<T>> for Module<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn add(mut self, rhs: &'a Module<T>) -> Module<T>
    {
        self += rhs;
        return self;
    }
}

impl<'a, T: Ring> Add<Module<T>> for &'a Module<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn add(self, mut rhs: Module<T>) -> Module<T>
    {
        rhs += self;
        return rhs;
    }
}

impl<'a, T: Ring> Sub<&'a Module<T>> for Module<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn sub(mut self, rhs: &'a Module<T>) -> Module<T>
    {
        self -= rhs;
        return self;
    }
}

impl<'a, T: Ring> Sub<Module<T>> for &'a Module<T>
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

impl<'a, T: Ring> Mul<&'a Module<T>> for &'a Module<T>
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

impl<'a, T: Ring> Mul<&'a T::FourierRepr> for Module<T>
{
    type Output = Module<T>;

    #[inline(always)]
    fn mul(mut self, rhs: &'a T::FourierRepr) -> Module<T>
    {
        self *= rhs;
        return self;
    }
}

impl<'a, T: Ring> AddAssign<&'a Module<T>> for Module<T>
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Module<T>) 
    {
        for i in 0..DIM {
            self.data[i] += &rhs.data[i];
        }
    }
}

impl<'a, T: Ring> SubAssign<&'a Module<T>> for Module<T>
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Module<T>) 
    {
        for i in 0..DIM {
            self.data[i] -= &rhs.data[i];
        }
    }
}

impl<'a, T: Ring> MulAssign<&'a T::FourierRepr> for Module<T>
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a T::FourierRepr) 
    {
        for i in 0..DIM {
            self.data[i] *= rhs;
        }
    }
}

impl<T: Ring> MulAssign<Zq> for Module<T>
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq) 
    {
        for i in 0..DIM {
            self.data[i].mul_scalar(rhs);
        }
    }
}

impl<'a, T: Ring> From<&'a [T::FourierRepr]> for Module<T>
{
    #[inline(always)]
    fn from(data: &'a [T::FourierRepr]) -> Module<T>
    {
        assert_eq!(DIM, data.len());
        Self::from(util::create_array(|i| data[i].clone()))
    }
}

impl<T: Ring> From<[T::FourierRepr; DIM]> for Module<T>
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
pub struct Matrix<T: Ring>
{
    rows: [Module<T>; DIM]
}

#[derive(PartialEq, Eq, Clone)]
pub struct TransposedMat<'a, T: Ring>
{
    data: &'a Matrix<T>
}

impl<T: Ring> Matrix<T>
{
    pub fn transpose<'a>(&'a self) -> TransposedMat<'a, T>
    {
        TransposedMat {
            data: self
        }
    }
}

impl<'a, T: Ring> TransposedMat<'a, T>
{
    pub fn transpose(&'a self) -> &'a Matrix<T>
    {
        self.data
    }
}

impl<'a, T: Ring> Mul<&'a Module<T>> for &'a Matrix<T>
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

impl<'a, T: Ring> Mul<&'a Module<T>> for TransposedMat<'a, T>
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

impl<T: Ring> From<[[T::FourierRepr; DIM]; DIM]> for Matrix<T>
{
    #[inline(always)]
    fn from(data: [[T::FourierRepr; DIM]; DIM]) -> Matrix<T>
    {
        let [fst, snd, trd] = data;
        Matrix {
            rows: [Module::from(fst), Module::from(snd), Module::from(trd)]
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressedM<const D: u16>
{
    data: [CompressedRq<D>; DIM]
}

impl<const D: u16> base64::Encodable for CompressedM<D>
{
    fn encode(&self, encoder: &mut base64::Encoder)
    {
        for element in &self.data {
            element.encode(encoder);
        }
    }

    fn decode(data: &mut base64::Decoder) -> base64::Result<Self>
    {
        Ok(CompressedM {
            data: util::try_create_array(|_i| CompressedRq::decode(data))?
        })
    }
}

impl<T: Ring> Module<T>
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