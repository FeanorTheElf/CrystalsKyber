use super::zq::*;
use super::ring::*;
use super::util;
use super::base64;
use super::base64::Encodable;

use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, SubAssign };
use std::convert::From;

pub const DIM: usize = 3;

/// The module R^d where R is the given algebraic ring and d = DIM = 3.
/// It supports multiplication which is done component-wise in the ring R.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Module<R: Ring>
{
    data: [R::ChineseRemainderRepr; DIM]
}

impl<R: Ring> Module<R>
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
            data: util::try_create_array(|_i| R::ChineseRemainderRepr::decode(data))?
        })
    }
}

impl<'a, R: Ring> Add<&'a Module<R>> for Module<R>
{
    type Output = Module<R>;

    #[inline(always)]
    fn add(mut self, rhs: &'a Module<R>) -> Module<R>
    {
        self += rhs;
        return self;
    }
}

impl<'a, R: Ring> Add<Module<R>> for &'a Module<R>
{
    type Output = Module<R>;

    #[inline(always)]
    fn add(self, mut rhs: Module<R>) -> Module<R>
    {
        rhs += self;
        return rhs;
    }
}

impl<'a, R: Ring> Sub<&'a Module<R>> for Module<R>
{
    type Output = Module<R>;

    #[inline(always)]
    fn sub(mut self, rhs: &'a Module<R>) -> Module<R>
    {
        self -= rhs;
        return self;
    }
}

impl<'a, R: Ring> Sub<Module<R>> for &'a Module<R>
{
    type Output = Module<R>;

    #[inline(always)]
    fn sub(self, mut rhs: Module<R>) -> Module<R>
    {
        rhs -= self;
        rhs *= -ONE;
        return rhs;
    }
}

impl<'a, R: Ring> Mul<&'a Module<R>> for &'a Module<R>
{
    type Output = R::ChineseRemainderRepr;

    #[inline(always)]
    fn mul(self, rhs: &'a Module<R>) -> R::ChineseRemainderRepr
    {
        let mut result = R::ChineseRemainderRepr::zero();
        for i in 0..DIM {
            result.add_product(&self.data[i], &rhs.data[i]);
        }
        return result;
    }
}

impl<'a, R: Ring> Mul<&'a R::ChineseRemainderRepr> for Module<R>
{
    type Output = Module<R>;

    #[inline(always)]
    fn mul(mut self, rhs: &'a R::ChineseRemainderRepr) -> Module<R>
    {
        self *= rhs;
        return self;
    }
}

impl<'a, R: Ring> AddAssign<&'a Module<R>> for Module<R>
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Module<R>) 
    {
        for i in 0..DIM {
            self.data[i] += &rhs.data[i];
        }
    }
}

impl<'a, R: Ring> SubAssign<&'a Module<R>> for Module<R>
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Module<R>) 
    {
        for i in 0..DIM {
            self.data[i] -= &rhs.data[i];
        }
    }
}

impl<'a, R: Ring> MulAssign<&'a R::ChineseRemainderRepr> for Module<R>
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a R::ChineseRemainderRepr) 
    {
        for i in 0..DIM {
            self.data[i] *= rhs;
        }
    }
}

impl<R: Ring> MulAssign<Zq> for Module<R>
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq) 
    {
        for i in 0..DIM {
            self.data[i].mul_scalar(rhs);
        }
    }
}

impl<'a, R: Ring> From<&'a [R::ChineseRemainderRepr]> for Module<R>
{
    #[inline(always)]
    fn from(data: &'a [R::ChineseRemainderRepr]) -> Module<R>
    {
        assert_eq!(DIM, data.len());
        Self::from(util::create_array(|i| data[i].clone()))
    }
}

impl<R: Ring> From<[R::ChineseRemainderRepr; DIM]> for Module<R>
{
    #[inline(always)]
    fn from(data: [R::ChineseRemainderRepr; DIM]) -> Module<R>
    {
        Module {
            data: data
        }
    }
}

/// A dxd matrix over the given ring R, where d = DIM = 3.
#[derive(PartialEq, Eq, Clone)]
pub struct Matrix<R: Ring>
{
    rows: [Module<R>; DIM]
}

/// A reference onto the transpose of a dxd matrix over the ring R.
#[derive(PartialEq, Eq, Clone)]
pub struct TransposedMat<'a, R: Ring>
{
    data: &'a Matrix<R>
}

impl<R: Ring> Matrix<R>
{
    pub fn transpose<'a>(&'a self) -> TransposedMat<'a, R>
    {
        TransposedMat {
            data: self
        }
    }
}

impl<'a, R: Ring> TransposedMat<'a, R>
{
    pub fn transpose(&'a self) -> &'a Matrix<R>
    {
        self.data
    }
}

impl<'a, R: Ring> Mul<&'a Module<R>> for &'a Matrix<R>
{
    type Output = Module<R>;

    #[inline(always)]
    fn mul(self, rhs: &'a Module<R>) -> Module<R> 
    {
        Module {
            data: [&self.rows[0] * rhs, &self.rows[1] * rhs, &self.rows[2] * rhs]
        }
    }
}

impl<'a, R: Ring> Mul<&'a Module<R>> for TransposedMat<'a, R>
{
    type Output = Module<R>;

    #[inline(always)]
    fn mul(self, rhs: &'a Module<R>) -> Module<R>
    {
        let mut result: [R::ChineseRemainderRepr; DIM] = util::create_array(|_i| R::ChineseRemainderRepr::zero());
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

impl<R: Ring> From<[[R::ChineseRemainderRepr; DIM]; DIM]> for Matrix<R>
{
    #[inline(always)]
    fn from(data: [[R::ChineseRemainderRepr; DIM]; DIM]) -> Matrix<R>
    {
        let [fst, snd, trd] = data;
        Matrix {
            rows: [Module::from(fst), Module::from(snd), Module::from(trd)]
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressedModule<const D: u16>
{
    data: [CompressedRq<D>; DIM]
}

impl<const D: u16> base64::Encodable for CompressedModule<D>
{
    fn encode(&self, encoder: &mut base64::Encoder)
    {
        for element in &self.data {
            element.encode(encoder);
        }
    }

    fn decode(data: &mut base64::Decoder) -> base64::Result<Self>
    {
        Ok(CompressedModule {
            data: util::try_create_array(|_i| CompressedRq::decode(data))?
        })
    }
}

impl<T: Ring> Module<T>
{
    pub fn compress<const D: u16>(self) -> CompressedModule<D>
    {
        let [fst, snd, trd] = self.data;
        CompressedModule {
            data: [RingChineseRemainderRepr::coefficient_repr(fst).compress(), 
                RingChineseRemainderRepr::coefficient_repr(snd).compress(), 
                RingChineseRemainderRepr::coefficient_repr(trd).compress()]
        }
    }

    pub fn decompress<const D: u16>(x: &CompressedModule<D>) -> Module<T>
    {
        Module {
            data: [T::chinese_remainder_repr(T::decompress(&x.data[0])),
                T::chinese_remainder_repr(T::decompress(&x.data[1])), 
                T::chinese_remainder_repr(T::decompress(&x.data[2]))]
        }
    }
}