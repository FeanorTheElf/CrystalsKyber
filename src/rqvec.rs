use super::zq::*;
use super::ring::*;
use super::util;
use super::encoding;

use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, SubAssign };
use std::convert::From;

pub const DIM: usize = 3;

/// The module R^d where R is an implementation of the ring Rq and d = DIM = 3.
/// It supports an inner product which is done component-wise in the ring R. Since
/// R is given as a generic parameter, this type is used for both the reference
/// and the avx implementation.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct RqVector3<R: RqElementCoefficientRepr>
{
    pub data: [R::ChineseRemainderRepr; DIM]
}

impl<R: RqElementCoefficientRepr> encoding::Encodable for RqVector3<R>
{
    fn encode<T: encoding::Encoder>(&self, encoder: &mut T)
    {
        for element in &self.data {
            element.encode(encoder);
        }
    }

    fn decode<T: encoding::Decoder>(data: &mut T) -> Self
    {
        RqVector3 {
            data: util::create_array(|_i| R::ChineseRemainderRepr::decode(data))
        }
    }
}

impl<'a, R: RqElementCoefficientRepr> Add<&'a RqVector3<R>> for RqVector3<R>
{
    type Output = RqVector3<R>;

    #[inline(always)]
    fn add(mut self, rhs: &'a RqVector3<R>) -> RqVector3<R>
    {
        self += rhs;
        return self;
    }
}

impl<'a, R: RqElementCoefficientRepr> Add<RqVector3<R>> for &'a RqVector3<R>
{
    type Output = RqVector3<R>;

    #[inline(always)]
    fn add(self, mut rhs: RqVector3<R>) -> RqVector3<R>
    {
        rhs += self;
        return rhs;
    }
}

impl<'a, R: RqElementCoefficientRepr> Sub<&'a RqVector3<R>> for RqVector3<R>
{
    type Output = RqVector3<R>;

    #[inline(always)]
    fn sub(mut self, rhs: &'a RqVector3<R>) -> RqVector3<R>
    {
        self -= rhs;
        return self;
    }
}

impl<'a, R: RqElementCoefficientRepr> Sub<RqVector3<R>> for &'a RqVector3<R>
{
    type Output = RqVector3<R>;

    #[inline(always)]
    fn sub(self, mut rhs: RqVector3<R>) -> RqVector3<R>
    {
        rhs -= self;
        rhs *= -ONE;
        return rhs;
    }
}

impl<'a, R: RqElementCoefficientRepr> Mul<&'a RqVector3<R>> for &'a RqVector3<R>
{
    type Output = R::ChineseRemainderRepr;

    #[inline(always)]
    fn mul(self, rhs: &'a RqVector3<R>) -> R::ChineseRemainderRepr
    {
        let mut result = R::ChineseRemainderRepr::get_zero();
        for i in 0..DIM {
            result.add_product(&self.data[i], &rhs.data[i]);
        }
        return result;
    }
}

impl<'a, R: RqElementCoefficientRepr> Mul<&'a R::ChineseRemainderRepr> for RqVector3<R>
{
    type Output = RqVector3<R>;

    #[inline(always)]
    fn mul(mut self, rhs: &'a R::ChineseRemainderRepr) -> RqVector3<R>
    {
        self *= rhs;
        return self;
    }
}

impl<'a, R: RqElementCoefficientRepr> AddAssign<&'a RqVector3<R>> for RqVector3<R>
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a RqVector3<R>) 
    {
        for i in 0..DIM {
            self.data[i] += &rhs.data[i];
        }
    }
}

impl<'a, R: RqElementCoefficientRepr> SubAssign<&'a RqVector3<R>> for RqVector3<R>
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a RqVector3<R>) 
    {
        for i in 0..DIM {
            self.data[i] -= &rhs.data[i];
        }
    }
}

impl<'a, R: RqElementCoefficientRepr> MulAssign<&'a R::ChineseRemainderRepr> for RqVector3<R>
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a R::ChineseRemainderRepr) 
    {
        for i in 0..DIM {
            self.data[i] *= rhs;
        }
    }
}

impl<R: RqElementCoefficientRepr> MulAssign<ZqElement> for RqVector3<R>
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: ZqElement) 
    {
        for i in 0..DIM {
            self.data[i].mul_scalar(rhs);
        }
    }
}

impl<'a, R: RqElementCoefficientRepr> From<&'a [R::ChineseRemainderRepr]> for RqVector3<R>
{
    #[inline(always)]
    fn from(data: &'a [R::ChineseRemainderRepr]) -> RqVector3<R>
    {
        assert_eq!(DIM, data.len());
        Self::from(util::create_array(|i| data[i].clone()))
    }
}

impl<R: RqElementCoefficientRepr> From<[R::ChineseRemainderRepr; DIM]> for RqVector3<R>
{
    #[inline(always)]
    fn from(data: [R::ChineseRemainderRepr; DIM]) -> RqVector3<R>
    {
        RqVector3 {
            data: data
        }
    }
}

impl<R: RqElementCoefficientRepr> std::ops::Index<usize> for RqVector3<R>
{
    type Output = R::ChineseRemainderRepr;

    fn index(&self, i: usize) -> &Self::Output
    {
        &self.data[i]
    }
}

/// A dxd matrix over the given RqElementCoefficientRepr R, where d = DIM = 3.
#[derive(PartialEq, Eq, Clone)]
pub struct RqSquareMatrix3<R: RqElementCoefficientRepr>
{
    rows: [RqVector3<R>; DIM]
}

impl<R: RqElementCoefficientRepr> RqSquareMatrix3<R>
{
    pub fn transpose<'a>(&'a self) -> TransposedMat<'a, R>
    {
        TransposedMat {
            data: self
        }
    }
}

impl<'a, R: RqElementCoefficientRepr> Mul<&'a RqVector3<R>> for &'a RqSquareMatrix3<R>
{
    type Output = RqVector3<R>;

    #[inline(always)]
    fn mul(self, rhs: &'a RqVector3<R>) -> RqVector3<R> 
    {
        RqVector3 {
            data: [&self.rows[0] * rhs, &self.rows[1] * rhs, &self.rows[2] * rhs]
        }
    }
}

impl<R: RqElementCoefficientRepr> From<[[R::ChineseRemainderRepr; DIM]; DIM]> for RqSquareMatrix3<R>
{
    #[inline(always)]
    fn from(data: [[R::ChineseRemainderRepr; DIM]; DIM]) -> RqSquareMatrix3<R>
    {
        let [fst, snd, trd] = data;
        RqSquareMatrix3 {
            rows: [RqVector3::from(fst), RqVector3::from(snd), RqVector3::from(trd)]
        }
    }
}

/// A reference onto the transpose of a dxd matrix over the RqElementCoefficientRepr R.
#[derive(PartialEq, Eq, Clone)]
pub struct TransposedMat<'a, R: RqElementCoefficientRepr>
{
    data: &'a RqSquareMatrix3<R>
}

impl<'a, R: RqElementCoefficientRepr> TransposedMat<'a, R>
{
    pub fn transpose(&'a self) -> &'a RqSquareMatrix3<R>
    {
        self.data
    }
}

impl<'a, R: RqElementCoefficientRepr> Mul<&'a RqVector3<R>> for TransposedMat<'a, R>
{
    type Output = RqVector3<R>;

    #[inline(always)]
    fn mul(self, rhs: &'a RqVector3<R>) -> RqVector3<R>
    {
        let mut result: [R::ChineseRemainderRepr; DIM] = util::create_array(|_i| R::ChineseRemainderRepr::get_zero());
        for row in 0..DIM {
            for col in 0..DIM {
                result[row].add_product(&self.data.rows[col].data[row], &rhs.data[col]);
            }
        }
        return RqVector3 {
            data: result
        };
    }
}

#[derive(Debug, Clone)]
pub struct CompressedRqVector<const D: u16>
{
    data: [CompressedRq<D>; DIM]
}

impl<const D: u16> encoding::Encodable for CompressedRqVector<D>
{
    fn encode<T: encoding::Encoder>(&self, encoder: &mut T)
    {
        for element in &self.data {
            element.encode(encoder);
        }
    }

    fn decode<T: encoding::Decoder>(data: &mut T) -> Self
    {
        CompressedRqVector {
            data: util::create_array(|_i| CompressedRq::decode(data))
        }
    }
}

impl<T: RqElementCoefficientRepr> RqVector3<T>
{
    pub fn compress<const D: u16>(self) -> CompressedRqVector<D>
    {
        let [fst, snd, trd] = self.data;
        CompressedRqVector {
            data: [fst.to_coefficient_repr().compress(), 
                snd.to_coefficient_repr().compress(), 
                trd.to_coefficient_repr().compress()]
        }
    }

    pub fn decompress<const D: u16>(x: &CompressedRqVector<D>) -> RqVector3<T>
    {
        RqVector3 {
            data: [T::decompress(&x.data[0]).to_chinese_remainder_repr(),
                T::decompress(&x.data[1]).to_chinese_remainder_repr(), 
                T::decompress(&x.data[2]).to_chinese_remainder_repr()]
        }
    }
}