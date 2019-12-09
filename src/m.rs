use super::zq::*;
use super::r::*;

use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, DivAssign, SubAssign };
use std::convert::From;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct M 
{
    data: [FourierReprR; 3]
}


impl<'a> Add<&'a M> for M
{
    type Output = M;

    #[inline(always)]
    fn add(mut self, rhs: &'a M) -> M
    {
        self += rhs;
        return self;
    }
}

impl<'a> Add<M> for &'a M
{
    type Output = M;

    #[inline(always)]
    fn add(self, mut rhs: M) -> M
    {
        rhs += self;
        return rhs;
    }
}

impl<'a> Sub<&'a M> for M
{
    type Output = M;

    #[inline(always)]
    fn sub(mut self, rhs: &'a M) -> M
    {
        self -= rhs;
        return self;
    }
}

impl<'a> Sub<M> for &'a M
{
    type Output = M;

    #[inline(always)]
    fn sub(self, mut rhs: M) -> M
    {
        rhs -= self;
        rhs *= ZERO - ONE;
        return rhs;
    }
}

impl<'a> Mul<&'a M> for &'a M
{
    type Output = FourierReprR;

    #[inline(always)]
    fn mul(self, rhs: &'a M) -> FourierReprR
    {
        let mut result = FourierReprR::zero();
        for i in 0..3 {
            result.add_product(&self.data[i], &rhs.data[i]);
        }
        return result;
    }
}

impl<'a> Mul<&'a FourierReprR> for M
{
    type Output = M;

    #[inline(always)]
    fn mul(mut self, rhs: &'a FourierReprR) -> M
    {
        self *= rhs;
        return self;
    }
}

impl<'a> AddAssign<&'a M> for M
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a M) 
    {
        for i in 0..3 {
            self.data[i] += &rhs.data[i];
        }
    }
}

impl<'a> SubAssign<&'a M> for M
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a M) 
    {
        for i in 0..3 {
            self.data[i] -= &rhs.data[i];
        }
    }
}

impl<'a> MulAssign<&'a FourierReprR> for M
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a FourierReprR) 
    {
        for i in 0..3 {
            self.data[i] *= rhs;
        }
    }
}

impl MulAssign<Zq> for M
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq) 
    {
        for i in 0..3 {
            self.data[i] *= rhs;
        }
    }
}

impl<'a> DivAssign<&'a FourierReprR> for M
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: &'a FourierReprR) 
    {
        for i in 0..3 {
            self.data[i] /= rhs;
        }
    }
}

impl DivAssign<Zq> for M
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: Zq) 
    {
        for i in 0..3 {
            self.data[i] /= rhs;
        }
    }
}

impl<'a> From<&'a [FourierReprR; 3]> for M
{
    #[inline(always)]
    fn from(data: &'a [FourierReprR; 3]) -> M
    {
        Self::from(data.clone())
    }
}

impl From<[FourierReprR; 3]> for M
{
    #[inline(always)]
    fn from(data: [FourierReprR; 3]) -> M
    {
        M {
            data: data
        }
    }
}

impl From<[[FourierReprR; 3]; 3]> for Mat
{
    #[inline(always)]
    fn from(data: [[FourierReprR; 3]; 3]) -> Mat
    {
        let [fst, snd, trd] = data;
        Mat {
            rows: [M::from(fst), M::from(snd), M::from(trd)]
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Mat
{
    rows: [M; 3]
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct TransposedMat<'a>
{
    data: &'a Mat
}

impl Mat 
{
    pub fn transpose<'a>(&'a self) -> TransposedMat<'a>
    {
        TransposedMat {
            data: self
        }
    }
}

impl<'a> TransposedMat<'a>
{
    pub fn transpose(&'a self) -> &'a Mat
    {
        self.data
    }
}

impl<'a> Mul<&'a M> for &'a Mat
{
    type Output = M;

    #[inline(always)]
    fn mul(self, rhs: &'a M) -> M 
    {
        M {
            data: [&self.rows[0] * rhs, &self.rows[1] * rhs, &self.rows[2] * rhs]
        }
    }
}

impl<'a> Mul<&'a M> for TransposedMat<'a>
{
    type Output = M;

    #[inline(always)]
    fn mul(self, rhs: &'a M) -> M 
    {
        let mut result: [FourierReprR; 3] = [FourierReprR::zero(), FourierReprR::zero(), FourierReprR::zero()];
        for row in 0..3 {
            for col in 0..3 {
                result[row].add_product(&self.data.rows[col].data[row], &rhs.data[col]);
            }
        }
        return M {
            data: result
        };
    }
}

pub struct CompressedM<const D : u16>
{
    data: [CompressedR<D>; 3]
}

impl M
{
    pub fn compress<const D : u16>(self) -> CompressedM<D>
    {
        let [fst, snd, trd] = self.data;
        CompressedM {
            data: [FourierReprR::inv_dft(fst).compress(), 
                FourierReprR::inv_dft(snd).compress(), 
                FourierReprR::inv_dft(trd).compress()]
        }
    }
}

impl<const D : u16> CompressedM<D>
{
    pub fn decompress(&self) -> M
    {
        M {
            data: [FourierReprR::dft(self.data[0].decompress()), 
                FourierReprR::dft(self.data[1].decompress()), 
                FourierReprR::dft(self.data[2].decompress())]
        }
    }
}