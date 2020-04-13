use super::zq::*;
use super::avx_zq::*;
use super::ring::*;

use super::util;
use super::avx_util;
use super::encoding;

use std::arch::x86_64::*;

use std::ops::{ Add, Mul, Sub, Neg, AddAssign, MulAssign, SubAssign };
use std::cmp::{ PartialEq, Eq };
use std::convert::From;
use std::fmt::{ Formatter, Debug };

/// The count of Zq values in one Zq8 vector
const VEC_SIZE: usize = 8;
/// The count of Zq8 vectors we need to store all coefficients from one element in Rq = Zq[X] / (X^256 + 1)
const VEC_COUNT: usize = N / VEC_SIZE;

/// The ring Rq := Zq[X] / (X^256 + 1), using avx instructions for algebraic operations.
#[derive(Clone)]
pub struct RqElementCoefficientReprImpl
{
    data: [ZqVector8; VEC_COUNT]
}

impl PartialEq for RqElementCoefficientReprImpl
{
    fn eq(&self, rhs: &RqElementCoefficientReprImpl) -> bool
    {
        (0..VEC_COUNT).all(|i| self.data[i] == rhs.data[i])
    }
}

impl Eq for RqElementCoefficientReprImpl {}

impl<'a> Add<&'a RqElementCoefficientReprImpl> for RqElementCoefficientReprImpl
{
    type Output = RqElementCoefficientReprImpl;

    #[inline(always)]
    fn add(mut self, rhs: &'a RqElementCoefficientReprImpl) -> RqElementCoefficientReprImpl
    {
        self += rhs;
        return self;
    }
}

impl<'a> Add<RqElementCoefficientReprImpl> for &'a RqElementCoefficientReprImpl
{
    type Output = RqElementCoefficientReprImpl;

    #[inline(always)]
    fn add(self, mut rhs: RqElementCoefficientReprImpl) -> RqElementCoefficientReprImpl
    {
        rhs += self;
        return rhs;
    }
}

impl<'a> Sub<&'a RqElementCoefficientReprImpl> for RqElementCoefficientReprImpl
{
    type Output = RqElementCoefficientReprImpl;

    #[inline(always)]
    fn sub(mut self, rhs: &'a RqElementCoefficientReprImpl) -> Self::Output
    {
        self -= rhs;
        return self;
    }
}

impl<'a> Sub<RqElementCoefficientReprImpl> for &'a RqElementCoefficientReprImpl
{
    type Output = RqElementCoefficientReprImpl;

    #[inline(always)]
    fn sub(self, mut rhs: RqElementCoefficientReprImpl) -> Self::Output
    {
        rhs -= self;
        return -rhs;
    }
}

impl Mul<ZqElement> for RqElementCoefficientReprImpl
{
    type Output = RqElementCoefficientReprImpl;

    #[inline(always)]
    fn mul(mut self, rhs: ZqElement) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

impl Neg for RqElementCoefficientReprImpl
{
    type Output = RqElementCoefficientReprImpl;
    
    #[inline(always)]
    fn neg(mut self) -> Self::Output
    {
        for i in 0..VEC_COUNT {
            self.data[i] = -self.data[i];
        }
        return self;
    }
}

impl<'a> AddAssign<&'a RqElementCoefficientReprImpl> for RqElementCoefficientReprImpl
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a RqElementCoefficientReprImpl)
    {
        for i in 0..VEC_COUNT {
            self.data[i] += rhs.data[i];
        }
    }
}

impl<'a> SubAssign<&'a RqElementCoefficientReprImpl> for RqElementCoefficientReprImpl
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a RqElementCoefficientReprImpl)
    {
        for i in 0..VEC_COUNT {
            self.data[i] -= rhs.data[i];
        }
    }
}

impl MulAssign<ZqElement> for RqElementCoefficientReprImpl
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: ZqElement)
    {
        for i in 0..VEC_COUNT {
            self.data[i] *= rhs;
        }
    }
}

impl<'a> From<&'a [i16]> for RqElementCoefficientReprImpl
{
    fn from(value: &'a [i16]) -> RqElementCoefficientReprImpl 
    {
        assert_eq!(N, value.len());
        return RqElementCoefficientReprImpl::from(util::create_array(|i| 
            ZqVector8::from(&value[i * VEC_SIZE..(i+1) * VEC_SIZE])
        ));
    }
}

impl From<[ZqElement; N]> for RqElementCoefficientReprImpl
{
    fn from(value: [ZqElement; N]) -> RqElementCoefficientReprImpl 
    {
        assert_eq!(N, value.len());
        return RqElementCoefficientReprImpl::from(util::create_array(|i| 
            ZqVector8::from(&value[i * VEC_SIZE..(i+1) * VEC_SIZE])
        ));
    }
}

impl From<[ZqVector8; VEC_COUNT]> for RqElementCoefficientReprImpl
{
    #[inline(always)]
    fn from(data: [ZqVector8; VEC_COUNT]) -> RqElementCoefficientReprImpl
    {
        RqElementCoefficientReprImpl {
            data: data
        }
    }
}

impl Debug for RqElementCoefficientReprImpl
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[")?;
        (0..31).try_for_each(|i| write!(f, "{:?}, ", self.data[i]))?;
        write!(f, "{:?}]", self.data[31])?;
        return Ok(());
    }
}

impl RqElementCoefficientRepr for RqElementCoefficientReprImpl
{
    type ChineseRemainderRepr = RqElementChineseRemainderReprImpl;

    fn get_zero() -> RqElementCoefficientReprImpl
    {
        return RqElementCoefficientReprImpl {
            data: [ZqVector8::zero(); VEC_COUNT]
        }
    }

    fn to_chinese_remainder_repr(self) -> RqElementChineseRemainderReprImpl
    {
        RqElementChineseRemainderReprImpl::chinese_remainder_repr(self)
    }

    fn compress<const D: u16>(&self) -> CompressedRq<D>
    {
        unsafe {
            let data: [__m256i; VEC_COUNT] = util::create_array(|i| {
                let element: CompressedZq8<D> = self.data[i].compress();
                return element.data;
            });
            let compressed_values = avx_util::decompose::<32, 256>(data);
            CompressedRq {
                data: util::create_array(|i| CompressedZq { data: compressed_values[i] as u16 })
            }
        }
    }

    fn decompress<const D: u16>(x: &CompressedRq<D>) -> RqElementCoefficientReprImpl
    {
        let data: [i32; N] = util::create_array(|i| x.data[i].data as i32);
        let vectorized_data = unsafe {
            avx_util::compose::<N, VEC_COUNT>(data)
        };
        RqElementCoefficientReprImpl {
            data: util::create_array(|i| {
                let compressed: CompressedZq8<D> = CompressedZq8 { data:  vectorized_data[i] };
                ZqVector8::decompress(compressed)
            })
        }
    }
}

/// Chinese remainder representation of an element of Rq, i.e.
/// the values of the polynomial at each root of unity
/// in Zq
#[derive(Clone)]
pub struct RqElementChineseRemainderReprImpl
{
    values: [ZqVector8; VEC_COUNT]
}

impl RqElementChineseRemainderReprImpl
{
    /// Executes the i-th step in the Cooley–Tukey FFT algorithm. Concretely, calculates the DFT
    /// of x_j, x_j+d, ..., x_j+(n-1)d for each j in 0..d and each k in 0..n where d=N/n and
    /// n = 2^i. For more detail, see ref_r::NTTDomainRq::fft.
    /// 
    /// As input, we require the DFTs of x_j, x_j+2d, ..., x_j+(n-1)4d to be stored in src[k * 2d + j] 
    /// for each j in 0..2d and each k in 0..n/2.  The result DFT of x_j, x_j+d, ..., x_j+(n-1)d for k in 0..n
    /// will be stored in dst[k * d + j]. Therefore, this is exactly fft_iter_dxn() except that in-and output
    /// are transposed.
    /// 
    /// The given function should, given l, return the 256-th root of unity to the power of l
    /// (or to the power of -l in case of an inverse DFT).
    #[inline(always)]
    fn fft_iter_nxd<F>(dst: &mut [ZqVector8; VEC_COUNT], src: &[ZqVector8; VEC_COUNT], i: usize, unity_root: F)
        where F: Fn(usize) -> ZqElement
    {
        let n = 1 << i;
        let d = 1 << (8 - i);
        let old_d = d << 1;
        let old_n = n >> 1;
        let d_vec = d / VEC_SIZE;
        let old_d_vec = old_d / VEC_SIZE;

        for k in 0..old_n {
            let unity_root = ZqVector8::broadcast(unity_root(k * d));
            for j in 0..d_vec {
                dst[k * d_vec + j] = src[k * old_d_vec + j] + src[k * old_d_vec + j + d_vec] * unity_root;
                dst[(k + old_n) * d_vec + j] = src[k * old_d_vec + j] - src[k * old_d_vec + j + d_vec] * unity_root;
            }
        }
    }

    /// Executes the i-th step in the Cooley–Tukey FFT algorithm. Concretely, calculates the DFT
    /// of x_j, x_j+d, ..., x_j+(n-1)d for each j in 0..d and each k in 0..n where d=N/n and
    /// n = 2^i. For more detail, see ref_r::NTTDomainRq::fft.
    /// 
    /// As input, we require the DFTs of x_j, x_j+2d, ..., x_j+(n-1)4d to be stored in src[j * n/2 + k] 
    /// for each j in 0..2d and each k in 0..n/2.  The result DFT of x_j, x_j+d, ..., x_j+(n-1)d for k in 0..n
    /// will be stored in dst[j * n + k]. Therefore, this is exactly fft_iter_nxd() except that in-and output
    /// are transposed.
    /// 
    /// The given function should, given l, return the 256-th root of unity to the power of l
    /// (or to the power of -l in case of an inverse DFT).
    #[inline(always)]
    fn fft_iter_dxn<F>(dst: &mut [ZqVector8; VEC_COUNT], src: &[ZqVector8; VEC_COUNT], i: usize, unity_root: F)
        where F: Fn(usize) -> ZqElement
    {
        let d = 1 << (8 - i);
        let n = 1 << i;
        let old_n = n >> 1;
        let n_vec = n / VEC_SIZE;
        let old_n_vec = old_n / VEC_SIZE;

        for vec_k in 0..old_n_vec {
            let unity_root = ZqVector8::from(util::create_array(|dk| unity_root((vec_k * VEC_SIZE + dk) * d)));
            for j in 0..d {
                dst[j * n_vec + vec_k] = src[j * old_n_vec + vec_k] + src[(j + d) * old_n_vec + vec_k] * unity_root;
                dst[j * n_vec + vec_k + old_n_vec] = src[j * old_n_vec + vec_k] - src[(j + d) * old_n_vec + vec_k] * unity_root;
            }
        }
    }

    #[inline(never)]
    fn fft<F>(mut values: [ZqVector8; VEC_COUNT], unity_root: F) -> [ZqVector8; VEC_COUNT]
        where F: Fn(usize) -> ZqElement
    {
        // see ref_r::NTTDomainRq::fft for details on what happens
        let mut temp: [ZqVector8; VEC_COUNT] = [ZqVector8::zero(); VEC_COUNT];

        Self::fft_iter_nxd(&mut temp, &values, 1, &unity_root);
        Self::fft_iter_nxd(&mut values, &temp, 2, &unity_root);
        Self::fft_iter_nxd(&mut temp, &values, 3, &unity_root);
        Self::fft_iter_nxd(&mut values, &temp, 4, &unity_root);

        values = transpose_vectorized_matrix::<16, 32>(values);

        Self::fft_iter_dxn(&mut temp, &values, 5, &unity_root);
        Self::fft_iter_dxn(&mut values, &temp, 6, &unity_root);
        Self::fft_iter_dxn(&mut temp, &values, 7, &unity_root);
        Self::fft_iter_dxn(&mut values, &temp, 8, &unity_root);

        return values;
    }

    #[inline(always)]
    fn chinese_remainder_repr(mut r: RqElementCoefficientReprImpl) -> RqElementChineseRemainderReprImpl
    {
        // we do not need the exact fourier transformation (i.e. the evaluation at
        // all 256-th roots of unity), but the evaluation at all primitive 512-th
        // roots of unity. Since the primitive 512-th roots of unity are exactly
        // the 256-th roots of unity multiplied with any primitive root of unity,
        // this approach lets us calculate the correct result
        for i in 0..VEC_COUNT {
            r.data[i] *= ZqVector8::from(&UNITY_ROOTS_512[VEC_SIZE * i..VEC_SIZE * i + VEC_SIZE])
        }
        RqElementChineseRemainderReprImpl {
            values: Self::fft(r.data, |i| UNITY_ROOTS_512[2 * i])
        }
    }

    #[inline(always)]
    fn coefficient_repr(ntt_repr: RqElementChineseRemainderReprImpl) -> RqElementCoefficientReprImpl
    {
        let inv_n: ZqElement = ONE / ZqElement::from(N as i16);
        let mut result = Self::fft(ntt_repr.values, |i| REV_UNITY_ROOTS_512[2 * i]);
        for i in 0..VEC_COUNT {
            // see dft for why this is necessary (we do not do a real fourier transformation)
            result[i] *= ZqVector8::from(&REV_UNITY_ROOTS_512[VEC_SIZE * i..VEC_SIZE * i + VEC_SIZE]);
            result[i] *= inv_n;
        }
        return RqElementCoefficientReprImpl {
            data: result
        };
    }
}

impl PartialEq for RqElementChineseRemainderReprImpl
{
    fn eq(&self, rhs: &RqElementChineseRemainderReprImpl) -> bool
    {
        (0..VEC_COUNT).all(|i| self.values[i] == rhs.values[i])
    }
}

impl Eq for RqElementChineseRemainderReprImpl {}

impl<'a> Add<&'a RqElementChineseRemainderReprImpl> for RqElementChineseRemainderReprImpl
{
    type Output = RqElementChineseRemainderReprImpl;

    #[inline(always)]
    fn add(mut self, rhs: &'a RqElementChineseRemainderReprImpl) -> Self::Output
    {
        self += rhs;
        return self;
    }
}

impl<'a> Add<RqElementChineseRemainderReprImpl> for &'a RqElementChineseRemainderReprImpl
{
    type Output = RqElementChineseRemainderReprImpl;

    #[inline(always)]
    fn add(self, mut rhs: RqElementChineseRemainderReprImpl) -> Self::Output
    {
        rhs += self;
        return rhs;
    }
}

impl<'a> Mul<&'a RqElementChineseRemainderReprImpl> for RqElementChineseRemainderReprImpl
{
    type Output = RqElementChineseRemainderReprImpl;

    #[inline(always)]
    fn mul(mut self, rhs: &'a RqElementChineseRemainderReprImpl) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

impl<'a> Mul<RqElementChineseRemainderReprImpl> for &'a RqElementChineseRemainderReprImpl
{
    type Output = RqElementChineseRemainderReprImpl;

    #[inline(always)]
    fn mul(self, mut rhs: RqElementChineseRemainderReprImpl) -> Self::Output
    {
        rhs *= self;
        return rhs;
    }
}

impl Mul<ZqElement> for RqElementChineseRemainderReprImpl
{
    type Output = RqElementChineseRemainderReprImpl;

    #[inline(always)]
    fn mul(mut self, rhs: ZqElement) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

impl<'a> Sub<&'a RqElementChineseRemainderReprImpl> for RqElementChineseRemainderReprImpl
{
    type Output = RqElementChineseRemainderReprImpl;

    #[inline(always)]
    fn sub(mut self, rhs: &'a RqElementChineseRemainderReprImpl) -> Self::Output
    {
        self -= rhs;
        return self;
    }
}

impl<'a> Sub<RqElementChineseRemainderReprImpl> for &'a RqElementChineseRemainderReprImpl
{
    type Output = RqElementChineseRemainderReprImpl;

    #[inline(always)]
    fn sub(self, mut rhs: RqElementChineseRemainderReprImpl) -> Self::Output
    {
        rhs -= self;
        rhs *= ZERO - ONE;
        return rhs;
    }
}

impl<'a> AddAssign<&'a RqElementChineseRemainderReprImpl> for RqElementChineseRemainderReprImpl
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a RqElementChineseRemainderReprImpl) {
        for i in 0..VEC_COUNT {
            self.values[i] += rhs.values[i];
        }
    }
}

impl<'a> SubAssign<&'a RqElementChineseRemainderReprImpl> for RqElementChineseRemainderReprImpl
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a RqElementChineseRemainderReprImpl) {
        for i in 0..VEC_COUNT {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl<'a> MulAssign<&'a RqElementChineseRemainderReprImpl> for RqElementChineseRemainderReprImpl
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a RqElementChineseRemainderReprImpl) {
        for i in 0..VEC_COUNT {
            self.values[i] *= rhs.values[i];
        }
    }
}

impl MulAssign<ZqElement> for RqElementChineseRemainderReprImpl
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: ZqElement) {
        for i in 0..VEC_COUNT {
            self.values[i] *= rhs;
        }
    }
}

impl<'a> From<&'a [i16]> for RqElementChineseRemainderReprImpl
{
    fn from(value: &'a [i16]) -> RqElementChineseRemainderReprImpl 
    {
        assert_eq!(N, value.len());
        return RqElementChineseRemainderReprImpl::from(util::create_array(|i| 
            ZqVector8::from(&value[i * VEC_SIZE..(i+1) * VEC_SIZE])
        ));
    }
}

impl From<[ZqElement; N]> for RqElementChineseRemainderReprImpl
{
    fn from(value: [ZqElement; N]) -> RqElementChineseRemainderReprImpl 
    {
        assert_eq!(N, value.len());
        return RqElementChineseRemainderReprImpl::from(util::create_array(|i| 
            ZqVector8::from(&value[i * VEC_SIZE..(i+1) * VEC_SIZE])
        ));
    }
}

impl From<[ZqVector8; VEC_COUNT]> for RqElementChineseRemainderReprImpl
{
    #[inline(always)]
    fn from(data: [ZqVector8; VEC_COUNT]) -> RqElementChineseRemainderReprImpl
    {
        RqElementChineseRemainderReprImpl {
            values: data
        }
    }
}

impl Debug for RqElementChineseRemainderReprImpl
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[")?;
        (0..31).try_for_each(|i| write!(f, "{:?}, ", self.values[i]))?;
        write!(f, "{:?}]", self.values[31])?;
        return Ok(());
    }
}

impl encoding::Encodable for RqElementChineseRemainderReprImpl
{
    fn encode<T: encoding::Encoder>(&self, encoder: &mut T)
    {
        for vector in &self.values {
            for element in &vector.as_array() {
                encoder.encode_bits(element.representative_pos() as u16, 16);
            }
        }
    }

    fn decode<T: encoding::Decoder>(data: &mut T) -> Self
    {
        RqElementChineseRemainderReprImpl {
            values: util::create_array(|_i| ZqVector8::from(util::create_array(|_j| data.read_bits(16).expect("Input too short") as i16)))
        }
    }
}

impl RqElementChineseRemainderRepr for RqElementChineseRemainderReprImpl
{
    type CoefficientRepr = RqElementCoefficientReprImpl;
    
    fn get_zero() -> RqElementChineseRemainderReprImpl
    {
        return RqElementChineseRemainderReprImpl {
            values: [ZqVector8::zero(); VEC_COUNT]
        }
    }

    fn add_product(&mut self, fst: &RqElementChineseRemainderReprImpl, snd: &RqElementChineseRemainderReprImpl) 
    {
        for i in 0..VEC_COUNT {
            self.values[i] += fst.values[i] * snd.values[i];
        }
    }

    fn mul_scalar(&mut self, x: ZqElement)
    {
        let broadcast_x = ZqVector8::broadcast(x);
        for i in 0..VEC_COUNT {
            self.values[i] *= broadcast_x;
        }
    }

    fn to_coefficient_repr(self) -> RqElementCoefficientReprImpl
    {
        RqElementChineseRemainderReprImpl::coefficient_repr(self)
    }

    fn value_at_zeta(&self, zeta_index: usize) -> ZqElement
    {
        self.values[zeta_index/8].as_array()[zeta_index % 8]
    }
}

#[cfg(test)]
const ELEMENT: [i16; N] = [5487, 7048, 1145, 6716, 88, 5957, 3742, 3441, 2663, 
    1301, 159, 4074, 2945, 6671, 1392, 3999, 2394, 7624, 2420, 4199, 2762, 4206, 4471, 1582, 
    3870, 5363, 4246, 1800, 4568, 2081, 5642, 1115, 1242, 704, 2348, 6823, 6135, 854, 3320, 
    2929, 6417, 7368, 535, 1491, 7271, 7666, 1256, 6093, 4767, 3442, 6055, 2757, 3953, 7391, 
    4429, 6526, 201, 5915, 5354, 6748, 425, 218, 5931, 2527, 20, 7017, 1235, 178, 5103, 1865, 
    1496, 3497, 6851, 5004, 2292, 1957, 5277, 1628, 5900, 5431, 1825, 1634, 4443, 3351, 1068, 
    1403, 657, 7428, 2085, 6387, 5712, 4364, 3339, 1917, 3655, 4328, 499, 5021, 5403, 3460, 
    6265, 1904, 6666, 2154, 3190, 3462, 4137, 4457, 2013, 1464, 4097, 6356, 2234, 2539, 3252, 
    7075, 3947, 5, 4724, 314, 5482, 120, 5968, 7268, 254, 2207, 5042, 5695, 3925, 1194, 6921, 
    7100, 6643, 2183, 2890, 535, 617, 4989, 5494, 4149, 2964, 3783, 6901, 2763, 6564, 6869, 
    5218, 2295, 4529, 6211, 1290, 4612, 3468, 1799, 2705, 2247, 5333, 703, 1287, 6690, 5906, 
    6011, 7655, 3022, 1544, 1152, 2740, 105, 7433, 7222, 3424, 4571, 7224, 4290, 5396, 5584, 
    6049, 826, 4647, 4640, 4674, 7317, 6580, 5295, 4560, 6353, 630, 3316, 6038, 3563, 1174, 
    940, 7458, 1966, 5348, 487, 3041, 6107, 1259, 5148, 2209, 6494, 7085, 5829, 2842, 5850, 
    4680, 5056, 5995, 5097, 1030, 2778, 554, 843, 4938, 7053, 6170, 5482, 408, 6923, 3935, 
    1488, 3311, 7459, 194, 4278, 5930, 1964, 4158, 2466, 7485, 2940, 1244, 4056, 5828, 3270, 
    1303, 2724, 1032, 2068, 1912, 7030, 7679, 1308, 1754, 330, 3715, 1865, 4588, 4813, 727, 
    6881, 1026, 4981, 3325, 4511];

#[bench]
fn bench_ntt(bencher: &mut test::Bencher) {
    let element = RqElementCoefficientReprImpl::from(&ELEMENT[..]);
    bencher.iter(|| {
        let ntt_repr = RqElementChineseRemainderReprImpl::chinese_remainder_repr(element.clone());
        assert_eq!(element, ntt_repr.to_coefficient_repr());
    });
}

#[test]
fn test_scalar_mul_div() {
    let mut element = RqElementCoefficientReprImpl::from(&ELEMENT[..]);
    let mut ntt_repr = RqElementChineseRemainderReprImpl::chinese_remainder_repr(element.clone());
    element *= ZqElement::from(653_i16);
    ntt_repr *= ZqElement::from(653_i16);
    assert_eq!(element, ntt_repr.clone().to_coefficient_repr());

    element *= ONE / ZqElement::from(5321_i16);
    ntt_repr *= ONE / ZqElement::from(5321_i16);
    assert_eq!(element, ntt_repr.to_coefficient_repr());
}

#[test]
fn test_add_sub() {
    let mut element = RqElementCoefficientReprImpl::from(&ELEMENT[..]);
    let mut ntt_repr = RqElementChineseRemainderReprImpl::chinese_remainder_repr(element.clone());
    let base_element = element.clone();
    let base_ntt_repr = ntt_repr.clone();

    element += &base_element;
    ntt_repr += &base_ntt_repr;
    assert_eq!(element, RqElementChineseRemainderReprImpl::coefficient_repr(ntt_repr.clone()));

    element -= &base_element;
    ntt_repr -= &base_ntt_repr;
    assert_eq!(element, RqElementChineseRemainderReprImpl::coefficient_repr(ntt_repr));
    assert_eq!(RqElementCoefficientReprImpl::from(&ELEMENT[..]), element);
}

#[test]
fn test_mul() {
    let mut data: [ZqElement; 256] = [ZERO; 256];
    data[128] = ONE;
    let element = RqElementCoefficientReprImpl::from(data);
    let ntt_repr = element.clone().to_chinese_remainder_repr() * &element.to_chinese_remainder_repr();

    let mut expected: [ZqElement; 256] = [ZERO; 256];
    expected[0] = -ONE;
    assert_eq!(RqElementCoefficientReprImpl::from(expected), ntt_repr.to_coefficient_repr());
}

#[test]
fn test_compress() {
    let mut element = RqElementCoefficientReprImpl::from(&ELEMENT[..]);
    let compressed: CompressedRq<3_u16> = element.compress();
    element = RqElementCoefficientReprImpl::decompress(&compressed);
    assert_eq!(ZqVector8::from(&[5761, 6721, 960, 6721, 0, 5761, 3840, 3840][..]), element.data[0]);
}