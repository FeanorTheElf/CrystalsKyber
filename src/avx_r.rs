use super::zq::*;
use super::avx_zq::*;
use super::ring::*;

use super::util;
use super::avx_util;
use super::base64;

use std::arch::x86_64::*;

use std::ops::{ Add, Mul, Sub, Neg, AddAssign, MulAssign, SubAssign };
use std::cmp::{ PartialEq, Eq };
use std::convert::From;
use std::fmt::{ Formatter, Debug };

const VEC_SIZE: usize = 8;
const VEC_COUNT: usize = N / VEC_SIZE;

// Type of elements in the ring R := Zq[X] / (X^256 + 1)
#[derive(Clone)]
pub struct Rq
{
    data: [Zq8; VEC_COUNT]
}

impl PartialEq for Rq
{
    fn eq(&self, rhs: &Rq) -> bool
    {
        (0..VEC_COUNT).all(|i| self.data[i] == rhs.data[i])
    }
}

impl Eq for Rq {}

impl<'a> Add<&'a Rq> for Rq
{
    type Output = Rq;

    #[inline(always)]
    fn add(mut self, rhs: &'a Rq) -> Rq
    {
        self += rhs;
        return self;
    }
}

impl<'a> Add<Rq> for &'a Rq
{
    type Output = Rq;

    #[inline(always)]
    fn add(self, mut rhs: Rq) -> Rq
    {
        rhs += self;
        return rhs;
    }
}

impl<'a> Sub<&'a Rq> for Rq
{
    type Output = Rq;

    #[inline(always)]
    fn sub(mut self, rhs: &'a Rq) -> Self::Output
    {
        self -= rhs;
        return self;
    }
}

impl<'a> Sub<Rq> for &'a Rq
{
    type Output = Rq;

    #[inline(always)]
    fn sub(self, mut rhs: Rq) -> Self::Output
    {
        rhs -= self;
        return -rhs;
    }
}

impl Mul<Zq> for Rq
{
    type Output = Rq;

    #[inline(always)]
    fn mul(mut self, rhs: Zq) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

impl Neg for Rq
{
    type Output = Rq;
    
    #[inline(always)]
    fn neg(mut self) -> Self::Output
    {
        for i in 0..VEC_COUNT {
            self.data[i] = -self.data[i];
        }
        return self;
    }
}

impl<'a> AddAssign<&'a Rq> for Rq
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Rq)
    {
        for i in 0..VEC_COUNT {
            self.data[i] += rhs.data[i];
        }
    }
}

impl<'a> SubAssign<&'a Rq> for Rq
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Rq)
    {
        for i in 0..VEC_COUNT {
            self.data[i] -= rhs.data[i];
        }
    }
}

impl MulAssign<Zq> for Rq
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq)
    {
        for i in 0..VEC_COUNT {
            self.data[i] *= rhs;
        }
    }
}

impl<'a> From<&'a [i16]> for Rq
{
    fn from(value: &'a [i16]) -> Rq 
    {
        assert_eq!(N, value.len());
        return Rq::from(util::create_array(|i| 
            Zq8::from(&value[i * VEC_SIZE..(i+1) * VEC_SIZE])
        ));
    }
}

impl<'a> From<[Zq; N]> for Rq
{
    fn from(value: [Zq; N]) -> Rq 
    {
        return Rq::from(util::create_array(|i| 
            Zq8::from(&value[i * VEC_SIZE..(i+1) * VEC_SIZE])
        ));
    }
}

impl From<[Zq8; VEC_COUNT]> for Rq
{
    #[inline(always)]
    fn from(data: [Zq8; VEC_COUNT]) -> Rq
    {
        Rq {
            data: data
        }
    }
}

impl Debug for Rq
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[")?;
        (0..31).try_for_each(|i| write!(f, "{:?}, ", self.data[i]))?;
        write!(f, "{:?}]", self.data[31])?;
        return Ok(());
    }
}

impl Ring for Rq
{
    type FourierRepr = FourierReprRq;

    fn zero() -> Rq
    {
        return Rq {
            data: [Zq8::zero(); VEC_COUNT]
        }
    }

    fn dft(self) -> FourierReprRq
    {
        FourierReprRq::dft(self)
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

    fn decompress<const D: u16>(x: &CompressedRq<D>) -> Rq
    {
        let data: [i32; N] = util::create_array(|i| x.data[i].data as i32);
        let vectorized_data = unsafe {
            avx_util::compose::<N, VEC_COUNT>(data)
        };
        Rq {
            data: util::create_array(|i| {
                let compressed: CompressedZq8<D> = CompressedZq8 { data:  vectorized_data[i] };
                Zq8::decompress(compressed)
            })
        }
    }
}

// Fourier representation of an element of R, i.e.
// the values of the polynomial at each root of unity
// in Zq
#[derive(Clone)]
pub struct FourierReprRq
{
    values: [Zq8; VEC_COUNT]
}

impl FourierReprRq
{

    #[inline(always)]
    fn fft_iter_nxd<F>(dst: &mut [Zq8; VEC_COUNT], src: &[Zq8; VEC_COUNT], i: usize, unity_root: F)
        where F: Fn(usize) -> Zq
    {
        let n = 1 << i;
        let d = 1 << (8 - i);
        let old_d = d << 1;
        let old_n = n >> 1;
        let d_vec = d / VEC_SIZE;
        let old_d_vec = old_d / VEC_SIZE;

        for k in 0..old_n {
            let unity_root = Zq8::broadcast(unity_root(k * d));
            for j in 0..d_vec {
                dst[k * d_vec + j] = src[k * old_d_vec + j] + src[k * old_d_vec + j + d_vec] * unity_root;
                dst[(k + old_n) * d_vec + j] = src[k * old_d_vec + j] - src[k * old_d_vec + j + d_vec] * unity_root;
            }
        }
    }

    #[inline(always)]
    fn fft_iter_dxn<F>(dst: &mut [Zq8; VEC_COUNT], src: &[Zq8; VEC_COUNT], i: usize, unity_root: F)
        where F: Fn(usize) -> Zq
    {
        let d = 1 << (8 - i);
        let n = 1 << i;
        let old_n = n >> 1;
        let n_vec = n / VEC_SIZE;
        let old_n_vec = old_n / VEC_SIZE;

        for vec_k in 0..old_n_vec {
            let unity_root = Zq8::from(util::create_array(|dk| unity_root((vec_k * VEC_SIZE + dk) * d)));
            for j in 0..d {
                dst[j * n_vec + vec_k] = src[j * old_n_vec + vec_k] + src[(j + d) * old_n_vec + vec_k] * unity_root;
                dst[j * n_vec + vec_k + old_n_vec] = src[j * old_n_vec + vec_k] - src[(j + d) * old_n_vec + vec_k] * unity_root;
            }
        }
    }

    #[inline(never)]
    fn fft<F>(mut values: [Zq8; VEC_COUNT], unity_root: F) -> [Zq8; VEC_COUNT]
        where F: Fn(usize) -> Zq
    {
        // see ref_r::FourierReprR::fft for details on what happens
        let mut temp: [Zq8; VEC_COUNT] = [Zq8::zero(); VEC_COUNT];

        Self::fft_iter_nxd(&mut temp, &values, 1, &unity_root);
        Self::fft_iter_nxd(&mut values, &temp, 2, &unity_root);
        Self::fft_iter_nxd(&mut temp, &values, 3, &unity_root);
        Self::fft_iter_nxd(&mut values, &temp, 4, &unity_root);

        // temp corresponds to nxd = 32x8 array
        values = transpose_vectorized_matrix::<16, 32>(values);
        // temp corresponds to dxn = 8x32 array

        Self::fft_iter_dxn(&mut temp, &values, 5, &unity_root);
        Self::fft_iter_dxn(&mut values, &temp, 6, &unity_root);
        Self::fft_iter_dxn(&mut temp, &values, 7, &unity_root);
        Self::fft_iter_dxn(&mut values, &temp, 8, &unity_root);

        return values;
    }

    // Calculates the fourier representation of an element in R
    #[inline(always)]
    fn dft(r: Rq) -> FourierReprRq
    {
        FourierReprRq {
            values: Self::fft(r.data, |i| UNITY_ROOTS[i])
        }
    }

    // Calculates the element in R with the given fourier representation
    #[inline(always)]
    fn inv_dft(fourier_repr: FourierReprRq) -> Rq
    {
        let inv_n: Zq = ONE / Zq::from(N as i16);
        let mut result = Self::fft(fourier_repr.values, |i| UNITY_ROOTS[(N - i) & 0xFF]);
        for i in 0..VEC_COUNT {
            result[i] *= inv_n;
        }
        return Rq {
            data: result
        };
    }
}

impl PartialEq for FourierReprRq
{
    fn eq(&self, rhs: &FourierReprRq) -> bool
    {
        (0..VEC_COUNT).all(|i| self.values[i] == rhs.values[i])
    }
}

impl Eq for FourierReprRq {}

impl<'a> Add<&'a FourierReprRq> for FourierReprRq
{
    type Output = FourierReprRq;

    #[inline(always)]
    fn add(mut self, rhs: &'a FourierReprRq) -> Self::Output
    {
        self += rhs;
        return self;
    }
}

impl<'a> Add<FourierReprRq> for &'a FourierReprRq
{
    type Output = FourierReprRq;

    #[inline(always)]
    fn add(self, mut rhs: FourierReprRq) -> Self::Output
    {
        rhs += self;
        return rhs;
    }
}

impl<'a> Mul<&'a FourierReprRq> for FourierReprRq
{
    type Output = FourierReprRq;

    #[inline(always)]
    fn mul(mut self, rhs: &'a FourierReprRq) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

impl<'a> Mul<FourierReprRq> for &'a FourierReprRq
{
    type Output = FourierReprRq;

    #[inline(always)]
    fn mul(self, mut rhs: FourierReprRq) -> Self::Output
    {
        rhs *= self;
        return rhs;
    }
}

impl Mul<Zq> for FourierReprRq
{
    type Output = FourierReprRq;

    #[inline(always)]
    fn mul(mut self, rhs: Zq) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

impl<'a> Sub<&'a FourierReprRq> for FourierReprRq
{
    type Output = FourierReprRq;

    #[inline(always)]
    fn sub(mut self, rhs: &'a FourierReprRq) -> Self::Output
    {
        self -= rhs;
        return self;
    }
}

impl<'a> Sub<FourierReprRq> for &'a FourierReprRq
{
    type Output = FourierReprRq;

    #[inline(always)]
    fn sub(self, mut rhs: FourierReprRq) -> Self::Output
    {
        rhs -= self;
        rhs *= ZERO - ONE;
        return rhs;
    }
}

impl<'a> AddAssign<&'a FourierReprRq> for FourierReprRq
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a FourierReprRq) {
        for i in 0..VEC_COUNT {
            self.values[i] += rhs.values[i];
        }
    }
}

impl<'a> SubAssign<&'a FourierReprRq> for FourierReprRq
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a FourierReprRq) {
        for i in 0..VEC_COUNT {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl<'a> MulAssign<&'a FourierReprRq> for FourierReprRq
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a FourierReprRq) {
        for i in 0..VEC_COUNT {
            self.values[i] *= rhs.values[i];
        }
    }
}

impl MulAssign<Zq> for FourierReprRq
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq) {
        for i in 0..VEC_COUNT {
            self.values[i] *= rhs;
        }
    }
}

impl<'a> From<&'a [i16]> for FourierReprRq
{
    fn from(value: &'a [i16]) -> FourierReprRq {
        assert_eq!(N, value.len());
        return FourierReprRq {
            values: util::create_array(|i| 
                Zq8::from(&value[i * VEC_SIZE..(i+1) * VEC_SIZE])
            )
        };
    }
}

impl<'a> From<&'a [Zq]> for FourierReprRq
{
    fn from(value: &'a [Zq]) -> FourierReprRq {
        assert_eq!(N, value.len());
        return FourierReprRq {
            values: util::create_array(|i| 
                Zq8::from(&value[i * VEC_SIZE..(i+1) * VEC_SIZE])
            )
        };
    }
}

impl Debug for FourierReprRq
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[")?;
        (0..31).try_for_each(|i| write!(f, "{:?}, ", self.values[i]))?;
        write!(f, "{:?}]", self.values[31])?;
        return Ok(());
    }
}

impl base64::Encodable for FourierReprRq
{
    fn encode(&self, encoder: &mut base64::Encoder)
    {
        for vector in &self.values {
            for element in &vector.as_array() {
                encoder.encode_bits(element.representative_pos() as u16, 16);
            }
        }
    }

    fn decode(data: &mut base64::Decoder) -> base64::Result<Self>
    {
        Ok(FourierReprRq {
            values: util::try_create_array(|_i| Ok(Zq8::from(util::try_create_array(|_j| Ok(data.read_bits(16)? as i16))?)))?
        })
    }
}

impl RingFourierRepr for FourierReprRq
{
    type StdRepr = Rq;
    
    fn zero() -> FourierReprRq
    {
        return FourierReprRq {
            values: [Zq8::zero(); VEC_COUNT]
        }
    }

    fn add_product(&mut self, fst: &FourierReprRq, snd: &FourierReprRq) 
    {
        for i in 0..VEC_COUNT {
            self.values[i] += fst.values[i] * snd.values[i];
        }
    }

    fn mul_scalar(&mut self, x: Zq)
    {
        let broadcast_x = Zq8::broadcast(x);
        for i in 0..VEC_COUNT {
            self.values[i] *= broadcast_x;
        }
    }

    fn inv_dft(self) -> Rq
    {
        FourierReprRq::inv_dft(self)
    }
}

#[cfg(test)]
const ELEMENT: [i16; N] = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 81, 0, 1, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 55, 0, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 16, 0, 76, 
    13, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 84, 0, 0, 99, 0, 60, 0, 0, 0, 7680, 
    0, 0, 0, 0, 0, 26, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 256, 0, 0, 0, 0, 0, 0, 
    0, 0, 71, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 0, 0, 3840, 0, 0, 
    2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
    0, 0, 0, 67, 0, 0, 7680, 0, 0, 0, 0, 48, 0, 63, 0, 0, 21, 0, 0, 0, 0, 0, 0, 1, 
    52, 0, 0, 0, 47, 0, 0, 0, 0, 95, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    73, 15, 0, 0, 22, 0, 0, 0, 0, 0, 0, 1, 64, 2, 0, 87, 0, 0, 1, 0, 0, 0, 1, 0, 0, 
    0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];

#[cfg(test)]
const DFT_ELEMENT: [i16; N] = [5487, 7048, 1145, 6716, 88, 5957, 3742, 3441, 2663, 
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
fn bench_fft(bencher: &mut test::Bencher) {
    let element = Rq::from(&ELEMENT[..]);
    let expected_fourier_reprn = FourierReprRq::from(&DFT_ELEMENT[..]);
    bencher.iter(|| {
        let fourier_repr = FourierReprRq::dft(element.clone());
        assert_eq!(expected_fourier_reprn, fourier_repr);
    });
}

#[test]
fn test_inv_fft() {
    let fourier_repr = FourierReprRq::from(&DFT_ELEMENT[..]);
    let expected_element = Rq::from(&ELEMENT[..]);
    let element = FourierReprRq::inv_dft(fourier_repr);
    assert_eq!(expected_element, element);
}

#[test]
fn test_scalar_mul_div() {
    let mut element = Rq::from(&ELEMENT[..]);
    let mut fourier_repr = FourierReprRq::dft(element.clone());
    element *= Zq::from(653_i16);
    fourier_repr *= Zq::from(653_i16);
    assert_eq!(element, FourierReprRq::inv_dft(fourier_repr.clone()));

    element *= ONE / Zq::from(5321_i16);
    fourier_repr *= ONE / Zq::from(5321_i16);
    assert_eq!(element, FourierReprRq::inv_dft(fourier_repr.clone()));
}

#[test]
fn test_add_sub() {
    let mut element = Rq::from(&ELEMENT[..]);
    let mut fourier_repr = FourierReprRq::dft(element.clone());
    let base_element = element.clone();
    let base_fourier_repr = fourier_repr.clone();

    element += &base_element;
    fourier_repr += &base_fourier_repr;
    assert_eq!(element, FourierReprRq::inv_dft(fourier_repr.clone()));

    element -= &base_element;
    fourier_repr -= &base_fourier_repr;
    assert_eq!(element, FourierReprRq::inv_dft(fourier_repr));
    assert_eq!(Rq::from(&ELEMENT[..]), element);
}

#[test]
fn test_compress() {
    let mut element = Rq::from(&DFT_ELEMENT[..]);
    let compressed: CompressedRq<3_u16> = element.compress();
    element = Rq::decompress(&compressed);
    assert_eq!(Zq8::from(&[5761, 6721, 960, 6721, 0, 5761, 3840, 3840][..]), element.data[0]);
}