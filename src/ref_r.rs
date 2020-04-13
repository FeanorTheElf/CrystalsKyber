use super::zq::*;
use super::ring::*;

use super::util;
use super::encoding;

use std::ops::{ Add, Mul, Sub, AddAssign, MulAssign, SubAssign };
use std::cmp::{ PartialEq, Eq };
use std::convert::From;
use std::fmt::{ Formatter, Debug };

#[derive(Clone)]
pub struct RqElementCoefficientReprImpl
{
    data: [ZqElement; N]
}

impl PartialEq for RqElementCoefficientReprImpl
{
    fn eq(&self, rhs: &RqElementCoefficientReprImpl) -> bool
    {
        (0..N).all(|i| self.data[i] == rhs.data[i])
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
        rhs *= -ONE;
        return rhs;
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

impl<'a> AddAssign<&'a RqElementCoefficientReprImpl> for RqElementCoefficientReprImpl
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a RqElementCoefficientReprImpl)
    {
        for i in 0..N {
            self.data[i] += rhs.data[i];
        }
    }
}

impl<'a> SubAssign<&'a RqElementCoefficientReprImpl> for RqElementCoefficientReprImpl
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a RqElementCoefficientReprImpl)
    {
        for i in 0..N {
            self.data[i] -= rhs.data[i];
        }
    }
}

impl MulAssign<ZqElement> for RqElementCoefficientReprImpl
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: ZqElement)
    {
        for i in 0..N {
            self.data[i] *= rhs;
        }
    }
}

impl<'a> From<&'a [i16]> for RqElementCoefficientReprImpl
{
    fn from(value: &'a [i16]) -> RqElementCoefficientReprImpl {
        assert_eq!(N, value.len());
        return RqElementCoefficientReprImpl {
            data: util::create_array(|i| ZqElement::from(value[i]))
        };
    }
}

impl From<[ZqElement; N]> for RqElementCoefficientReprImpl
{
    #[inline(always)]
    fn from(data: [ZqElement; N]) -> RqElementCoefficientReprImpl
    {
        assert_eq!(N, data.len());
        RqElementCoefficientReprImpl {
            data: util::create_array(|i| data[i])
        }
    }
}

impl Debug for RqElementCoefficientReprImpl
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[")?;
        (0..255).try_for_each(|i| write!(f, "{}, ", self.data[i]))?;
        write!(f, "{}]", self.data[255])?;
        return Ok(());
    }
}

impl RqElementCoefficientRepr for RqElementCoefficientReprImpl
{
    type ChineseRemainderRepr = RqElementChineseRemainderReprImpl;

    fn get_zero() -> RqElementCoefficientReprImpl
    {
        return RqElementCoefficientReprImpl {
            data: [ZERO; N]
        }
    }

    fn to_chinese_remainder_repr(mut self) -> RqElementChineseRemainderReprImpl
    {
        // we do not need the exact fourier transformation (i.e. the evaluation at
        // all 256-th roots of unity), but the evaluation at all primitive 512-th
        // roots of unity. Since the primitive 512-th roots of unity are exactly
        // the 256-th roots of unity multiplied with any primitive root of unity,
        // this approach lets us calculate the correct result
        for i in 0..N {
            self.data[i] *= UNITY_ROOTS_512[i]
        }
        RqElementChineseRemainderReprImpl {
            values: RqElementChineseRemainderReprImpl::fft(self.data, |i| UNITY_ROOTS_512[2 * i])
        }
    }

    
    fn compress<const D: u16>(&self) -> CompressedRq<D>
    {
        let mut data = [CompressedZq::zero(); N];
        for i in 0..N {
            data[i] = self.data[i].compress();
        }
        return CompressedRq {
            data: data
        };
    }

    fn decompress<const D: u16>(x: &CompressedRq<D>) -> RqElementCoefficientReprImpl
    {
        let mut data = [ZERO; N];
        for i in 0..N {
            data[i] = ZqElement::decompress(x.data[i]);
        }
        return RqElementCoefficientReprImpl {
            data: data
        };
    }
}

/// Chinese remainder representation of an element of Rq, i.e.
/// the values of the polynomial at each root of unity
/// in Zq
#[derive(Clone)]
pub struct RqElementChineseRemainderReprImpl
{
    values: [ZqElement; N]
}

impl RqElementChineseRemainderReprImpl
{
    #[inline(never)]
    fn fft<F>(mut values: [ZqElement; N], unity_root: F) -> [ZqElement; N]
        where F: Fn(usize) -> ZqElement
    {
        // Use the Cooley–Tukey FFT algorithm (N = N):
        // for i from 1 to log(N) do:
        //   n = 2^i
        //   Calculate the DFT of [x_j, x_j+d, x_j+2d, ..., x_j+(n-1)d]
        //   for each j in 0..d and each k in 0..n where d=N/n
        //   using only the DFTs from the last iteration
        // During each loop, values and temp will hold the DFTs of this and the last loop (they change the roles each iteration)
        // Both contain the value: values[k * d + j] is the DFT with j and k
        let mut temp: [ZqElement; N] = [ZERO; N];

        // values already contain the k=1 DFT of [x_j], so start with i = 7
        let mut n: usize = 1;
        let mut d: usize = 1 << 8;
        let mut old_d: usize;
        for _i in 0..4 {
            n = n << 1;
            d = d >> 1;
            old_d = d << 1;
            for k in 0..n/2 {
                // w := n-th root of unity, unity_root := w^k
                let unity_root = unity_root(k * d);
                for j in 0..d {
                    // calculate the k-DFT of [x_j, x_j+d, x_j+2d, ..., x_j+(n-1)d],
                    // have x_j + w^k * x_j+d + w^2k * x_j+2d + ...
                    //  = (x_j + w^2k * x_j+2d + w^4k * w_j+4d + ...)  (1)
                    //  + (x_j+d + w^2k * x_j+3d + w^4k * w_j+4d + ...) * w  (2)
                    // (1) is the k-DFT of [x_j, x_j+2d, x_j+4d, ...], so it has been
                    //     calculated in the last step (with j_old = j, d_old = 2d)
                    // (2) is w * the k-DFT of [x_j+d, x_j+3d, x_j+5d, ...], so it has
                    //     been calculated in the last step (with j_old = j + d < d_old, d_old = 2d)
                    // (1) is at the location old_values [k * last_d]
                    temp[k * d + j] = values[k * old_d + j] + unity_root * values[k * old_d + j + d];

                    // Use the trick that w^(n/2) = -1, so (1) above stays the same
                    // (we get a factor of w^(2l * (k+n/2)) = w^2lk * w^ln = w^2lk * 1 in each term)
                    // and (2) above is negated
                    // (we get a factor of w^((2l+1) * (k+n/2)) = w^(2l+1)k * w^2l(n/2) * w^(n/2) 
                    // = w^(2l+1)k * 1 * w^(n/2) = w^(2l+1)k * -1 in each term)
                    temp[(k + n/2) * d + j] = values[k * old_d + j] - unity_root * values[k * old_d + j + d];
                }
            }

            // This corresponds to the next iteration of the loop, but we
            // unroll it so that no swap between values and temp is necessary
            // (so now values and temp switch roles, otherwise it is the same)
            n = n << 1;
            d = d >> 1;
            old_d = d << 1;
            for k in 0..n/2 {
                let unity_root = unity_root(k * d);
                for j in 0..d {
                    values[k * d + j] = temp[k * old_d + j] + unity_root * temp[k * old_d + j + d];
                    values[(k + n/2) * d + j] = temp[k * old_d + j] - unity_root * temp[k * old_d + j + d];
                }
            }
        }
        return values;
    }

    fn coefficient_repr(ntt_repr: RqElementChineseRemainderReprImpl) -> RqElementCoefficientReprImpl
    {
        let inv_n: ZqElement = ONE / ZqElement::from(N as i16);
        let mut result = Self::fft(ntt_repr.values, |i| UNITY_ROOTS_512[2 * ((N - i) % 256)]);
        for i in 0..N {
            // see dft for why this is necessary (we do not do a real fourier transformation)
            result[i] *= REV_UNITY_ROOTS_512[i];
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
        (0..N).all(|i| self.values[i] == rhs.values[i])
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
        for i in 0..N {
            self.values[i] += rhs.values[i];
        }
    }
}

impl<'a> SubAssign<&'a RqElementChineseRemainderReprImpl> for RqElementChineseRemainderReprImpl
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a RqElementChineseRemainderReprImpl) {
        for i in 0..N {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl<'a> MulAssign<&'a RqElementChineseRemainderReprImpl> for RqElementChineseRemainderReprImpl
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a RqElementChineseRemainderReprImpl) {
        for i in 0..N {
            self.values[i] *= rhs.values[i];
        }
    }
}

impl MulAssign<ZqElement> for RqElementChineseRemainderReprImpl
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: ZqElement) {
        for i in 0..N {
            self.values[i] *= rhs;
        }
    }
}

impl Debug for RqElementChineseRemainderReprImpl
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[")?;
        (0..255).try_for_each(|i| write!(f, "{}, ", self.values[i]))?;
        write!(f, "{}]", self.values[255])?;
        return Ok(());
    }
}

// The count of bits we write when encoding an element of Zq 
const ENCODE_ENTRY_BITS: usize = 13;

impl encoding::Encodable for RqElementChineseRemainderReprImpl
{
    fn encode<T: encoding::Encoder>(&self, encoder: &mut T)
    {
        for i in 0..N {
            encoder.encode_bits(self.values[i].representative_pos() as u16, ENCODE_ENTRY_BITS);
        }
    }

    fn decode<T: encoding::Decoder>(data: &mut T) -> Self
    {
        RqElementChineseRemainderReprImpl {
            values: util::create_array(|_i| {
                let data_bits = data.read_bits(ENCODE_ENTRY_BITS).expect("Input too short");
                ZqElement::from_perfect(data_bits as i16)
            })
        }
    }
}

impl<'a> From<&'a [i16]> for RqElementChineseRemainderReprImpl
{
    fn from(value: &'a [i16]) -> Self
    {
        assert_eq!(N, value.len());
        RqElementChineseRemainderReprImpl {
            values: util::create_array(|i| ZqElement::from(value[i]))
        }
    }
}

impl From<[ZqElement; N]> for RqElementChineseRemainderReprImpl
{
    fn from(value: [ZqElement; N]) -> Self
    {
        assert_eq!(N, value.len());
        RqElementChineseRemainderReprImpl {
            values: util::create_array(|i| value[i])
        }
    }
}

impl RqElementChineseRemainderRepr for RqElementChineseRemainderReprImpl
{
    type CoefficientRepr = RqElementCoefficientReprImpl;

    fn get_zero() -> RqElementChineseRemainderReprImpl
    {
        return RqElementChineseRemainderReprImpl {
            values: [ZERO; N]
        }
    }

    fn to_coefficient_repr(self) -> RqElementCoefficientReprImpl 
    {
        RqElementChineseRemainderReprImpl::coefficient_repr(self)
    }

    fn mul_scalar(&mut self, x: ZqElement)
    {
        *self *= x;
    }

    fn add_product(&mut self, a: &RqElementChineseRemainderReprImpl, b: &RqElementChineseRemainderReprImpl)
    {
        for i in 0..N {
            self.values[i] += a.values[i] * b.values[i];
        }
    }
    
    fn value_at_zeta(&self, zeta_index: usize) -> ZqElement
    {
        self.values[zeta_index]
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
        let ntt_repr = element.clone().to_chinese_remainder_repr();
        assert_eq!(element, ntt_repr.to_coefficient_repr());
    });
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
fn test_scalar_mul_div() {
    let mut element = RqElementCoefficientReprImpl::from(&ELEMENT[..]);
    let mut ntt_repr = element.clone().to_chinese_remainder_repr();
    element *= ZqElement::from(653_i16);
    ntt_repr *= ZqElement::from(653_i16);
    assert_eq!(element, RqElementChineseRemainderReprImpl::coefficient_repr(ntt_repr.clone()));

    element *= ONE / ZqElement::from(5321_i16);
    ntt_repr *= ONE / ZqElement::from(5321_i16);
    assert_eq!(element, RqElementChineseRemainderReprImpl::coefficient_repr(ntt_repr.clone()));
}

#[test]
fn test_add_sub() {
    let mut element = RqElementCoefficientReprImpl::from(&ELEMENT[..]);
    let mut ntt_repr = element.clone().to_chinese_remainder_repr();
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