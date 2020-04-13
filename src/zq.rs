use std::ops::{ Add, Mul, Div, Sub, Neg, AddAssign, MulAssign, DivAssign, SubAssign };
use std::cmp::{ PartialEq, Eq };
use std::fmt::{ Debug, Display, Formatter };
use std::convert::From;
use std::mem::swap;

use super::encoding;

macro_rules! zq_arr {
    ($($num:literal),*) => {
        [$(ZqElement { value: $num }),*]
    };
}

pub const Q: u32 = 7681;

pub const ZERO: ZqElement = ZqElement { value: 0 };
pub const ONE: ZqElement = ZqElement { value: 1 };

/// All 512-th root of unity
pub const UNITY_ROOTS_512: [ZqElement; 512] = zq_arr![
    1, 1704, 198, 7109, 799, 1959, 4582, 3832, 878, 5998, 
    4862, 4730, 2551, 7139, 5833, 218, 2784, 4759, 5881, 5200, 4607, 346, 5828, 7060, 1794, 7619, 1886, 
    3086, 4740, 4229, 1438, 113, 527, 7012, 4493, 5796, 6299, 3139, 2880, 7042, 1846, 4055, 4501, 4066, 
    202, 6244, 1591, 7352, 97, 3987, 3844, 5964, 693, 5679, 6637, 3016, 675, 5731, 3073, 5631, 1655, 
    1193, 5088, 5784, 1213, 763, 2063, 5135, 1381, 2838, 4603, 1211, 5036, 1667, 6279, 7464, 6601, 3120, 
    1228, 3280, 5033, 4236, 5685, 1499, 4204, 4924, 2844, 7146, 2399, 1604, 6461, 2671, 4232, 6550, 707, 
    6492, 1728, 2689, 4180, 2433, 5773, 5512, 6266, 674, 4027, 2875, 6203, 856, 6915, 506, 1952, 335, 
    2446, 4882, 405, 6511, 3380, 6451, 993, 2252, 4589, 398, 2264, 1994, 2774, 3081, 3901, 3239, 4298, 
    3799, 6094, 7145, 695, 1406, 7033, 1872, 2273, 1968, 4556, 5614, 3411, 5508, 7131, 7563, 6315, 7360, 
    6048, 5571, 6949, 4675, 1003, 3930, 6569, 2359, 2573, 6222, 2508, 2996, 5000, 1771, 6832, 5013, 880, 
    1725, 5258, 3586, 4149, 3376, 7316, 201, 4540, 1393, 243, 6979, 2028, 6943, 2132, 7496, 7362, 1775, 
    5967, 5805, 6273, 4921, 5413, 6552, 4115, 6888, 584, 4287, 417, 3916, 5756, 7268, 2900, 2717, 5806, 
    296, 5119, 4841, 7351, 6074, 3789, 4416, 5165, 6415, 1097, 2805, 2138, 2358, 869, 6024, 3080, 2197, 
    3041, 4870, 3000, 4135, 2563, 4544, 528, 1035, 4691, 5224, 7098, 5098, 7462, 3193, 2724, 2372, 1682, 
    1115, 2753, 5702, 7424, 7570, 2881, 1065, 2044, 3483, 5300, 6025, 4784, 2395, 2469, 5669, 4959, 1036, 
    6395, 5422, 6526, 5897, 1740, 94, 6556, 3250, 7680, 5977, 7483, 572, 6882, 5722, 3099, 3849, 6803, 
    1683, 2819, 2951, 5130, 542, 1848, 7463, 4897, 2922, 1800, 2481, 3074, 7335, 1853, 621, 5887, 62, 5795, 
    4595, 2941, 3452, 6243, 7568, 7154, 669, 3188, 1885, 1382, 4542, 4801, 639, 5835, 3626, 3180, 3615, 
    7479, 1437, 6090, 329, 7584, 3694, 3837, 1717, 6988, 2002, 1044, 4665, 7006, 1950, 4608, 2050, 6026, 
    6488, 2593, 1897, 6468, 6918, 5618, 2546, 6300, 4843, 3078, 6470, 2645, 6014, 1402, 217, 1080, 4561, 
    6453, 4401, 2648, 3445, 1996, 6182, 3477, 2757, 4837, 535, 5282, 6077, 1220, 5010, 3449, 1131, 6974, 
    1189, 5953, 4992, 3501, 5248, 1908, 2169, 1415, 7007, 3654, 4806, 1478, 6825, 766, 7175, 5729, 7346, 
    5235, 2799, 7276, 1170, 4301, 1230, 6688, 5429, 3092, 7283, 5417, 5687, 4907, 4600, 3780, 4442, 3383, 
    3882, 1587, 536, 6986, 6275, 648, 5809, 5408, 5713, 3125, 2067, 4270, 2173, 550, 118, 1366, 321, 1633, 
    2110, 732, 3006, 6678, 3751, 1112, 5322, 5108, 1459, 5173, 4685, 2681, 5910, 849, 2668, 6801, 5956, 
    2423, 4095, 3532, 4305, 365, 7480, 3141, 6288, 7438, 702, 5653, 738, 5549, 185, 319, 5906, 1714, 1876, 
    1408, 2760, 2268, 1129, 3566, 793, 7097, 3394, 7264, 3765, 1925, 413, 4781, 4964, 1875, 7385, 2562, 
    2840, 330, 1607, 3892, 3265, 2516, 1266, 6584, 4876, 5543, 5323, 6812, 1657, 4601, 5484, 4640, 2811, 
    4681, 3546, 5118, 3137, 7153, 6646, 2990, 2457, 583, 2583, 219, 4488, 4957, 5309, 5999, 6566, 4928, 
    1979, 257, 111, 4800, 6616, 5637, 4198, 2381, 1656, 2897, 5286, 5212, 2012, 2722, 6645, 1286, 2259, 
    1155, 1784, 5941, 7587, 1125, 4431
];

/// The inverses of the first 256 512-th roots of unity given by UNITY_ROOTS_512, these are
/// also 512-th roots of unity
pub const REV_UNITY_ROOTS_512: [ZqElement; 256] = zq_arr![
    1, 4431, 1125, 7587, 5941, 1784, 1155, 2259, 1286, 6645, 2722, 2012, 5212, 5286, 2897, 1656, 2381, 
    4198, 5637, 6616, 4800, 111, 257, 1979, 4928, 6566, 5999, 5309, 4957, 4488, 219, 2583, 583, 2457, 
    2990, 6646, 7153, 3137, 5118, 3546, 4681, 2811, 4640, 5484, 4601, 1657, 6812, 5323, 5543, 4876, 6584, 
    1266, 2516, 3265, 3892, 1607, 330, 2840, 2562, 7385, 1875, 4964, 4781, 413, 1925, 3765, 7264, 3394,
    7097, 793, 3566, 1129, 2268, 2760, 1408, 1876, 1714, 5906, 319, 185, 5549, 738, 5653, 702, 7438, 
    6288, 3141, 7480, 365, 4305, 3532, 4095, 2423, 5956, 6801, 2668, 849, 5910, 2681, 4685, 5173, 
    1459, 5108, 5322, 1112, 3751, 6678, 3006, 732, 2110, 1633, 321, 1366, 118, 550, 2173, 4270, 2067, 
    3125, 5713, 5408, 5809, 648, 6275, 6986, 536, 1587, 3882, 3383, 4442, 3780, 4600, 4907, 5687, 5417, 
    7283, 3092, 5429, 6688, 1230, 4301, 1170, 7276, 2799, 5235, 7346, 5729, 7175, 766, 6825, 1478, 4806, 
    3654, 7007, 1415, 2169, 1908, 5248, 3501, 4992, 5953, 1189, 6974, 1131, 3449, 5010, 1220, 6077, 5282,
    535, 4837, 2757, 3477, 6182, 1996, 3445, 2648, 4401, 6453, 4561, 1080, 217, 1402, 6014, 2645, 6470, 
    3078, 4843, 6300, 2546, 5618, 6918, 6468, 1897, 2593, 6488, 6026, 2050, 4608, 1950, 7006, 4665, 1044, 
    2002, 6988, 1717, 3837, 3694, 7584, 329, 6090, 1437, 7479, 3615, 3180, 3626, 5835, 639, 4801, 4542, 
    1382, 1885, 3188, 669, 7154, 7568, 6243, 3452, 2941, 4595, 5795, 62, 5887, 621, 1853, 7335, 3074, 2481, 
    1800, 2922, 4897, 7463, 1848, 542, 5130, 2951, 2819, 1683, 6803, 3849, 3099, 5722, 6882, 572, 7483, 5977
];

fn extended_euclidean_algorithm_mod_q(fst: u32, snd: u32) -> (u32, u32) 
{
    let (mut a, mut b): (u32, u32) = (fst, snd);
    let (mut sa, mut ta): (u32, u32) = (1, 0);
    let (mut sb, mut tb): (u32, u32) = (0, 1);

    // TODO: the loop runs at most 20 times (20 >= 1 + log_phi(7681)), optimize?

    // invariant:
    // a = sa * fst + ta * snd mod q,
    // b = sb * fst + tb * snd mod q
    while b != 0 {
        ta = ta + Q - ((a / b * tb) % Q);
        sa = sa + Q - ((a / b * sb) % Q);
        if ta >= Q {
            ta -= Q;
        }
        if sa >= Q {
            sa -= Q;
        }
        a = a % b;
        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }
    return (sa, ta);
}

/// The type of elements of the ring Zq := Z / qZ
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ZqElement 
{
    value: u32
}

impl ZqElement 
{
    // Raises this element to the power of a natural number
    pub fn pow(self, mut rhs: usize) -> ZqElement
    {
        let mut power: ZqElement = self;
        let mut result: ZqElement = ONE;
        while rhs != 0 {
            if rhs & 1 == 1 {
                result *= power;
            }
            power *= power;
            rhs = rhs >> 1;
        }
        return result;
    }

    // contract: Zq::From(x.representative_pos()) == x
    pub fn representative_pos(self) -> i16
    {
        self.value as i16
    }

    // contract: Zq::From(x.representative_posneg()) == x
    pub fn representative_posneg(self) -> i16
    {
        if self.value > Q/2 {
            -(self.value as i16)
        } else {
            self.value as i16
        }
    }

    pub fn from_perfect(value: i16) -> ZqElement
    {
        debug_assert!(value >= 0 && (value as u32) < Q, "Got value {} which is not in range 0..{}", value, Q);
        ZqElement {
            value: value as u32
        }
    }
}

impl Debug for ZqElement 
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[{}]q", self.value)
    }
}

impl Display for ZqElement 
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "{}", self.value)
    }
}

impl Add<ZqElement> for ZqElement
{
    type Output = ZqElement;

    #[inline(always)]
    fn add(mut self, rhs: ZqElement) -> Self::Output
    {
        self += rhs;
        return self;
    }
}

impl Mul<ZqElement> for ZqElement
{
    type Output = ZqElement;

    #[inline(always)]
    fn mul(mut self, rhs: ZqElement) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

impl Sub<ZqElement> for ZqElement
{
    type Output = ZqElement;

    #[inline(always)]
    fn sub(mut self, rhs: ZqElement) -> Self::Output
    {
        self -= rhs;
        return self;
    }
}

impl Div<ZqElement> for ZqElement
{
    type Output = ZqElement;

    #[inline(always)]
    fn div(mut self, rhs: ZqElement) -> Self::Output
    {
        self /= rhs;
        return self;
    }
}

impl Neg for ZqElement
{
    type Output = ZqElement;

    #[inline(always)]
    fn neg(self) -> Self::Output
    {
        ZqElement {
            value: Q - self.value
        }
    }
}

impl AddAssign<ZqElement> for ZqElement
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: ZqElement)
    {
        self.value = self.value + rhs.value;
        if self.value >= Q {
            self.value -= Q;
        }
    }
}

impl MulAssign<ZqElement> for ZqElement
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: ZqElement)
    {
        self.value = self.value * rhs.value % Q
    }
}

impl SubAssign<ZqElement> for ZqElement
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: ZqElement)
    {
        self.value = self.value + Q - rhs.value;
        if self.value >= Q {
            self.value -= Q;
        }
    }
}

impl DivAssign<ZqElement> for ZqElement
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: ZqElement)
    {
        self.value = (self.value * extended_euclidean_algorithm_mod_q(Q, rhs.value).1) % Q;
    }
}

impl From<i16> for ZqElement
{
    // Returns the equivalence class of the argument in Zq
    #[inline(always)]
    fn from(value: i16) -> ZqElement
    {
        // A i16 is positive for sure after adding 5 * Q > 32768
        // and this addition will not overflow as i32
        ZqElement {
            value: ((value as i32 + 5 * Q as i32) % Q as i32) as u32 
        }
    }
}

#[derive(Clone, Copy)]
pub struct CompressedZq<const D: u16>
{
    pub data: u16
}

impl<const D: u16> encoding::Encodable for CompressedZq<D>
{
    fn encode<T: encoding::Encoder>(&self, encoder: &mut T)
    {
        encoder.encode_bits(self.data, D as usize);
    }

    fn decode<T: encoding::Decoder>(data: &mut T) -> Self
    {
        CompressedZq {
            data: data.read_bits(D as usize).expect("Input too short")
        }
    }
}

impl<const D: u16> Debug for CompressedZq<D>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "{} [0..{}]", self.data, 1 << D)
    }
}

impl ZqElement 
{
    // Returns the element y in 0, ..., 2^d - 1 such
    // that q/2^n * y is nearest to x.representative_pos()
    pub fn compress<const D: u16>(self) -> CompressedZq<D>
    {
        // this floating point approach always leads to the right result:
        // for each x, n, |0.5 - (x * n / 7681) mod 1| >= |0.5 - (x * 1 / 7681) mod 1|
        // >= |0.5 - (3840 / 7681) mod 1| >= 6.509569066531773E-5 
        // > (error in floating point representation of 1/7681) * 7681
        let n = (1 << D) as f32;
        CompressedZq {
            data: (self.representative_pos() as f32 * n / Q as f32).round() as u16 % (1 << D)
        }
    }
    
    // Returns the element y of Zq for which
    // y.representative_pos() is nearest to 2^d/q * x 
    pub fn decompress<const D: u16>(x: CompressedZq<D>) -> ZqElement
    {
        let n = (1 << D) as f32;
        ZqElement::from((x.data as f32 * Q as f32 / n).round() as i16)
    }
}

impl<const D: u16> CompressedZq<D>
{
    pub fn zero() -> CompressedZq<D>
    {
        CompressedZq {
            data: 0
        }
    }
}

impl CompressedZq<1>
{
    pub fn get_bit(&self) -> u8
    {
        self.data as u8
    }

    pub fn from_bit(m: u8) -> CompressedZq<1>
    {
        CompressedZq {
            data: (m & 1) as u16
        }
    }
}

#[test]
fn test_mul() {
    for i in 0..Q {
        for j in 0..Q {
            assert_eq!(ZqElement::from((i * j % Q) as i16), ZqElement::from_perfect(i as i16) * ZqElement::from_perfect(j as i16), "{} * {}", i, j);
        }
    }
}

use super::ref_impl_compat;
use super::encoding::Encodable;

#[test]
fn test_decompress() {
    let value: CompressedZq<11> = CompressedZq {
        data: 1578
    };
    assert_eq!(ZqElement::from(5918), ZqElement::decompress(value));
}