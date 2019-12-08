use std::ops::{ Add, Mul, Div, Sub, AddAssign, MulAssign, DivAssign, SubAssign };
use std::cmp::{ PartialEq, Eq };
use std::fmt::{ Debug, Display, Formatter };
use std::convert::From;
use std::mem::swap;

macro_rules! zq_arr {
    ($($num:literal),*) => {
        [$(Zq { value: $num }),*]
    };
}

pub const UNITY_ROOTS: [Zq; 256] = zq_arr![1, 198, 799, 4582, 878, 4862, 2551, 5833, 2784, 5881, 
    4607, 5828, 1794, 1886, 4740, 1438, 527, 4493, 6299, 2880, 1846, 4501, 202, 1591, 97, 3844, 
    693, 6637, 675, 3073, 1655, 5088, 1213, 2063, 1381, 4603, 5036, 6279, 6601, 1228, 5033, 5685, 
    4204, 2844, 2399, 6461, 4232, 707, 1728, 4180, 5773, 6266, 4027, 6203, 6915, 1952, 2446, 405, 
    3380, 993, 4589, 2264, 2774, 3901, 4298, 6094, 695, 7033, 2273, 4556, 3411, 7131, 6315, 6048, 
    6949, 1003, 6569, 2573, 2508, 5000, 6832, 880, 5258, 4149, 7316, 4540, 243, 2028, 2132, 7362, 
    5967, 6273, 5413, 4115, 584, 417, 5756, 2900, 5806, 5119, 7351, 3789, 5165, 1097, 2138, 869,
    3080, 3041, 3000, 2563, 528, 4691, 7098, 7462, 2724, 1682, 2753, 7424, 2881, 2044, 5300, 4784, 
    2469, 4959, 6395, 6526, 1740, 6556, 7680, 7483, 6882, 3099, 6803, 2819, 5130, 1848, 4897, 1800, 
    3074, 1853, 5887, 5795, 2941, 6243, 7154, 3188, 1382, 4801, 5835, 3180, 7479, 6090, 7584, 3837, 
    6988, 1044, 7006, 4608, 6026, 2593, 6468, 5618, 6300, 3078, 2645, 1402, 1080, 6453, 2648, 1996, 
    3477, 4837, 5282, 1220, 3449, 6974, 5953, 3501, 1908, 1415, 3654, 1478, 766, 5729, 5235, 7276, 
    4301, 6688, 3092, 5417, 4907, 3780, 3383, 1587, 6986, 648, 5408, 3125, 4270, 550, 1366, 1633, 
    732, 6678, 1112, 5108, 5173, 2681, 849, 6801, 2423, 3532, 365, 3141, 7438, 5653, 5549, 319, 1714, 
    1408, 2268, 3566, 7097, 7264, 1925, 4781, 1875, 2562, 330, 3892, 2516, 6584, 5543, 6812, 4601, 
    4640, 4681, 5118, 7153, 2990, 583, 219, 4957, 5999, 4928, 257, 4800, 5637, 2381, 2897, 5212, 
    2722, 1286, 1155, 5941, 1125];

pub const INV_UNITY_ROOTS: [Zq; 256] = zq_arr![1, 1125, 5941, 1155, 1286, 2722, 5212, 2897, 2381, 
    5637, 4800, 257, 4928, 5999, 4957, 219, 583, 2990, 7153, 5118, 4681, 4640, 4601, 6812, 5543, 
    6584, 2516, 3892, 330, 2562, 1875, 4781, 1925, 7264, 7097, 3566, 2268, 1408, 1714, 319, 5549, 
    5653, 7438, 3141, 365, 3532, 2423, 6801, 849, 2681, 5173, 5108, 1112, 6678, 732, 1633, 1366, 
    550, 4270, 3125, 5408, 648, 6986, 1587, 3383, 3780, 4907, 5417, 3092, 6688, 4301, 7276, 5235, 
    5729, 766, 1478, 3654, 1415, 1908, 3501, 5953, 6974, 3449, 1220, 5282, 4837, 3477, 1996, 2648, 
    6453, 1080, 1402, 2645, 3078, 6300, 5618, 6468, 2593, 6026, 4608, 7006, 1044, 6988, 3837, 7584, 
    6090, 7479, 3180, 5835, 4801, 1382, 3188, 7154, 6243, 2941, 5795, 5887, 1853, 3074, 1800, 4897, 
    1848, 5130, 2819, 6803, 3099, 6882, 7483, 7680, 6556, 1740, 6526, 6395, 4959, 2469, 4784, 5300, 
    2044, 2881, 7424, 2753, 1682, 2724, 7462, 7098, 4691, 528, 2563, 3000, 3041, 3080, 869, 2138, 
    1097, 5165, 3789, 7351, 5119, 5806, 2900, 5756, 417, 584, 4115, 5413, 6273, 5967, 7362, 2132, 
    2028, 243, 4540, 7316, 4149, 5258, 880, 6832, 5000, 2508, 2573, 6569, 1003, 6949, 6048, 6315, 
    7131, 3411, 4556, 2273, 7033, 695, 6094, 4298, 3901, 2774, 2264, 4589, 993, 3380, 405, 2446, 
    1952, 6915, 6203, 4027, 6266, 5773, 4180, 1728, 707, 4232, 6461, 2399, 2844, 4204, 5685, 5033, 
    1228, 6601, 6279, 5036, 4603, 1381, 2063, 1213, 5088, 1655, 3073, 675, 6637, 693, 3844, 97, 1591, 
    202, 4501, 1846, 2880, 6299, 4493, 527, 1438, 4740, 1886, 1794, 5828, 4607, 5881, 2784, 5833, 2551, 
    4862, 878, 4582, 799, 198];

pub const Q: u32 = 7681;

pub const ZERO: Zq = Zq { value: 0 };
pub const ONE: Zq = Zq { value: 1 };

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

// The type of elements of the ring Zq := Z / q*Z
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Zq 
{
    value: u32
}

impl Zq 
{
    // Raises this element to the power of a natural number
    pub fn pow(self, mut rhs: usize) -> Zq
    {
        let mut power: Zq = self;
        let mut result: Zq = ONE;
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
    pub fn representative_pos(self) -> u16
    {
        self.value as u16
    }

    // contract: Zq::From(x.representative_posneg()) == x
    pub fn representative_posneg(self) -> i16
    {
        if self.value > 3840 {
            -(self.value as i16)
        } else {
            self.value as i16
        }
    }

    // Returns the element y in 0, ..., 2^d - 1 such
    // that q/2^n * y is nearest to x.representative_pos()
    pub fn compress(self, d: u16) -> u16
    {
        // this floating point approach always leads to the right result:
        // for each x, n, |0.5 - (x * n / 7681) mod 1| >= |0.5 - (x * 1 / 7681) mod 1|
        // >= |0.5 - (3840 / 7681) mod 1| >= 6.509569066531773E-5 > error in float * 7681
        let n = (1 << d) as f32;
        (self.representative_pos() as f32 * n / Q as f32).round() as u16 % (1 << d)
    }

    // Returns the element y of Zq for which
    // y.representative_pos() is nearest to 2^d/q * x 
    pub fn decompress(x: u16, d: u16) -> Self
    {
        let n = (1 << d) as f32;
        Zq::from((x as f32 * Q as f32 / n).round() as u16)
    }
}

impl Debug for Zq 
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[{}]q", self.value)
    }
}

impl Display for Zq 
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "{}", self.value)
    }
}

impl Add<Zq> for Zq
{
    type Output = Zq;

    #[inline(always)]
    fn add(mut self, rhs: Zq) -> Self::Output
    {
        self += rhs;
        return self;
    }
}

impl Mul<Zq> for Zq
{
    type Output = Zq;

    #[inline(always)]
    fn mul(mut self, rhs: Zq) -> Self::Output
    {
        self *= rhs;
        return self;
    }
}

impl Sub<Zq> for Zq
{
    type Output = Zq;

    #[inline(always)]
    fn sub(mut self, rhs: Zq) -> Self::Output
    {
        self -= rhs;
        return self;
    }
}

impl Div<Zq> for Zq
{
    type Output = Zq;

    #[inline(always)]
    fn div(mut self, rhs: Zq) -> Self::Output
    {
        self /= rhs;
        return self;
    }
}

impl AddAssign<Zq> for Zq
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Zq)
    {
        self.value = self.value + rhs.value;
        if self.value >= Q {
            self.value -= Q;
        }
    }
}

impl MulAssign<Zq> for Zq
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq)
    {
        self.value = (self.value * rhs.value) % Q;
    }
}

impl SubAssign<Zq> for Zq
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Zq)
    {
        self.value = self.value + Q - rhs.value;
        if self.value >= Q {
            self.value -= Q;
        }
    }
}

impl DivAssign<Zq> for Zq
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: Zq)
    {
        self.value = (self.value * extended_euclidean_algorithm_mod_q(Q, rhs.value).1) % Q;
    }
}

impl From<u16> for Zq
{
    // Returns the equivalence class of the argument in Zq
    #[inline(always)]
    fn from(value: u16) -> Zq
    {
        Zq {
            value: value as u32 % Q
        }
    }
}

impl From<i16> for Zq
{
    // Returns the equivalence class of the argument in Zq
    #[inline(always)]
    fn from(value: i16) -> Zq
    {
        // A i16 is positive for sure after adding 5 * Q > 32768
        // and this addition will not overflow as i32
        Zq {
            value: ((value as i32 + 9 * Q as i32) % Q as i32) as u32 
        }
    }
}

#[bench]
fn bench_add_mul(bencher: &mut test::Bencher)
{
    let data: [u32; 32] = [1145, 6716, 88, 5957, 3742, 3441, 2663, 1301, 159, 4074, 2945, 6671, 1392, 3999, 
        2394, 7624, 2420, 4199, 2762, 4206, 4471, 1582, 3870, 5363, 4246, 1800, 4568, 2081, 5642, 1115, 1242, 704];
    let mut elements: [Zq; 32] = [ZERO; 32];
    for i in 0..32 {
        elements[i] = Zq::from(data[i] as u16);
    }
    bencher.iter(|| {
        for i in 0..32 {
            for j in 0..32 {
                assert_eq!((data[i] * data[j] + data[i] + data[j]) % Q, (elements[i] * elements[j] + elements[i] + elements[j]).value % 7681);
            }
        }
    });
}