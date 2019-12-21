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

pub const Q: u32 = 7681;

pub const ZERO: Zq = Zq { value: 0 };
pub const ONE: Zq = Zq { value: 1 };

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

    pub fn from_perfect(value: i32) -> Zq
    {
        assert!(value >= 0 && (value as u32) < Q);
        Zq {
            value: value as u32
        }
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

impl From<i16> for Zq
{
    // Returns the equivalence class of the argument in Zq
    #[inline(always)]
    fn from(value: i16) -> Zq
    {
        // A i16 is positive for sure after adding 5 * Q > 32768
        // and this addition will not overflow as i32
        Zq {
            value: ((value as i32 + 5 * Q as i32) % Q as i32) as u32 
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CompressedZq<const d : u16>
{
    data: u16
}

impl Zq 
{
    // Returns the element y in 0, ..., 2^d - 1 such
    // that q/2^n * y is nearest to x.representative_pos()
    pub fn compress<const d: u16>(self) -> CompressedZq<d>
    {
        // this floating point approach always leads to the right result:
        // for each x, n, |0.5 - (x * n / 7681) mod 1| >= |0.5 - (x * 1 / 7681) mod 1|
        // >= |0.5 - (3840 / 7681) mod 1| >= 6.509569066531773E-5 
        // > (error in floating point representation of 1/7681) * 7681
        let n = (1 << d) as f32;
        CompressedZq {
            data: (self.representative_pos() as f32 * n / Q as f32).round() as u16 % (1 << d)
        }
    }
}

impl<const d: u16> CompressedZq<d>
{
    // Returns the element y of Zq for which
    // y.representative_pos() is nearest to 2^d/q * x 
    pub fn decompress(self) -> Zq
    {
        let n = (1 << d) as f32;
        Zq::from((self.data as f32 * Q as f32 / n).round() as i16)
    }

    pub fn zero() -> CompressedZq<d>
    {
        CompressedZq {
            data: 0
        }
    }
}

impl CompressedZq<1>
{
    pub fn get_data(&self) -> u8
    {
        self.data as u8
    }

    pub fn from_data(m: u8) -> CompressedZq<1>
    {
        CompressedZq {
            data: (m & 1) as u16
        }
    }
}

#[bench]
fn bench_add_mul(bencher: &mut test::Bencher)
{
    let data: [u32; 32] = core::hint::black_box([1145, 6716, 88, 5957, 3742, 3441, 2663, 1301, 159, 4074, 2945, 6671, 1392, 3999, 
        2394, 7624, 2420, 4199, 2762, 4206, 4471, 1582, 3870, 5363, 4246, 1800, 4568, 2081, 5642, 1115, 1242, 704]);
    let mut elements: [Zq; 32] = [ZERO; 32];
    for i in 0..32 {
        elements[i] = Zq::from(data[i] as i16);
    }
    bencher.iter(|| {
        let mut result: Zq = ZERO;
        for i in 0..32 {
            for j in 0..32 {
                result += elements[i] * elements[j];
            }
        }
        assert_eq!(Zq::from(4050), result);
    });
}