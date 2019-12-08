use super::zq::*;

use std::ops::{ Add, Mul, Div, Sub, AddAssign, MulAssign, DivAssign, SubAssign };
use std::cmp::{ PartialEq, Eq };
use std::convert::From;
use std::fmt::{ Formatter, Debug };

// Type of elements in the ring R := Zq[X] / (X^256 + 1)
#[derive(Clone)]
pub struct R
{
    data: [Zq; 256]
}

impl R
{
    // Returns the i-th coefficient of this polynomial
    pub fn get_coefficient(&self, i: usize) -> Zq
    {
        self.data[i]
    }

    pub fn zero() -> FourierReprR
    {
        return FourierReprR {
            values: [ZERO; 256]
        }
    }

    // Calculates the values which are obtained
    // by applying compress_zq to each coefficient
    // of the given polynomial
    pub fn compress(&self, d: u16) -> [u16; 256]
    {
        let mut result: [u16; 256] = [0; 256];
        for i in 0..256 {
            result[i] = self.data[i].compress(d);
        }
        return result;
    }

    // Calculates the polynomial which coefficients
    // are obtained by applying decompress_zq to the
    // given values
    pub fn decompress(x: &[u16; 256], d: u16) -> Self
    {
        let mut result: [Zq; 256] = [ZERO; 256];
        for i in 0..256 {
            result[i] = Zq::decompress(x[i], d);
        }
        return R {
            data: result
        };
    }
}

impl PartialEq for R
{
    fn eq(&self, rhs: &R) -> bool
    {
        (0..256).all(|i| self.data[i] == rhs.data[i])
    }
}

impl Eq for R {}

impl<'a> Add<&'a R> for R
{
    type Output = R;

    #[inline(always)]
    fn add(mut self, rhs: &'a R) -> R
    {
        self += rhs;
        return self;
    }
}

impl<'a> Add<R> for &'a R
{
    type Output = R;

    #[inline(always)]
    fn add(self, mut rhs: R) -> R
    {
        rhs += self;
        return rhs;
    }
}

impl<'a> Sub<&'a R> for R
{
    type Output = R;

    #[inline(always)]
    fn sub(mut self, rhs: &'a R) -> R
    {
        self -= rhs;
        return self;
    }
}

impl<'a> Sub<R> for &'a R
{
    type Output = R;

    #[inline(always)]
    fn sub(self, mut rhs: R) -> R
    {
        rhs -= self;
        rhs *= ZERO - ONE;
        return rhs;
    }
}

impl Mul<Zq> for R
{
    type Output = R;

    #[inline(always)]
    fn mul(mut self, rhs: Zq) -> R
    {
        self *= rhs;
        return self;
    }
}

impl Div<Zq> for R
{
    type Output = R;

    #[inline(always)]
    fn div(mut self, rhs: Zq) -> R
    {
        self *= rhs;
        return self;
    }
}

impl<'a> AddAssign<&'a R> for R
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a R)
    {
        for i in 0..256 {
            self.data[i] += rhs.data[i];
        }
    }
}

impl<'a> SubAssign<&'a R> for R
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a R)
    {
        for i in 0..256 {
            self.data[i] -= rhs.data[i];
        }
    }
}

impl MulAssign<Zq> for R
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq)
    {
        for i in 0..256 {
            self.data[i] *= rhs;
        }
    }
}

impl DivAssign<Zq> for R
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: Zq)
    {
        for i in 0..256 {
            self.data[i] /= rhs;
        }
    }
}

impl<'a, T> From<&'a [T; 256]> for R
    where Zq: From<T>, T: Copy
{
    fn from(value: &'a [T; 256]) -> R {
        let mut data: [Zq; 256] = [ZERO; 256];
        for i in 0..256 {
            data[i] = Zq::from(value[i]);
        }
        return R {
            data: data
        };
    }
}

impl From<[Zq; 256]> for R
{
    #[inline(always)]
    fn from(data: [Zq; 256]) -> R
    {
        R {
            data: data
        }
    }
}

impl Debug for R
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[")?;
        (0..255).try_for_each(|i| write!(f, "{}, ", self.data[i]))?;
        write!(f, "{}]", self.data[255])?;
        return Ok(());
    }
}

// Fourier representation of an element of R, i.e.
// the values of the polynomial at each root of unity
// in Zq
#[derive(Clone)]
pub struct FourierReprR
{
    values: [Zq; 256]
}

impl FourierReprR
{
    #[no_mangle]
    #[inline(never)]
    fn fft(mut values: [Zq; 256], unity_roots: [Zq; 256]) -> [Zq; 256]
    {
        // Use the Cooley–Tukey FFT algorithm (N = 256):
        // for i from 1 to log(N) do:
        //   n = 2^i
        //   Calculate the DFT of [x_j, x_j+d, x_j+2d, ..., x_j+(n-1)d]
        //   for each j in 0..d and each k in 0..n where d=N/n
        //   using only the DFTs from the last iteration
        // During each loop, values and temp will hold
        // the DFTs of this and the last loop (they change
        // the roles each iteration)
        // Both contain the value:
        //   values[k * d + j] is the DFT with j and k
        let mut temp: [Zq; 256] = [ZERO; 256];

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
                let unity_root = unity_roots[k * d];
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
                let unity_root = unity_roots[k * d];
                for j in 0..d {
                    values[k * d + j] = temp[k * old_d + j] + unity_root * temp[k * old_d + j + d];
                    values[(k + n/2) * d + j] = temp[k * old_d + j] - unity_root * temp[k * old_d + j + d];
                }
            }
        }
        return values;
    }

    // Calculates the fourier representation of an element in R
    pub fn dft(r: &R) -> FourierReprR
    {
        FourierReprR {
            values: Self::fft(r.data.clone(), UNITY_ROOTS)
        }
    }

    // Calculates the element in R with the given fourier representation
    pub fn inv_dft(fourier_repr: &FourierReprR) -> R
    {
        let inv_256: Zq = ONE / Zq::from(256 as u16);
        let mut result = Self::fft(fourier_repr.values.clone(), INV_UNITY_ROOTS);
        for i in 0..256 {
            result[i] *= inv_256;
        }
        return R {
            data: result
        };
    }

    pub fn zero() -> FourierReprR
    {
        return FourierReprR {
            values: [ZERO; 256]
        }
    }

    pub fn add_product(&mut self, fst: &FourierReprR, snd: &FourierReprR) 
    {
        for i in 0..256 {
            self.values[i] += fst.values[i] * snd.values[i];
        }
    }
}

impl PartialEq for FourierReprR
{
    fn eq(&self, rhs: &FourierReprR) -> bool
    {
        (0..256).all(|i| self.values[i] == rhs.values[i])
    }
}

impl Eq for FourierReprR {}

impl<'a> Add<&'a FourierReprR> for FourierReprR
{
    type Output = FourierReprR;

    #[inline(always)]
    fn add(mut self, rhs: &'a FourierReprR) -> FourierReprR
    {
        self += rhs;
        return self;
    }
}

impl<'a> Add<FourierReprR> for &'a FourierReprR
{
    type Output = FourierReprR;

    #[inline(always)]
    fn add(self, mut rhs: FourierReprR) -> FourierReprR
    {
        rhs += self;
        return rhs;
    }
}

impl<'a> Mul<&'a FourierReprR> for FourierReprR
{
    type Output = FourierReprR;

    #[inline(always)]
    fn mul(mut self, rhs: &'a FourierReprR) -> FourierReprR
    {
        self *= rhs;
        return self;
    }
}

impl<'a> Mul<FourierReprR> for &'a FourierReprR
{
    type Output = FourierReprR;

    #[inline(always)]
    fn mul(self, mut rhs: FourierReprR) -> FourierReprR
    {
        rhs *= self;
        return rhs;
    }
}

impl Mul<Zq> for FourierReprR
{
    type Output = FourierReprR;

    #[inline(always)]
    fn mul(mut self, rhs: Zq) -> FourierReprR
    {
        self *= rhs;
        return self;
    }
}

impl<'a> Sub<&'a FourierReprR> for FourierReprR
{
    type Output = FourierReprR;

    #[inline(always)]
    fn sub(mut self, rhs: &'a FourierReprR) -> FourierReprR
    {
        self -= rhs;
        return self;
    }
}

impl<'a> Sub<FourierReprR> for &'a FourierReprR
{
    type Output = FourierReprR;

    #[inline(always)]
    fn sub(self, mut rhs: FourierReprR) -> FourierReprR
    {
        rhs -= self;
        rhs *= ZERO - ONE;
        return rhs;
    }
}

impl<'a> Div<&'a FourierReprR> for FourierReprR
{
    type Output = FourierReprR;

    #[inline(always)]
    fn div(mut self, rhs: &'a FourierReprR) -> FourierReprR
    {
        self /= rhs;
        return self;
    }
}

impl Div<Zq> for FourierReprR
{
    type Output = FourierReprR;

    #[inline(always)]
    fn div(mut self, rhs: Zq) -> FourierReprR
    {
        self /= rhs;
        return self;
    }
}

impl<'a> AddAssign<&'a FourierReprR> for FourierReprR
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a FourierReprR) {
        for i in 0..256 {
            self.values[i] += rhs.values[i];
        }
    }
}

impl<'a> SubAssign<&'a FourierReprR> for FourierReprR
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a FourierReprR) {
        for i in 0..256 {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl<'a> MulAssign<&'a FourierReprR> for FourierReprR
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a FourierReprR) {
        for i in 0..256 {
            self.values[i] *= rhs.values[i];
        }
    }
}

impl MulAssign<Zq> for FourierReprR
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Zq) {
        for i in 0..256 {
            self.values[i] *= rhs;
        }
    }
}

impl<'a> DivAssign<&'a FourierReprR> for FourierReprR
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: &'a FourierReprR) {
        for i in 0..256 {
            self.values[i] /= rhs.values[i];
        }
    }
}

impl DivAssign<Zq> for FourierReprR
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: Zq) {
        let inv = ONE / rhs;
        for i in 0..256 {
            self.values[i] *= inv;
        }
    }
}

impl<'a, T> From<&'a [T; 256]> for FourierReprR
    where Zq: From<T>, T: Copy
{
    fn from(value: &'a [T; 256]) -> FourierReprR {
        let mut values: [Zq; 256] = [ZERO; 256];
        for i in 0..256 {
            values[i] = Zq::from(value[i]);
        }
        return FourierReprR {
            values: values
        };
    }
}

impl Debug for FourierReprR
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[")?;
        (0..255).try_for_each(|i| write!(f, "{}, ", self.values[i]))?;
        write!(f, "{}]", self.values[255])?;
        return Ok(());
    }
}

#[cfg(test)]
const ELEMENT: [u16; 256] = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 81, 0, 1, 0, 0, 0, 0, 0, 
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
const DFT_ELEMENT: [u16; 256] = [5487, 7048, 1145, 6716, 88, 5957, 3742, 3441, 2663, 
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
fn test_fft(bencher: &mut test::Bencher) {
    let element = R::from(&ELEMENT);
    let expected_fourier_reprn = FourierReprR::from(&DFT_ELEMENT);
    bencher.iter(|| {
        let fourier_repr = FourierReprR::dft(&element);
        assert_eq!(expected_fourier_reprn, fourier_repr);
    });
}

#[test]
fn test_inv_fft() {
    let fourier_repr = FourierReprR::from(&DFT_ELEMENT);
    let expected_element = R::from(&ELEMENT);
    let element = FourierReprR::inv_dft(&fourier_repr);
    assert_eq!(expected_element, element);
}

#[test]
fn test_scalar_mul_div() {
    let mut element = R::from(&ELEMENT);
    let mut fourier_repr = FourierReprR::dft(&element);
    element *= Zq::from(653 as u16);
    fourier_repr *= Zq::from(653 as u16);
    assert_eq!(element, FourierReprR::inv_dft(&fourier_repr));

    element /= Zq::from(5321 as u16);
    fourier_repr /= Zq::from(5321 as u16);
    assert_eq!(element, FourierReprR::inv_dft(&fourier_repr));
}

#[test]
fn test_add_sub() {
    let mut element = R::from(&ELEMENT);
    let mut fourier_repr = FourierReprR::dft(&element);
    let base_element = element.clone();
    let base_fourier_repr = fourier_repr.clone();

    element += &base_element;
    fourier_repr += &base_fourier_repr;
    assert_eq!(element, FourierReprR::inv_dft(&fourier_repr));

    element -= &base_element;
    fourier_repr -= &base_fourier_repr;
    assert_eq!(element, FourierReprR::inv_dft(&fourier_repr));
    assert_eq!(R::from(&ELEMENT), element);
}