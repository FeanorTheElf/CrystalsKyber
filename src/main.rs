#![allow(dead_code, non_snake_case)]

mod unity_roots;

use unity_roots::*;

fn mul(a: u16, b: u16) -> u16
{
    ((a as u32) * (b as u32) % 7681) as u16
}

pub struct R
{
    data: [u16; 256]
}

pub struct FourierReprR
{
    values: [u16; 256]
}

impl FourierReprR
{
    fn fft(mut values: [u16; 256]) -> [u16; 256]
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
        let mut temp: [u16; 256] = [0; 256];

        // TODO: perform +, * in Z / 7681Z

        // values already contain the k=1 DFT of [x_j], so start with i = 7
        for ri in 0..4 {
            let mut i: usize = 2 * ri + 1;
            let mut n: usize = 1 << i;
            let mut d: usize = 1 << 8 - i;

            for k in 0..n/2 {
                let unity_root = FIRST_UNITY_ROOTS[i].pow(k as u32);
                for j in 0..d {
                    temp[k * d + j] = values[k * d + j] + unity_root * values[k * d + j + d]; 
                    temp[(k + n/2) * d + j] = values[k * d + j] - unity_root * values[k * d + j + d];
                }
            }

            n = n << 1;
            d = d >> 1;
            i = i + 1;
            for k in 0..n/2 {
                let unity_root = FIRST_UNITY_ROOTS[i].pow(k as u32);
                for j in 0..d {
                    values[k * d + j] = temp[k * d + j] + unity_root * temp[k * d + j + d];
                }
            }
        }
        return values;
    }

    fn fft4(mut values: [u16; 4]) -> [u16; 4]
    {

        let mut temp: [u16; 4] = [0; 4];

        for ri in 0..1 {
            let mut i: usize = 2 * ri + 1;
            let mut n: usize = 1 << i;
            let mut d: usize = 1 << 2 - i;

            for k in 0..n/2 {
                let unity_root = FIRST_UNITY_ROOTS[i].pow(k as u32);
                for j in 0..d {
                    temp[k * d + j] = values[k * d + j] + 7681 + mul(unity_root, values[k * d + j + d]) % 7681;
                    temp[(k + n/2) * d + j] = values[k * d + j] + 7681 - mul(unity_root, values[k * d + j + d]) % 7681;
                }
            }

            n = n << 1;
            d = d >> 1;
            i = i + 1;
            for k in 0..n/2 {
                let unity_root = FIRST_UNITY_ROOTS[i].pow(k as u32);
                for j in 0..d {
                    values[k * d + j] = temp[k * d + j] + 7681 + mul(unity_root, temp[k * d + j + d]) % 7681;
                    values[(k + n/2) * d + j] = temp[k * d + j] + 7681 - mul(unity_root, temp[k * d + j + d]) % 7681;
                }
            }
        }
        return values;
    }
}

fn main() {
    println!("{:?}", FourierReprR::fft4([1, 2, 3, 4]));
}
