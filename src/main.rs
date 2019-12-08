#![allow(dead_code, non_snake_case)]
#![feature(test)]

extern crate test;
extern crate rand;

mod zq;
mod r;
mod m;

use zq::*;
use r::*;
use m::*;
use rand::prelude::*;

// Returns the element y in 0, ..., 2^d - 1 such
// that q/2^n * y is nearest to x.representative_pos()
fn compress_zq(x: Zq, d: u16) -> u16
{
    // this floating point approach always leads to the right result:
    // for each x, n, |0.5 - (x * n / 7681) mod 1| >= |0.5 - (x * 1 / 7681) mod 1|
    // >= |0.5 - (3840 / 7681) mod 1| >= 6.509569066531773E-5 > error in float * 7681
    let n = (1 << d) as f32;
    (x.representative_pos() as f32 * n / Q as f32).round() as u16 % (1 << d)
}

// Returns the element y of Zq for which
// y.representative_pos() is nearest to 2^d/q * x 
fn decompress_zq(x: u16, d: u16) -> Zq
{
    let n = (1 << d) as f32;
    Zq::from((x as f32 * Q as f32 / n).round() as u16)
}

// Calculates the values which are obtained
// by applying compress_zq to each coefficient
// of the given polynomial
fn compress(r: &R, d: u16) -> [u16; 256]
{
    let mut result: [u16; 256] = [0; 256];
    for i in 0..256 {
        result[i] = compress_zq(r.get_coefficient(i), d);
    }
    return result;
}

// Calculates the polynomial which coefficients
// are obtained by applying decompress_zq to the
// given values
fn decompress(x: &[u16; 256], d: u16) -> R
{
    let mut result: [Zq; 256] = [ZERO; 256];
    for i in 0..256 {
        result[i] = Zq::from(decompress_zq(x[i], d));
    }
    return R::from(result);
}

type PK = (M, Mat);
type SK = M;

fn sample_centered_binomial_distribution(random: u8) -> i8
{
    return (random << 4).count_ones() as i8 - (random >> 4).count_ones() as i8;
}

fn sample_r_centered_binomial_distribution<RNG>(rng: &mut RNG) -> R
    where RNG: FnMut() -> u32
{
    let mut data: [Zq; 256] = [ZERO; 256];
    for i in 0..64 {
        let random: u32 = rng();
        data[4*i] = Zq::from(sample_centered_binomial_distribution((random & 0xFF) as u8) as i16);
        data[4*i + 1] = Zq::from(sample_centered_binomial_distribution(((random >> 8) & 0xFF) as u8) as i16);
        data[4*i + 2] = Zq::from(sample_centered_binomial_distribution(((random >> 16) & 0xFF) as u8) as i16);
        data[4*i + 3] = Zq::from(sample_centered_binomial_distribution(((random >> 24) & 0xFF) as u8) as i16);
    }
    return R::from(data);
}

fn sample_m_centered_binomial_distribution<RNG>(rng: &mut RNG) -> M
    where RNG: FnMut() -> u32
{
    return M::from([
        FourierReprR::dft(&sample_r_centered_binomial_distribution(rng)),
        FourierReprR::dft(&sample_r_centered_binomial_distribution(rng)),
        FourierReprR::dft(&sample_r_centered_binomial_distribution(rng))
    ]);
}

fn enc<RNG>(pk: &PK, m: &[u16; 256], mut rng: RNG) -> (M, FourierReprR)
    where RNG: FnMut() -> u32
{
    let r = sample_m_centered_binomial_distribution(&mut rng);
    let e1 = sample_m_centered_binomial_distribution(&mut rng);
    let e2 = FourierReprR::dft(&sample_r_centered_binomial_distribution(&mut rng));
    let u = pk.1.transpose() * &r + &e1;
    let v = (&pk.0 * &r) + &e2 + &(FourierReprR::dft(&R::from(m)) * Zq::from(3840 as u16)); 
    return (u, v);
}

fn dec(sk: &SK, c: (M, FourierReprR)) -> [u16; 256]
{
    return compress(&FourierReprR::inv_dft(&(c.1 - &(sk * &c.0))), 1);
}

fn uniform_r(rng: &mut ThreadRng) -> FourierReprR
{
    let mut result: [Zq; 256] = [ZERO; 256];
    for i in 0..256 {
        result[i] = Zq::from(rng.gen_range(0, 7681) as u16);
    }
    return FourierReprR::dft(&R::from(result));
}

fn key_gen(rng: &mut ThreadRng) -> (SK, PK)
{
    let mut A_data: [[FourierReprR; 3]; 3] = [[FourierReprR::zero(), FourierReprR::zero(), FourierReprR::zero()],
        [FourierReprR::zero(), FourierReprR::zero(), FourierReprR::zero()],
        [FourierReprR::zero(), FourierReprR::zero(), FourierReprR::zero()]];
    for row in 0..3 {
        for col in 0..3 {
            A_data[row][col] = uniform_r(rng);
        }
    }
    let A = Mat::from(A_data);
    let s: M = sample_m_centered_binomial_distribution(&mut || rng.next_u32());
    let e: M = sample_m_centered_binomial_distribution(&mut || rng.next_u32());
    let b: M = &A * &s + &e;
    return (s, (b, A));
}

fn main() 
{
    let mut thread_rng = rand::thread_rng();
    let (sk, pk) = key_gen(&mut thread_rng);
    let mut expected_message = [0; 256];
    expected_message[0] = 1;
    expected_message[10] = 1;
    let ciphertext = enc(&pk, &expected_message, || thread_rng.next_u32());
    let message = dec(&sk, ciphertext);
    for i in 0..256 {
        print!("{}, ", message[i]);
    }
}

#[bench]
fn bench_all(bencher: &mut test::Bencher) {
    let mut thread_rng = rand::thread_rng();
    bencher.iter(|| {
        let (sk, pk) = key_gen(&mut thread_rng);
        let mut expected_message = [0; 256];
        expected_message[0] = 1;
        expected_message[10] = 1;
        let ciphertext = enc(&pk, &expected_message, || thread_rng.next_u32());
        let message = dec(&sk, ciphertext);
        for i in 0..256 {
            assert!(expected_message[i] == message[i], "Expected messages to be the same, differ at index {}: {} != {}", i, expected_message[i], message[i]);
        }
    });
}