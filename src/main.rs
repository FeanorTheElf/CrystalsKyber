#![allow(dead_code)]
#![feature(test)]
#![feature(const_generics)]

extern crate test;
extern crate rand;

#[macro_use]
mod util;
mod avx_util;

mod zq;
mod r;
mod m;

mod ref_r;
mod avx_zq;
mod avx_r;

use zq::*;
use r::*;
use m::CompressedM;
use rand::prelude::*;

type R = ref_r::R;
type FourierReprR = ref_r::FourierReprR;
type M = m::M<R>;
type Mat = m::Mat<R>;

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
        sample_r_centered_binomial_distribution(rng).dft(),
        sample_r_centered_binomial_distribution(rng).dft(),
        sample_r_centered_binomial_distribution(rng).dft()
    ]);
}

fn uniform_r(rng: &mut ThreadRng) -> FourierReprR
{
    let mut result: [Zq; 256] = [ZERO; 256];
    for i in 0..256 {
        result[i] = Zq::from(rng.gen_range(0, 7681) as i16);
    }
    return R::from(result).dft();
}

const COMPRESSION_VECTOR: u16 = 11;
const COMPRESSION_ELEMENT: u16 = 3;

type PK = (CompressedM<COMPRESSION_VECTOR>, Mat);
type SK = M;
type Ciphertext = (CompressedM<COMPRESSION_VECTOR>, CompressedR<COMPRESSION_ELEMENT>);
type Message = [u8; 32];

fn enc<RNG>(pk: &PK, plaintext: Message, mut rng: RNG) -> Ciphertext
    where RNG: FnMut() -> u32
{
    let r = sample_m_centered_binomial_distribution(&mut rng);
    let e1 = sample_m_centered_binomial_distribution(&mut rng);
    let e2 = sample_r_centered_binomial_distribution(&mut rng).dft();
    let u = pk.1.transpose() * &r + &e1;
    let t = M::decompress(&pk.0);
    let message = R::decompress(&CompressedR::from_data(plaintext));
    let v = (&t * &r) + &e2 + &message.dft(); 
    return (u.compress(), v.inv_dft().compress());
}

fn dec(sk: &SK, c: Ciphertext) -> Message
{
    let m = R::decompress(&c.1) - &(sk * &M::decompress(&c.0)).inv_dft();
    return m.compress().get_data();
}

fn key_gen(rng: &mut ThreadRng) -> (SK, PK)
{
    let mut a_data: [[FourierReprR; 3]; 3] = [
        [FourierReprR::zero(), FourierReprR::zero(), FourierReprR::zero()],
        [FourierReprR::zero(), FourierReprR::zero(), FourierReprR::zero()],
        [FourierReprR::zero(), FourierReprR::zero(), FourierReprR::zero()]];
    for row in 0..3 {
        for col in 0..3 {
            a_data[row][col] = uniform_r(rng);
        }
    }
    let a = Mat::from(a_data);
    let s: M = sample_m_centered_binomial_distribution(&mut || rng.next_u32());
    let e: M = sample_m_centered_binomial_distribution(&mut || rng.next_u32());
    let b: M = &a * &s + &e;
    return (s, (b.compress(), a));
}

fn main() 
{
    println!("{}", std::mem::size_of::<PK>());
}

#[bench]
fn bench_all(bencher: &mut test::Bencher) {
    let mut thread_rng = rand::thread_rng();
    bencher.iter(|| {
        let (sk, pk) = key_gen(&mut thread_rng);
        let mut expected_message: Message = [0; 32];
        expected_message[0] = 1;
        expected_message[10] = 1;
        let ciphertext = enc(&pk, expected_message.clone(), || thread_rng.next_u32());
        let message = dec(&sk, ciphertext);
        for i in 0..32 {
            assert!(expected_message[i] == message[i], "Expected messages to be the same, differ at index {}: {} != {}", i, expected_message[i], message[i]);
        }
    });
}