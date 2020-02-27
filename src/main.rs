#![allow(dead_code)]
#![feature(test)]
#![feature(const_generics)]

extern crate test;
extern crate sha3;

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
use m::*;

use sha3::digest::{ ExtendableOutput, Input, XofReader };

type R = avx_r::R;
type FourierReprR = avx_r::FourierReprR;
type M = m::M<R>;
type Mat = m::Mat<R>;

const COMPRESSION_VECTOR: u16 = 11;
const COMPRESSION_ELEMENT: u16 = 3;

type PK = (CompressedM<COMPRESSION_VECTOR>, [u8; 32]);
type SK = M;
type Ciphertext = (CompressedM<COMPRESSION_VECTOR>, CompressedR<COMPRESSION_ELEMENT>);
type Message = [u8; 32];

fn enc(pk: &PK, plaintext: Message, enc_seed: [u8; 32]) -> Ciphertext
{
    let mut noise = noise(&enc_seed);
    let r = expand_error_distribution_vector(&mut noise);
    let e1 = expand_error_distribution_vector(&mut noise);
    let e2 = expand_error_distribution_element(&mut noise).dft();
    let a = expand_matrix(&pk.1);
    let u = a.transpose() * &r + &e1;
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

fn key_gen(matrix_seed: [u8; 32], secret_seed: [u8; 32]) -> (SK, PK)
{
    let a: Mat = expand_matrix(&matrix_seed);
    let mut noise = noise(&secret_seed);
    let s: M = expand_error_distribution_vector(&mut noise);
    let e: M = expand_error_distribution_vector(&mut noise);
    let b: M = &a * &s + &e;
    return (s, (b.compress(), matrix_seed));
}

fn uniform_zq<T: XofReader>(mut reader: T) -> impl Iterator<Item = Zq>
{
    let mut buffer: [u8; 2] = [0, 0];
    std::iter::repeat(()).map(move |_| {
        loop {
            reader.read(&mut buffer);
            let val: i16 = unsafe { std::mem::transmute::<_, i16>(buffer) } & 0x1FFF;
            debug_assert!(val >= 0);
            if val < Q as i16 {
                return Zq::from_perfect(val);
            }
        }
    })
}

fn centered_binomial_distribution(random: u8) -> Zq
{
    let mut value = (random << 4).count_ones() as i16 - (random >> 4).count_ones() as i16;
    if value < 0 {
        value += Q as i16;
    }
    return Zq::from_perfect(value);
}

fn expand_error_distribution_vector<T: XofReader>(reader: &mut T) -> M
{
    let mut buffer: [u8; N] = [0; N];
    let data = util::create_array(|_| {
        reader.read(&mut buffer);
        R::from(util::create_array(|i| centered_binomial_distribution(buffer[i]))).dft()
    });
    return M::from(data);
}

fn expand_error_distribution_element<T: XofReader>(reader: &mut T) -> R
{
    let mut buffer: [u8; N] = [0; N];
    reader.read(&mut buffer);
    return R::from(util::create_array(|i| centered_binomial_distribution(buffer[i])));
}

fn noise(seed: &[u8; 32]) -> sha3::Sha3XofReader
{
    let mut hasher = sha3::Shake256::default();
    hasher.input(&seed);
    return hasher.xof_result();
}

fn expand_matrix(seed: &[u8; 32]) -> Mat
{
    let mut hasher = sha3::Shake128::default();
    hasher.input(&seed);
    let mut iter = uniform_zq(hasher.xof_result());
    let data: [[FourierReprR; DIM]; DIM] = util::create_array(|_row| 
        util::create_array(|_col| {
            R::from(util::create_array_it(&mut iter)).dft()
        })
    );
    return Mat::from(data);
}

fn example()
{
    let matrix_seed = [186, 203, 37, 232, 216, 184, 94, 78, 3, 131, 61, 210, 236, 36, 7, 14, 175, 128, 72, 102, 223, 101, 60, 28, 157, 205, 28, 55, 135, 93, 19, 33];
    let secret_seed = [194, 76, 38, 216, 214, 43, 172, 134, 181, 97, 182, 181, 162, 190, 28, 151, 161, 129, 176, 109, 111, 12, 83, 58, 79, 220, 223, 207, 190, 191, 4, 98];
    let enc_seed = [221, 222, 74, 103, 3, 143, 117, 20, 254, 227, 59, 53, 154, 129, 5, 5, 237, 42, 84, 72, 172, 195, 156, 153, 99, 80, 43, 85, 166, 64, 137, 74];

    let (sk, pk) = key_gen(matrix_seed, secret_seed);
    let mut input: Message = [0; 32];
    input[0] = 1;
    input[2] = 0xF0;
    input[12] = 0x1A;
    let ciphertext = enc(&pk, input, enc_seed);
    let plaintext = dec(&sk, ciphertext);
    println!("{:?}", plaintext);
    println!("Ciphertext: {}", std::mem::size_of::<Ciphertext>());
    println!("Secret Key: {}", std::mem::size_of::<SK>());
    println!("Public Key: {}", std::mem::size_of::<PK>());
}

fn main() 
{
    let mut matrix_seed = [0; 32];
    matrix_seed[0] = 123;
    let mut secret_seed = [0; 32];
    secret_seed[1] = 41;
    let mut enc_seed = [0; 32];
    enc_seed[2] = 64;

    let (sk, pk) = key_gen(matrix_seed, secret_seed);
    let mut expected_message: Message = [0; 32];
    expected_message[0] = 1;
    expected_message[2] = 0xF0;
    expected_message[12] = 0x1A;
    let ciphertext = enc(&pk, expected_message, enc_seed);
    let message = dec(&sk, ciphertext);
    for i in 0..32 {
        assert!(expected_message[i] == message[i], "Expected messages to be the same, differ at index {}: {} != {}", i, expected_message[i], message[i]);
    }
}

#[bench]
fn bench_all(bencher: &mut test::Bencher) {
    let mut i = 0;
    bencher.iter(|| {
        let mut matrix_seed = [0; 32];
        matrix_seed[0] = i;
        let mut secret_seed = [0; 32];
        secret_seed[1] = i;
        let mut enc_seed = [0; 32];
        enc_seed[2] = i;

        let (sk, pk) = key_gen(matrix_seed, secret_seed);
        let mut expected_message: Message = [0; 32];
        expected_message[0] = 1;
        expected_message[2] = 0xF0;
        expected_message[12] = 0x1A;
        let ciphertext = enc(&pk, expected_message, enc_seed);
        let message = dec(&sk, ciphertext);
        for j in 0..32 {
            assert!(expected_message[j] == message[j], "Expected messages to be the same, differ at index {}: {} != {}", i, expected_message[j], message[j]);
        }
        i += 1;
    });
}